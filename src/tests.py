"""无需CIFAR数据的分段测试。填空前报NotImplementedError属于预期。

在 src 中运行 python tests.py，或从实验根目录运行 python src/tests.py。
分组：support、tensors、grad、layers、training、cnn、all。
layers/cnn 的结构断言验收任务书指定基线，后续自由调参可改变该基线。
"""

import argparse
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from torch.utils.data import TensorDataset
import module as s
import tensor as t
from tensor import check_grad
from data import DEFAULT_DATA_DIR, loader
from engine import build_model, evaluate, save_model, load_model


class Support(unittest.TestCase):
    def test_default_data_path(self):
        self.assertTrue(DEFAULT_DATA_DIR.is_absolute())
        self.assertEqual(DEFAULT_DATA_DIR, Path(__file__).resolve().parents[1] / 'data')

    def test_image_interface(self):
        x=torch.randint(0,256,(7,3,32,32),dtype=torch.uint8)
        y=torch.arange(7)
        images=x.float()/127.5-1
        self.assertEqual(images.flatten(1).shape,(7,3072))
        self.assertLessEqual(images.max().item(),1)
        self.assertGreaterEqual(images.min().item(),-1)
        self.assertEqual([len(a) for a,b in loader(TensorDataset(images,y),3)],[3,3,1])


class Tensors(unittest.TestCase):
    def test_teacher_pixel_batch(self):
        pixels = t.make_pixel_batch()
        self.assertEqual(pixels.dtype, torch.uint8)
        self.assertEqual(pixels.shape, (2, 3, 2, 2))
        self.assertEqual(pixels.reshape(-1).tolist(), list(range(24)))

    def test_preprocess_broadcast_and_noncontiguous(self):
        for batch, channels, height, width in ((1, 3, 32, 32), (3, 2, 4, 5)):
            pixels = (torch.arange(batch*channels*height*width) % 256).to(torch.uint8)
            pixels = pixels.reshape(batch, channels, height, width).transpose(2, 3)
            mean = torch.linspace(0.1, 0.6, channels)
            std = torch.linspace(0.2, 0.8, channels)
            saved = [x.clone() for x in (pixels, mean, std)]
            images = t.preprocess_pixels(pixels, mean, std)
            expected = torch.stack([(pixels[:, c].float()/255-mean[c])/std[c]
                                    for c in range(channels)], dim=1)
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(images.device, pixels.device)
            self.assertEqual(images.shape, pixels.shape)
            torch.testing.assert_close(images, expected)
            for current, before in zip((pixels, mean, std), saved):
                torch.testing.assert_close(current, before)

    def test_preprocess_default_endpoints(self):
        pixels = torch.tensor([[[[0, 64, 255]]]], dtype=torch.uint8)
        half = torch.tensor([0.5])
        images = t.preprocess_pixels(pixels, half, half)
        torch.testing.assert_close(images, pixels.float()/127.5-1)
        self.assertEqual(images[0, 0, 0, 0].item(), -1.)
        self.assertEqual(images[0, 0, 0, -1].item(), 1.)

    def test_flatten_preserves_sample_order(self):
        images = torch.arange(120, dtype=torch.float64).reshape(3, 2, 4, 5).transpose(2, 3)
        before = images.clone()
        flat = t.flatten_images(images)
        self.assertEqual(flat.shape, (3, 40))
        self.assertEqual(flat.dtype, images.dtype)
        self.assertEqual(flat.device, images.device)
        for i in range(3):
            torch.testing.assert_close(flat[i], images[i].reshape(-1))
        torch.testing.assert_close(images, before)

    def test_channel_statistics_axes(self):
        images = torch.tensor([
            [[[1., 3.]], [[10., 14.]], [[-3., 1.]]],
            [[[20., 24.]], [[4., 4.]], [[7., 13.]]],
        ], dtype=torch.float64)
        before = images.clone()
        first, mean, minimum, maximum = t.channel_statistics(images)
        self.assertEqual(first.shape, (2, 1, 1, 2))
        torch.testing.assert_close(first, images[:, :1])
        for result, values in ((mean, [12., 8., 4.5]), (minimum, [1., 4., -3.]),
                               (maximum, [24., 14., 13.])):
            self.assertEqual(result.shape, (3,))
            torch.testing.assert_close(result, torch.tensor(values, dtype=images.dtype))
        torch.testing.assert_close(images, before)

    def test_data_entry_calls_student_preprocess(self):
        import json
        from unittest.mock import patch
        from data import digest, load_raw_split, load_split

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pixels = (torch.arange(3*3*32*32) % 256).to(torch.uint8).reshape(3, 3, 32, 32)
            labels = torch.tensor([1, 2, 3])
            files = {}
            for split in ('train', 'valid', 'test'):
                torch.save(dict(images=pixels, labels=labels), path/f'{split}.pt')
                files[f'{split}.pt'] = digest(path/f'{split}.pt')
            (path/'manifest.json').write_text(json.dumps(dict(files=files)))
            raw, raw_labels = load_raw_split(path, 'train')
            torch.testing.assert_close(raw, pixels)
            torch.testing.assert_close(raw_labels, labels)
            for split in ('train', 'valid', 'test'):
                with patch.object(t, 'preprocess_pixels', wraps=t.preprocess_pixels) as called:
                    dataset = load_split(path, split)
                self.assertEqual(called.call_count, 1)
                images, targets = dataset.tensors
                torch.testing.assert_close(images, pixels.float()/127.5-1)
                torch.testing.assert_close(targets, labels)
                flat = t.flatten_images(images)
                output = t.project_batch(flat, torch.ones(128, 3072), torch.zeros(128))
                self.assertEqual(output.shape, (3, 128))
                self.assertEqual([len(x) for x, y in loader(dataset, 2)], [2, 1])

    def test_matrix_projection(self):
        for batch, dim, outputs in ((1, 3, 2), (4, 7, 5)):
            flat = torch.randn(dim, batch, dtype=torch.float64).T
            weight = torch.randn(dim, outputs, dtype=torch.float64).T
            bias = torch.randn(outputs, dtype=torch.float64)
            saved = [t.clone() for t in (flat, weight, bias)]
            actual = t.project_batch(flat, weight, bias)
            expected = torch.stack([torch.stack([(row*wk).sum()+bk for wk, bk in zip(weight, bias)]) for row in flat])
            self.assertEqual(actual.shape, (batch, outputs))
            torch.testing.assert_close(actual, expected)
            for current, old in zip((flat, weight, bias), saved):
                torch.testing.assert_close(current, old)


class Gradients(unittest.TestCase):
    def test_manual_gradient(self):
        check_grad()


class Layers(unittest.TestCase):
    def test_linear_values_and_gradients(self):
        torch.manual_seed(4)
        layer=s.MyLinear(4,5).double()
        x=torch.randn(2,4,3,dtype=torch.float64).transpose(1,2).requires_grad_()
        result=layer(x)
        reference=nn.functional.linear(x,layer.weight,layer.bias)
        self.assertEqual(result.shape,(2,3,5))
        torch.testing.assert_close(result,reference)
        actual=torch.autograd.grad(result.square().sum(),(x,layer.weight,layer.bias))
        expected=torch.autograd.grad(reference.square().sum(),(x,layer.weight,layer.bias))
        for a,e in zip(actual,expected,strict=True):torch.testing.assert_close(a,e)
        self.assertEqual(set(dict(layer.named_parameters())),{'weight','bias'})

    def test_relu_values_and_gradient_at_zero(self):
        x=torch.tensor([-2.,0.,3.],requires_grad=True)
        result=s.MyReLU()(x)
        torch.testing.assert_close(result,torch.tensor([0.,0.,3.]))
        result.sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([0.,0.,1.]))
        torch.testing.assert_close(x.detach(),torch.tensor([-2.,0.,3.]))

    def test_mlp_composition(self):
        model=s.ShallowMLP()
        self.assertEqual(sum(p.numel() for p in model.parameters()),394634)
        images=torch.randn(3,3,32,32)
        logits=model(images)
        self.assertEqual(logits.shape,(3,10))
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


class Training(unittest.TestCase):
    def test_train_step_clears_gradients_and_matches_update(self):
        torch.manual_seed(19)
        model = nn.Linear(3, 2).double()
        expected_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        expected_optimizer = torch.optim.SGD(expected_model.parameters(), lr=0.05)
        for _ in range(2):
            # 注入旧梯度，让漏掉 zero_grad 的实现不能通过。
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 7.0)
            images = torch.randn(5, 3, dtype=torch.float64)
            labels = torch.tensor([0, 1, 0, 1, 1])
            expected_optimizer.zero_grad()
            expected_loss = nn.functional.cross_entropy(expected_model(images), labels)
            expected_loss.backward()
            expected_optimizer.step()
            actual_loss = s.train_step(model, optimizer, images, labels)
            self.assertIsInstance(actual_loss, float)
            self.assertAlmostEqual(actual_loss, expected_loss.item(), places=10)
            for actual, expected in zip(model.parameters(), expected_model.parameters(), strict=True):
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual.grad, expected.grad)

    def test_ce_value_and_gradient(self):
        x=torch.tensor([[1000.,1001.,-1000.],[-999.,-1000.,-1001.]],requires_grad=True)
        y=torch.tensor([1,0])
        loss=s.cross_entropy(x,y)
        expected=nn.functional.cross_entropy(x,y)
        torch.testing.assert_close(loss,expected)
        torch.testing.assert_close(torch.autograd.grad(loss,x)[0],torch.autograd.grad(expected,x)[0])

    def test_train_step_and_weighted_validation(self):
        torch.manual_seed(8)
        model=nn.Linear(3,2)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        x=torch.randn(11,3)
        y=(x[:,0]>0).long()
        before=s.cross_entropy(model(x),y).item()
        for _ in range(30):self.assertIsInstance(s.train_step(model,optimizer,x,y),float)
        self.assertLess(s.cross_entropy(model(x),y).item(),before)
        saved=copy.deepcopy(model.state_dict())
        grads=[p.grad.clone() for p in model.parameters()]
        result=evaluate(model,loader(TensorDataset(x,y),4),'cpu')
        self.assertAlmostEqual(result['loss'],s.cross_entropy(model(x),y).item(),places=6)
        self.assertEqual(result['samples'],11)
        self.assertTrue(model.training)
        for k in saved:torch.testing.assert_close(saved[k],model.state_dict()[k])
        for old,p in zip(grads,model.parameters(),strict=True):torch.testing.assert_close(old,p.grad)

    def test_checkpoint(self):
        model=s.ShallowMLP().eval()
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'model.pt'
            save_model(file,model,'mlp','test-data',2)
            other,meta=load_model(file)
            images=torch.randn(2,3,32,32)
            torch.testing.assert_close(model(images),other(images))
            self.assertEqual(meta['epoch'],2)


class CNN(unittest.TestCase):
    def test_cnn_baseline_structure_and_intermediate_shapes(self):
        model = build_model('cnn')
        self.assertIsInstance(model, nn.Sequential)
        layers = [m for m in model.modules() if not list(m.children())]
        expected_types = (nn.Conv2d, nn.ReLU, nn.MaxPool2d,
                          nn.Conv2d, nn.ReLU, nn.MaxPool2d,
                          nn.Flatten, nn.Linear, nn.ReLU, nn.Linear)
        self.assertEqual(len(layers), len(expected_types))
        for layer, expected_type in zip(layers, expected_types, strict=True):
            self.assertIsInstance(layer, expected_type)
        for conv, channels in ((layers[0], (3, 16)), (layers[3], (16, 32))):
            self.assertEqual((conv.in_channels, conv.out_channels), channels)
            self.assertEqual(conv.kernel_size, (3, 3))
            self.assertEqual(conv.stride, (1, 1))
            self.assertEqual(conv.padding, (1, 1))
        for pool in (layers[2], layers[5]):
            self.assertIn(pool.kernel_size, (2, (2, 2)))
            self.assertIn(pool.stride, (2, (2, 2)))
        self.assertEqual((layers[7].in_features, layers[7].out_features), (2048, 64))
        self.assertEqual((layers[9].in_features, layers[9].out_features), (64, 10))
        shapes = []
        handles = [layer.register_forward_hook(
            lambda layer, inputs, output: shapes.append(tuple(output.shape))) for layer in layers]
        try:
            with torch.no_grad():
                model(torch.randn(2, 3, 32, 32))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes, [(2, 16, 32, 32), (2, 16, 32, 32), (2, 16, 16, 16),
                                  (2, 32, 16, 16), (2, 32, 16, 16), (2, 32, 8, 8),
                                  (2, 2048), (2, 64), (2, 64), (2, 10)])

    def test_cnn_shape_and_trainability(self):
        model=build_model('cnn')
        self.assertEqual(sum(p.numel() for p in model.parameters()),136874)
        self.assertEqual(sum(isinstance(m,nn.Conv2d) for m in model.modules()),2)
        x=torch.randn(2,3,32,32)
        result=model(x)
        self.assertEqual(result.shape,(2,10))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=('support','tensors','grad','layers','training','cnn','all'),default='all')
    a=p.parse_args()
    torch.set_num_threads(2)
    groups=dict(support=Support,tensors=Tensors,grad=Gradients,layers=Layers,training=Training,cnn=CNN)
    classes=groups.values() if a.part=='all' else [groups[a.part]]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in classes)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""任务一：四个张量函数（W0a–W0d）；任务二：手写梯度（W1）。

在 src 中运行 python tensor.py，或从实验根目录运行 python src/tensor.py。
--demo-grad 演示自动求导；--check-grad 检查手写梯度。
默认演示真实 CIFAR-10；--tiny 使用微型图片，--data 指定课程数据目录。
分段测试：python tests.py --part tensors / grad（分别运行）。
"""

import argparse
import torch


def make_pixel_batch():
    """已提供：用于手算检查的两张 RGB 微型图片，每张 2×2。"""
    return torch.arange(24, dtype=torch.uint8).reshape(2, 3, 2, 2)


def preprocess_pixels(pixels, mean, std):
    """任务一（W0a）：原始像素 → 网络输入，正式数据入口也调用此函数。

    pixels：uint8 [B,C,H,W]；mean/std：长度 C 的 float32 一维张量，
    与 pixels 在同一设备，std 均为正数（调用者保证）。
    将像素转 float32 并除以 255；将 mean/std 变为 [1,C,1,1]，
    通过广播完成 (images-mean)/std。返回同设备、同形状的 float32 张量。
    使用各通道 mean=std=0.5，映射到 [-1,1]；也应支持其他给定参数。
    不修改输入，不用 repeat，不硬编码 B/C/H/W；支持非连续输入。
    """
    raise NotImplementedError("W0a: preprocess_pixels")


def flatten_images(images):
    """任务一（W0b）：[B,C,H,W] → [B,C*H*W]，准备全连接层输入。

    保留 batch 维，每张图按 C/H/W 顺序展平；不混合不同样本。
    不改变 dtype/device 或输入；不硬编码维度，支持非连续张量。
    后续 ShallowMLP 中已提供的 nn.Flatten(start_dim=1) 执行同样的操作。
    """
    raise NotImplementedError("W0b: flatten_images")


def channel_statistics(images):
    """任务一（W0c）：检查预处理结果，返回 (first_channel, mean, minimum, maximum)。

    输入浮点 [B,C,H,W]，first_channel 切出全部图片的第 0 通道并保留维度
    [B,1,H,W]；其余三个结果均为 [C]，在 B/H/W 维上分别取均值、最小值、
    最大值。保留 dtype/device，不修改输入；
    """
    raise NotImplementedError("W0c: channel_statistics")


def project_batch(flat, weight, bias):
    """任务一（W0d）：flat [B,D]、weight [K,D]、bias [K]，返回[B,K]。
    每张图使用同一组权重：output[i,k]=sum_d flat[i,d]*weight[k,d]+bias[k]。
    用矩阵乘法、转置和广播完成；不逐样本循环，不调用nn.Linear/F.linear。
    支持不同B/D/K和非连续张量，不改变dtype、device或输入。
    真实图像演示中 D=3072、K=128，对应后续 MyLinear 第一层的计算。
    """
    raise NotImplementedError("W0d: project_batch")


def manual_gradients(x, w, b, target):
    """任务二（W1）：Y=XW+b，L=mean((Y-target)^2)，返回 dW、db、dX。
    X:[N,D]，W:[D,1]，b:[1]。只用张量运算，不调用 autograd。
    先算 G=2*(Y-target)/N，再根据链式法则反向传播。
    """
    raise NotImplementedError("W1: manual_gradients")


def check_grad():
    torch.manual_seed(7)
    x, w, b = [torch.randn(*shape, dtype=torch.float64) for shape in ((4, 3), (3, 1), (1,))]
    target = torch.randn(4, 1, dtype=torch.float64)
    actual = manual_gradients(x, w, b, target)
    xx, ww, bb = [t.clone().requires_grad_() for t in (x, w, b)]
    expected = torch.autograd.grad(((xx @ ww + bb - target)**2).mean(), (ww, bb, xx))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)
    eps = 1e-6
    plus, minus = w.clone(), w.clone()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    numeric = (((x @ plus+b-target)**2).mean()-((x @ minus+b-target)**2).mean())/(2*eps)
    torch.testing.assert_close(actual[0][0, 0], numeric)
    print("W1：手写梯度与 autograd、有限差分一致。")


def demo_tensors(data_dir=None, use_tiny=False):
    from data import DEFAULT_DATA_DIR, load_raw_split

    if use_tiny:
        pixels = make_pixel_batch()
        print("手算用微型图片")
    else:
        pixels, labels = load_raw_split(data_dir or DEFAULT_DATA_DIR, "train")
        pixels, labels = pixels[:4], labels[:4]
        print("真实 CIFAR-10 原始像素，标签：", labels.tolist())
    channels = pixels.shape[1]
    mean = torch.full((channels,), 0.5, device=pixels.device)
    std = torch.full((channels,), 0.5, device=pixels.device)
    images = preprocess_pixels(pixels, mean, std)
    flat = flatten_images(images)
    first, measured_mean, minimum, maximum = channel_statistics(images)
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(128, flat.shape[1], generator=generator) / flat.shape[1]**0.5
    bias = torch.zeros(128)
    output = project_batch(flat, weight, bias)
    print("原始像素：", pixels.shape, pixels.dtype)
    print("网络输入：", images.shape, images.dtype, images.device)
    print("MLP 展平：", flat.shape, "第一层投影：", output.shape)
    print("CNN 保留图像形状：", images.shape)
    print("第0通道：", first.shape)
    print("各通道实测均值 / 最小值 / 最大值：", measured_mean, minimum, maximum)
    print("第一张图投影的前8个数：", output[0, :8])
    print("固定归一化不保证实测均值为0；投影权重尚未训练，结果不是分类预测。")
    print("正式数据读取也调用 preprocess_pixels；两种模型共用同一份预处理。")


def demo_grad():
    w = torch.tensor(2., requires_grad=True)
    ((3*w-1)**2).backward()
    print("L=(3w-1)^2，w=2，梯度应为30：", w.grad.item())
    with torch.no_grad():
        w -= .01*w.grad
    w.grad = None
    print("一次手动 SGD 后 w：", w.item(), "清梯度后：", w.grad)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-grad", action="store_true")
    mode.add_argument("--demo-grad", action="store_true")
    p.add_argument("--tiny", action="store_true", help="任务一：使用教师提供的微型图片")
    p.add_argument("--data", help="任务一：课程 CIFAR-10 数据目录")
    args = p.parse_args()
    if args.check_grad:
        check_grad()
    elif args.demo_grad:
        demo_grad()
    else:
        demo_tensors(args.data, args.tiny)

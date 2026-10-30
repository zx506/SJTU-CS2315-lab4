"""教师提供：共用验证和保存加载，不需学生设计工程结构。"""

import torch
from module import ShallowMLP, make_cnn, batch_statistics


def build_model(kind):
    if kind == "mlp":
        return ShallowMLP()
    if kind == "cnn":
        return make_cnn()
    raise ValueError("model 应为 mlp/cnn")


@torch.no_grad()
def evaluate(model, batches, device):
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    try:
        for x, y in batches:
            loss, hits, n = batch_statistics(model(x.to(device)), y.to(device))
            loss_sum += loss
            correct += hits
            count += n
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("评估数据不能为空")
    return dict(loss=loss_sum/count, accuracy=correct/count, samples=count)


def save_model(path, model, kind, data_id, epoch):
    torch.save(dict(model=kind, state=model.state_dict(), data_id=data_id, epoch=epoch), path)


def load_model(path, device="cpu"):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(obj["model"])
    model.load_state_dict(obj["state"])
    return model.to(device).eval(), obj

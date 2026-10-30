"""任务三：MyLinear；任务四：MyReLU；M1：连接 ShallowMLP。
任务五：cross_entropy、train_step、batch_statistics；任务六：make_cnn。

只填写已有 TODO。构造器、数据接口及外层训练驱动已提供。
任务五需要自己编写损失与工作流，允许张量运算、autograd 和优化器基本方法。
"""

import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        """任务三（L1）：y=x@W.T+b，输入 [...,Din]，输出 [...,Dout]。
        使用矩阵乘法与广播，不调用 nn.Linear 或 F.linear。
        """
        raise NotImplementedError("L1: MyLinear.forward")


class MyReLU(nn.Module):
    def forward(self, x):
        """任务四（L2）：正数原样通过，非正数输出0。使用 torch.where，不调用内置relu。
        零点采用导数0，不做原地修改；反向传播交给 autograd。
        """
        raise NotImplementedError("L2: MyReLU.forward")


class ShallowMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)  # 教师提供图像展平接口
        self.hidden = MyLinear(3 * 32 * 32, 128)
        self.activation = MyReLU()
        self.output = MyLinear(128, 10)

    def forward(self, images):
        """M1：按 flatten、hidden、activation、output 的顺序调用。
        输入 [B,3,32,32]，返回 [B,10] logits；末尾不加 softmax。
        """
        raise NotImplementedError("M1: ShallowMLP.forward")


def cross_entropy(logits, targets):
    """任务五（T1）：实现平均交叉熵。形状/索引已给出，只补稳定公式。
    输入 [B,C] logits、[B] long labels，不调用 nn.CrossEntropyLoss/F.cross_entropy。
    使用张量运算保持计算图，让 autograd 计算梯度，末尾返回标量 Tensor。
    提示：每行 loss=log(sum(exp(shifted)))-target_scores。
    """
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    target_scores = shifted.gather(1, targets[:, None]).squeeze(1)  # noqa: F841 - 留给T1使用
    raise NotImplementedError("T1: cross_entropy")


def train_step(model, optimizer, images, labels):
    """任务五（T2）：清梯度、前向、调用上面的损失、反向、更新，返回 float loss。
    允许 optimizer.zero_grad()、loss.backward()、optimizer.step()。
    自行组织上述步骤，不需手写优化器或整个网络的 backward。
    外层驱动负责 model.train()、batch 循环和设备搬运。
    """
    raise NotImplementedError("T2: train_step")


def batch_statistics(logits, targets):
    """任务五（T3）：返回 (loss_sum, correct, count)，均为 Python 数值。
    使用上面的 cross_entropy，配合 argmax、比较和求和自行计算统计量。
    cross_entropy 返回均值，需换算成总损失；最后一批可能不足128个样本。
    """
    count = len(targets)  # noqa: F841 - 留给T3使用
    raise NotImplementedError("T3: batch_statistics")


def make_cnn():
    """任务六（C1）：用 nn.Sequential 按给定结构组装，全部使用内置层。
    Conv2d(3,16,3,padding=1), ReLU, MaxPool2d(2),
    Conv2d(16,32,3,padding=1), ReLU, MaxPool2d(2),
    Flatten, Linear(32*8*8,64), ReLU, Linear(64,10)。
    卷积步长使用默认值1。最后不加 softmax。返回 nn.Module。
    """
    raise NotImplementedError("C1: make_cnn")

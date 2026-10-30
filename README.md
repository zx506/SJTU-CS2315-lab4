# SJTU-CS2315 Lab 4：PyTorch

## 实验环境与运行位置

- **Python**：代码最低要求 Python 3.10，建议使用 Python 3.11 或 3.12。部分校验使用 `zip(..., strict=True)`，Python 3.9 及更早版本不支持。
- **PyTorch**：使用 PyTorch 2.x；本实验唯一直接依赖的第三方包是 `torch`。
- **设备**：默认使用 CPU，即可完成全部任务。MLP 与 CNN 的正式对照应使用同一 Python/PyTorch 环境、设备和训练设置。

### 1. 创建并激活虚拟环境

以下命令从仓库根目录执行，并假定已安装 Python 3.11；使用 Python 3.12 时，将命令中的版本号对应替换。

Linux / macOS（Bash 或 Zsh）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows（PowerShell，已安装 Python 启动器 `py`）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装 PyTorch

Linux / Windows 的 CPU 环境：

```bash
python -m pip install "torch>=2.0,<3.0" --index-url https://download.pytorch.org/whl/cpu
```

macOS：

```bash
python -m pip install "torch>=2.0,<3.0"
```

需要使用 GPU 时，请按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择与操作系统和驱动匹配的安装命令；安装时使用当前环境的 `python -m pip`。仅在下方自检确认 CUDA 可用后，才使用实验命令中的 `--device cuda`。

### 3. 自检并进入代码目录

在已激活的环境中，从仓库根目录执行：

```bash
python --version
python -c "import sys; assert sys.version_info >= (3, 10), '需要 Python 3.10 或更新版本'"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python src/tests.py --part support
```

保存版本输出，便于记录实验环境。CPU 环境下 `CUDA available: False` 属于正常结果。`support` 仅检查已提供的数据接口和路径；尚未完成填空时，其他测试组报 `NotImplementedError` 属于预期。

**后续各节的实验命令统一在 `src/` 目录执行：**

```bash
cd src
```

数据默认读取仓库根目录的 `data/`；后续命令中的 `runs/` 位于 `src/runs/`。每次重新打开终端，先从仓库根目录激活虚拟环境，再进入 `src/`。

## 一. 数据接口

CIFAR-10有10个类别，每张图片为32×32的RGB图像。官方提供50,000张训练图片和10,000张测试图片。 本实验从官方训练集按类别固定抽取10,000张训练、2,000张验证，二者不重叠；测试使用官方10,000张测试图片,见[数据来源与说明](https://www.cs.toronto.edu/~kriz/cifar.html)。
实验框架已经预先把数据预处理成方便读取的二进制文件，并且提供了文件读取接口。文件读取与数据切分已经提供；任务一将由你完成像素张量的归一化，并接入后续模型共用的数据入口。


你需要完成`src`文件夹中的 `tensor.py` 和 `module.py` 。

## 二. 基本张量计算

### 任务一：把 CIFAR-10 像素送进网络

在 `src/tensor.py` 中补齐 W0a–W0d。

```text
课程数据：uint8 [B,3,32,32]
    ↓ preprocess_pixels：类型转换、缩放、广播
网络输入：float32 [B,3,32,32]
    ├─ MLP：展平为 [B,3072] → 第一层线性投影 [B,128]
    └─ CNN：保留 [B,3,32,32] 的空间结构
```

| 填空 | 函数 | 需要自己完成的操作 | 输出 |
| --- | --- | --- | --- |
| W0a | `preprocess_pixels(pixels, mean, std)` | 转 float32、除以 255，将长度 C 的 mean/std 整理成 `[1,C,1,1]`，广播计算 `(images-mean)/std` | `[B,C,H,W]` |
| W0b | `flatten_images(images)` | 保留 batch 维，每张图按 C/H/W 顺序展平；兼容非连续张量 | `[B,C*H*W]` |
| W0c | `channel_statistics(images)` | 切出全部图片第 0 通道；沿 B/H/W 求每个通道的均值、最小值、最大值 | `[B,1,H,W]` 与三个 `[C]` |
| W0d | `project_batch(flat, weight, bias)` | 转置权重、矩阵乘法、广播加偏置 | `[B,D]`、`[K,D]`、`[K]` → `[B,K]` |

四个函数都不能修改输入，也不能硬编码 batch size 或图片尺寸。W0a 的 mean/std 由调用者提供，已经是同设备的 float32 张量，std 为正数；不要使用 repeat 复制它们。

本实验固定使用每个通道 `mean=0.5、std=0.5`，将像素从 `[0,255]` 映射到 `[-1,1]`。

**这段代码会被后续实验真正使用：** `data.py` 的 `load_split` 调用 W0a，训练集、验证集和测试集均使用相同的预处理规则。不要在模型中再归一化一次。W0b/W0d 在任务一的真实图片演示中连接输入与第一层计算：后续 MLP 已提供的 `nn.Flatten` 与 W0b 等价，`MyLinear` 的计算与 W0d 对应。后续任务的代码框架保持原样。

**正确性测试：** 在 `src` 目录运行：

```bash
python tests.py --part tensors
python tensor.py --tiny
python tensor.py
# 如课程数据存放于其他位置：
python tensor.py --data /path/to/course_data
```

`--tiny` 使用框架中提供的 `arange` 微型图片，便于手算；默认演示读取课程 CIFAR-10 训练数据中的前 4 张图片，打印原始像素、网络输入、展平和第一层投影的形状，以及各通道的实测统计量。


### 任务二：手写一个线性计算的梯度

在`tensor.py`中填写 `manual_gradients`函数。**W1**：对线性计算 `Y=XW+b`、`L=mean((Y-target)²)` ，用张量操作手写 `dW、db、dX`。

测试：

```bash
python tensor.py --check-grad
python tests.py --part grad
```


## 三.构建神经网络

### 任务三：手动实现全连接层MyLinear

在`module.py`中补充`MyLinear`类的实现。用张量操作直接实现`forward`方法

### 任务四：手动实现MyReLU

在`module.py`中补充`MyReLU`类的实现。

### M1：ShallowMLP

按顺序调用已经创建的子模块：

| 步骤 | 输出 |
| --- | --- |
| Flatten（已提供） | `[B,3072]` |
| MyLinear(3072,128) | `[B,128]` |
| MyReLU | `[B,128]` |
| MyLinear(128,10) | `[B,10]` |

测试代码：

```bash
python tests.py --part layers
```

## 四. 训练与验证

### 任务五：完成工作流模块

在 `module.py` 中自行实现损失计算、训练步和统计逻辑。各函数允许使用的接口如下：

- `cross_entropy`：用张量运算实现平均交叉熵，返回保留计算图的标量 Tensor；**禁止直接调用 `nn.CrossEntropyLoss` 或 `torch.nn.functional.cross_entropy`**。
- `train_step`：自行组织一个训练步，并调用自己实现的 `cross_entropy`；**允许使用自动求导，以及 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`**。不要求手写优化器或完整网络的反向传播。
- `batch_statistics`：调用自己实现的 `cross_entropy`，允许使用 `argmax`、比较、求和等张量操作，自行计算并返回当前 batch 的总损失、正确数和样本数，三者均为 Python 数值。

以上接口许可针对任务五；其他任务仍遵循各自函数注释中的要求，例如任务二 W1 必须手写梯度，不能调用 autograd。

测试代码：

```bash
python tests.py --part training
python train.py --model mlp --out runs/mlp
```

默认用CPU训练5轮，batch size=128，内置Adam，lr=0.001，seed=7。使用GPU时加 `--device cuda`；两次对照使用相同设备和配置。默认数据路径为 `../data/`，

输出 `metrics.csv`、`run_info.json`、`best.pt` 和 `last.pt`。epoch=0是训练前评估；每轮结束后重新评估训练集与验证集，二者都有loss和准确率。best按验证loss选取，last保存最后一轮。每次实验换一个输出目录，避免覆盖记录。

## 五. 卷积神经网络

### 任务六：搭建卷积神经网络

在`module.py`中，填写`make_cnn()`函数。 使用PyTorch的网络层，搭建以下卷积神经网络：

| 步骤 | 输出形状 |
| --- | --- |
| Conv2d(3,16,3,padding=1) + ReLU | `[B,16,32,32]` |
| MaxPool2d(2) | `[B,16,16,16]` |
| Conv2d(16,32,3,padding=1) + ReLU | `[B,32,16,16]` |
| MaxPool2d(2) | `[B,32,8,8]` |
| Flatten | `[B,2048]` |
| Linear(2048,64) + ReLU | `[B,64]` |
| Linear(64,10) | `[B,10]` |

测试代码：

```bash
python tests.py --part cnn
python tests.py --part all
python train.py --model cnn --out runs/cnn
python compare.py runs/mlp runs/cnn --out runs/comparison.html
```

用浏览器打开comparison.html查看你搭建的两种模型的训练/验证曲线与指标表。比较器会检查数据、轮数、batch、学习率、优化器、seed和设备是否相同。


你可以调试模型的结构和参数，然后加载两种模型中最佳的checkpoint来评估测试集：

```bash
python infer.py runs/mlp/best.pt --out runs/mlp/test.json
python infer.py runs/cnn/best.pt --out runs/cnn/test.json
```


## 提交与讨论

提交：`tensor.py`、`module.py`、和你的实验报告。


参考：[CIFAR-10数据集](https://www.cs.toronto.edu/~kriz/cifar.html)、[PyTorch图像分类教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)、[nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。

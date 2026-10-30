"""教师提供：读取共享数据。读取与切分已提供；归一化调用任务一的学生函数。"""

import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_split(directory, split):
    """读取已切分的 uint8 像素与 long 标签，不进行归一化。"""
    if split not in ("train", "valid", "test"):
        raise ValueError("split 应为 train/valid/test")
    directory = Path(directory)
    if not (directory/"manifest.json").is_file():
        raise FileNotFoundError("缺少课程数据。请将 --data 指向教师准备的 CIFAR-10 数据目录。")
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    path = directory/f"{split}.pt"
    if digest(path) != manifest["files"][path.name]:
        raise ValueError(f"{path} 校验失败，请使用完整的课程数据")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    x, y = obj["images"], obj["labels"]
    if x.dtype != torch.uint8 or x.shape != (len(y), 3, 32, 32) or y.dtype != torch.long:
        raise ValueError("图像或标签格式不正确")
    if len(y) == 0 or y.min() < 0 or y.max() > 9:
        raise ValueError("类别必须在0到9之间，且数据不能为空")
    return x, y


def load_split(directory, split):
    """任务一接入点：训练、验证和推理统一使用学生完成的归一化。"""
    from tensor import preprocess_pixels

    x, y = load_raw_split(directory, split)
    mean = torch.full((x.shape[1],), 0.5, dtype=torch.float32, device=x.device)
    std = torch.full_like(mean, 0.5)
    return TensorDataset(preprocess_pixels(x, mean, std), y)


def loader(dataset, batch_size=128, shuffle=False, seed=7):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))

"""冻结实验选择后，独立进程加载模型并评估官方测试集。

在 src 中运行 python infer.py，或从实验根目录运行 python src/infer.py。
默认数据为实验根目录的 data；显式相对路径按当前工作目录解释。
"""

import argparse
import json
from pathlib import Path
import torch
from data import DEFAULT_DATA_DIR, digest, load_split, loader
from engine import load_model, evaluate


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', help='checkpoint 路径，相对路径按当前工作目录解释')
    p.add_argument('--data', default=str(DEFAULT_DATA_DIR),
                   help='数据目录，默认使用本实验 data/，不受当前工作目录影响')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', required=True, help='测试记录路径，相对路径按当前工作目录解释')
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('输出文件已存在，请不要覆盖测试记录')
    torch.set_num_threads(2)
    model, saved = load_model(a.checkpoint, a.device)
    if saved['data_id'] != digest(Path(a.data)/'manifest.json'):
        raise ValueError('数据身份与checkpoint不一致')
    result = evaluate(model, loader(load_split(a.data, 'test')), a.device)
    result.update(model=saved['model'], epoch=saved['epoch'], data_id=saved['data_id'], split='official test')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

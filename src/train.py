"""完成 module.py 的相应填空后运行。默认5轮、Adam、lr=0.001。

在 src 中运行 python train.py，或从实验根目录运行 python src/train.py。
默认数据为实验根目录的 data；显式传入的相对路径按当前工作目录解释。
"""

import argparse
import csv
import json
from pathlib import Path
import time
import torch
from data import DEFAULT_DATA_DIR, digest, load_split, loader
from engine import build_model, evaluate, save_model
from module import train_step


def run(kind, data, out, epochs=5, batch_size=128, lr=.001, device="cpu", seed=7):
    if epochs < 1 or batch_size < 1 or not 0 < lr < float("inf"):
        raise ValueError("epochs、batch_size、lr必须为有限正数")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("输出目录非空，请换一个 --out")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    train, valid = load_split(data, "train"), load_split(data, "valid")
    data_id = digest(Path(data)/"manifest.json")
    train_batches = loader(train, batch_size, True, seed)
    train_eval, valid_batches = loader(train, batch_size), loader(valid, batch_size)
    model = build_model(kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    out.mkdir(parents=True, exist_ok=True)
    rows, best, best_epoch = [], float("inf"), 0
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(epochs+1):
        if epoch:
            model.train()
            for x, y in train_batches:
                loss = train_step(model, optimizer, x.to(device), y.to(device))
                if not torch.isfinite(torch.tensor(loss)):
                    raise FloatingPointError("训练损失非有限，请检查损失、梯度和学习率")
        tr, va = evaluate(model, train_eval, device), evaluate(model, valid_batches, device)
        row = dict(epoch=epoch, train_loss=tr['loss'], valid_loss=va['loss'],
                   train_accuracy=tr['accuracy'], valid_accuracy=va['accuracy'])
        rows.append(row)
        if va['loss'] < best:
            best, best_epoch = va['loss'], epoch
            save_model(out/'best.pt', model, kind, data_id, epoch)
        with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{kind} epoch={epoch} train_loss={tr['loss']:.4f} valid_loss={va['loss']:.4f} valid_acc={va['accuracy']:.3f}", flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-start
    save_model(out/'last.pt', model, kind, data_id, epochs)
    info = dict(model=kind, epochs=epochs, batch_size=batch_size, lr=lr, optimizer='Adam', seed=seed,
                train_samples=len(train), valid_samples=len(valid), data_id=data_id,
                parameters=sum(p.numel() for p in model.parameters()), seconds=seconds,
                device=str(device), torch=str(torch.__version__), cpu_threads=2,
                timing='epoch loop including train/valid evaluation, metrics writes and best checkpoint saves; excludes data/model setup and last checkpoint export',
                best_epoch=best_epoch, best_valid_loss=best, final=rows[-1])
    (out/'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('mlp', 'cnn'), required=True)
    p.add_argument('--data', default=str(DEFAULT_DATA_DIR),
                   help='数据目录，默认使用本实验 data/，不受当前工作目录影响')
    p.add_argument('--out', required=True, help='输出目录，相对路径按当前工作目录解释')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    run(a.model, a.data, a.out, a.epochs, a.batch_size, a.lr, a.device, a.seed)

"""教师提供：比较同设置的两次运行，输出HTML曲线与指标表。无需绘图库。"""

import argparse
import csv
import html
import json
from pathlib import Path


def compare(first, second, output):
    runs = []
    for path in (Path(first), Path(second)):
        info = json.loads((path/'run_info.json').read_text(encoding='utf-8'))
        with (path/'metrics.csv').open(encoding='utf-8') as f:
            rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        runs.append((info, rows))
    for key in ('data_id', 'epochs', 'batch_size', 'lr', 'optimizer', 'seed', 'device'):
        if runs[0][0][key] != runs[1][0][key]:
            raise ValueError(f'对照设置不一致：{key}')
    if {i['model'] for i, _ in runs} != {'mlp', 'cnn'}:
        raise ValueError('请选择mlp和cnn两种模型的运行目录')
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>CIFAR-10 模型对照</title>',
             '<style>body{font:18px system-ui;max-width:960px;margin:40px auto;color:#142e40}td,th{padding:12px;text-align:left;border-bottom:1px solid #ccd}svg{width:100%}</style>',
             '<h1>CIFAR-10 模型对照</h1><p>相同训练设置，不代表相同计算量。指标为最后一轮验证结果；测试结果见独立 test.json。</p>',
             '<table><tr><th>模型</th><th>参数</th><th>验证CE</th><th>验证准确率</th><th>轮内总秒数</th></tr>']
    for i, _ in runs:
        parts.append(f"<tr><td>{html.escape(i['model'])}</td><td>{i['parameters']}</td><td>{i['final']['valid_loss']:.4f}</td><td>{i['final']['valid_accuracy']:.1%}</td><td>{i['seconds']:.2f}</td></tr>")
    parts.append('</table>')
    for metric, title in [('loss','交叉熵'), ('accuracy','准确率')]:
        maximum = 1 if metric == 'accuracy' else max(r[k+'_'+metric] for _, rs in runs for r in rs for k in ('train','valid'))*1.1
        epochs = runs[0][0]['epochs']
        parts.append(f'<h2>{title}</h2><svg viewBox="0 0 900 410"><path d="M70 20 V330 H860" fill="none" stroke="#888"/>')
        for f in (0,.25,.5,.75,1):
            parts.append(f'<text x="60" y="{335-f*310}" text-anchor="end" font-size="14">{f*maximum:.2f}</text>')
        for epoch in range(epochs+1):
            parts.append(f'<text x="{70+790*epoch/epochs}" y="355" text-anchor="middle" font-size="14">{epoch}</text>')
        n = 0
        for j,(info,rows) in enumerate(runs):
            for split in ('train','valid'):
                color = ['#007c83','#d94841'][j]
                dash = ' stroke-dasharray="6 4"' if split=='train' else ''
                coords = ' '.join(f"{70+790*r['epoch']/epochs},{330-310*r[split+'_'+metric]/maximum}" for r in rows)
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/><text x="{70+n*190}" y="395" fill="{color}" font-size="16">{info["model"]} {split}</text>')
                n += 1
        parts.append('<text x="830" y="375" font-size="14">epoch</text></svg>')
    parts.append('<p>train为每轮结束后重新评估的训练集指标。实线valid，虚线train。计时包括训练、两集评估、日志和best保存，排除数据/模型初始化和last导出。</p></html>')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(parts),encoding='utf-8')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mlp')
    p.add_argument('cnn')
    p.add_argument('--out',default='runs/comparison.html')
    a=p.parse_args()
    compare(a.mlp,a.cnn,a.out)

"""360D mean/std에서 pos120 인덱스만 추출"""
import torch

POS120_IDX = list(range(0,30)) + list(range(90,135)) + list(range(225,270))

datasets = [
    '/home/user/Projects/research/SOKE/data/data360/How2Sign',
    '/home/user/Projects/research/SOKE/data/data360/CSL-Daily',
    '/home/user/Projects/research/SOKE/data/data360/Phoenix_2014T',
]

for d in datasets:
    for stat in ['mean', 'std']:
        src = f'{d}/{stat}_360.pt'
        dst = f'{d}/{stat}_pos120.pt'
        try:
            t = torch.load(src, map_location='cpu')
            torch.save(t[POS120_IDX], dst)
            print(f'✅ {dst}')
        except Exception as e:
            print(f'⚠ {src}: {e}')

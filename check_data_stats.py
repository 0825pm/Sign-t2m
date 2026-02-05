"""
SOKE 133-dim 전체 데이터 통계 분석 (H2S + CSL + Phoenix)
Usage: cd ~/Projects/research/sign-t2m && python check_data_stats.py
"""
import torch, numpy as np, sys, os, pandas as pd
sys.path.insert(0, '.')
from src.data.signlang.load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample

SOKE_DIR = '/home/user/Projects/research/SOKE/data'
MAX_SAMPLES = 99999999

parts = {
    'upper_body': (0, 30),
    'lhand': (30, 75),
    'rhand': (75, 120),
    'jaw': (120, 123),
    'expr': (123, 133),
}

dim_names = []
for i in range(30):
    dim_names.append("body_%d" % i)
for i in range(45):
    dim_names.append("lhand_%d" % i)
for i in range(45):
    dim_names.append("rhand_%d" % i)
for i in range(3):
    dim_names.append("jaw_%d" % i)
for i in range(10):
    dim_names.append("expr_%d" % i)


def load_dataset(name):
    all_raw = []
    lengths = []
    skip = 0

    if name == 'h2s':
        data_root = os.path.join(SOKE_DIR, 'How2Sign')
        for split in ['train', 'val', 'test']:
            csv_path = os.path.join(data_root, split, 're_aligned',
                                    'how2sign_realigned_%s_preprocessed_fps.csv' % split)
            if not os.path.exists(csv_path):
                print("  [%s] CSV not found: %s" % (split, csv_path))
                continue
            df = pd.read_csv(csv_path)
            count = 0
            for i in range(len(df)):
                if len(all_raw) >= MAX_SAMPLES:
                    break
                ann = {'name': df.iloc[i]['SENTENCE_NAME'], 'text': '', 'split': split}
                motion, _, _, _ = load_h2s_sample(ann, data_root)
                if motion is not None and motion.shape[1] == 133:
                    all_raw.append(motion)
                    lengths.append(motion.shape[0])
                    count += 1
                else:
                    skip += 1
            print("  [%s] loaded %d samples" % (split, count))

    elif name == 'csl':
        data_root = os.path.join(SOKE_DIR, 'CSL-Daily')
        # CSL split file 자동 탐색
        for split in ['train', 'val', 'test']:
            if len(all_raw) >= MAX_SAMPLES:
                break
            loaded = False
            # csv 시도
            for csv_try in [
                os.path.join(data_root, split + '.csv'),
                os.path.join(data_root, 'splits', split + '.csv'),
            ]:
                if os.path.exists(csv_try):
                    df = pd.read_csv(csv_try)
                    name_col = None
                    for col in ['name', 'SENTENCE_NAME', 'NAME', 'video_name', 'id']:
                        if col in df.columns:
                            name_col = col
                            break
                    if name_col is None:
                        print("  [%s] cols=%s (no name col)" % (split, list(df.columns)))
                        break
                    count = 0
                    for i in range(len(df)):
                        if len(all_raw) >= MAX_SAMPLES:
                            break
                        ann = {'name': df.iloc[i][name_col], 'text': str(df.iloc[i].get('text', '')), 'split': split}
                        motion, _, _, _ = load_csl_sample(ann, data_root)
                        if motion is not None and motion.shape[1] == 133:
                            all_raw.append(motion)
                            lengths.append(motion.shape[0])
                            count += 1
                        else:
                            skip += 1
                    print("  [%s] loaded %d samples" % (split, count))
                    loaded = True
                    break
            if loaded:
                continue
            # txt 시도
            txt_path = os.path.join(data_root, split + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    names_list = [l.strip() for l in f if l.strip()]
                count = 0
                for n in names_list:
                    if len(all_raw) >= MAX_SAMPLES:
                        break
                    ann = {'name': n, 'text': '', 'split': split}
                    motion, _, _, _ = load_csl_sample(ann, data_root)
                    if motion is not None and motion.shape[1] == 133:
                        all_raw.append(motion)
                        lengths.append(motion.shape[0])
                        count += 1
                    else:
                        skip += 1
                print("  [%s] loaded %d samples (txt)" % (split, count))
                continue
            # poses 디렉토리에서 직접 로드
            poses_dir = os.path.join(data_root, 'poses')
            if os.path.exists(poses_dir):
                dirs = sorted(os.listdir(poses_dir))[:MAX_SAMPLES]
                count = 0
                for n in dirs:
                    if len(all_raw) >= MAX_SAMPLES:
                        break
                    ann = {'name': n, 'text': '', 'split': split}
                    motion, _, _, _ = load_csl_sample(ann, data_root)
                    if motion is not None and motion.shape[1] == 133:
                        all_raw.append(motion)
                        lengths.append(motion.shape[0])
                        count += 1
                    else:
                        skip += 1
                print("  [%s] loaded %d from poses dir" % (split, count))

    elif name == 'phoenix':
        data_root = os.path.join(SOKE_DIR, 'Phoenix_2014T')
        for split in ['train', 'val', 'test']:
            if len(all_raw) >= MAX_SAMPLES:
                break
            loaded = False
            for csv_try in [
                os.path.join(data_root, split + '.csv'),
                os.path.join(data_root, 'splits', split + '.csv'),
            ]:
                if os.path.exists(csv_try):
                    df = pd.read_csv(csv_try)
                    name_col = None
                    for col in ['name', 'SENTENCE_NAME', 'NAME', 'video_name', 'id']:
                        if col in df.columns:
                            name_col = col
                            break
                    if name_col is None:
                        print("  [%s] cols=%s" % (split, list(df.columns)))
                        break
                    count = 0
                    for i in range(len(df)):
                        if len(all_raw) >= MAX_SAMPLES:
                            break
                        ann = {'name': df.iloc[i][name_col], 'text': str(df.iloc[i].get('text', '')), 'split': split}
                        motion, _, _, _ = load_phoenix_sample(ann, data_root)
                        if motion is not None and motion.shape[1] == 133:
                            all_raw.append(motion)
                            lengths.append(motion.shape[0])
                            count += 1
                        else:
                            skip += 1
                    print("  [%s] loaded %d samples" % (split, count))
                    loaded = True
                    break
            if loaded:
                continue
            txt_path = os.path.join(data_root, split + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    names_list = [l.strip() for l in f if l.strip()]
                count = 0
                for n in names_list:
                    if len(all_raw) >= MAX_SAMPLES:
                        break
                    ann = {'name': n, 'text': '', 'split': split}
                    motion, _, _, _ = load_phoenix_sample(ann, data_root)
                    if motion is not None and motion.shape[1] == 133:
                        all_raw.append(motion)
                        lengths.append(motion.shape[0])
                        count += 1
                    else:
                        skip += 1
                print("  [%s] loaded %d samples (txt)" % (split, count))
                continue
            poses_dir = os.path.join(data_root, 'poses')
            if os.path.exists(poses_dir):
                dirs = sorted(os.listdir(poses_dir))[:MAX_SAMPLES]
                count = 0
                for n in dirs:
                    if len(all_raw) >= MAX_SAMPLES:
                        break
                    ann = {'name': n, 'text': '', 'split': split}
                    motion, _, _, _ = load_phoenix_sample(ann, data_root)
                    if motion is not None and motion.shape[1] == 133:
                        all_raw.append(motion)
                        lengths.append(motion.shape[0])
                        count += 1
                    else:
                        skip += 1
                print("  [%s] loaded %d from poses dir" % (split, count))

    return all_raw, lengths, skip


def analyze_dataset(ds_name, all_raw, lengths, use_mean, use_std):
    if len(all_raw) == 0:
        print("  데이터 없음!")
        return None

    raw = np.concatenate(all_raw, axis=0)
    print("  Samples=%d, Frames=%d" % (len(all_raw), raw.shape[0]))
    print("  Length: min=%d, max=%d, mean=%.1f, median=%.1f" % (
        min(lengths), max(lengths), np.mean(lengths), np.median(lengths)))

    # Raw 통계
    print("\n  [Raw 전체]")
    print("  mean=%.6f, std=%.6f, range=%.4f ~ %.4f" % (raw.mean(), raw.std(), raw.min(), raw.max()))
    print("  NaN=%d, Inf=%d" % (np.isnan(raw).sum(), np.isinf(raw).sum()))

    # Part별 raw
    print("\n  [Part별 Raw]")
    for pname, (s, e) in parts.items():
        p = raw[:, s:e]
        print("  %-12s: mean=%10.6f  std=%10.6f  min=%10.4f  max=%10.4f" % (
            pname, p.mean(), p.std(), p.min(), p.max()))

    # Dim별 이상치
    print("\n  [Dim 이상치]")
    flagged = []
    for d in range(133):
        col = raw[:, d]
        zero_pct = (np.abs(col) < 1e-8).mean() * 100
        flag = ''
        if col.std() < 1e-4:
            flag = 'DEAD'
        elif np.abs(col).max() > 10:
            flag = 'OUTLIER'
        elif zero_pct > 90:
            flag = 'SPARSE'
        if flag:
            flagged.append((d, dim_names[d], flag, col.mean(), col.std(), col.min(), col.max(), zero_pct))
    if flagged:
        print("  %4s %-10s %-8s %10s %10s %10s %10s %7s" % (
            "Dim", "Name", "Flag", "Mean", "Std", "Min", "Max", "Zero%"))
        for d, dn, fl, mn, sd, mi, mx, zp in flagged:
            print("  %4d %-10s %-8s %10.4f %10.6f %10.4f %10.4f %6.1f%%" % (
                d, dn, fl, mn, sd, mi, mx, zp))
    else:
        print("  없음 (모두 정상)")

    # Frame norm
    print("\n  [Frame norm]")
    frame_norms = np.linalg.norm(raw, axis=1)
    print("  mean=%.4f, std=%.4f, range=%.4f ~ %.4f" % (
        frame_norms.mean(), frame_norms.std(), frame_norms.min(), frame_norms.max()))
    for sigma in [3, 5, 10]:
        thr = frame_norms.mean() + sigma * frame_norms.std()
        cnt = (frame_norms > thr).sum()
        print("    %d-sigma: %d frames (%.3f%%)" % (sigma, cnt, cnt / len(frame_norms) * 100))

    # Velocity
    print("\n  [Velocity]")
    big_jumps = 0
    total_trans = 0
    for motion in all_raw[:min(100, len(all_raw))]:
        vel = np.diff(motion, axis=0)
        vel_norm = np.linalg.norm(vel, axis=1)
        big_jumps += (vel_norm > 1.0).sum()
        total_trans += len(vel_norm)
    if total_trans > 0:
        print("  vel > 1.0: %d / %d (%.2f%%)" % (big_jumps, total_trans, big_jumps / total_trans * 100))

    # vs normalization mean/std
    if use_mean is not None and use_std is not None:
        data_mean = raw.mean(axis=0)
        data_std = raw.std(axis=0)
        print("\n  [Data vs Norm mean/std]")
        print("  %12s %10s %10s %10s | %10s %10s %10s" % (
            "Part", "DataMean", "NormMean", "Diff", "DataStd", "NormStd", "Ratio"))
        for pname, (s, e) in parts.items():
            dm = data_mean[s:e].mean()
            cm = use_mean[s:e].mean()
            ds = data_std[s:e].mean()
            cs = use_std[s:e].mean()
            ratio = ds / (cs + 1e-10)
            warn = ""
            if abs(dm - cm) > 0.5:
                warn += " <<MEAN_MISMATCH>>"
            if ratio > 3.0 or ratio < 0.3:
                warn += " <<STD_MISMATCH>>"
            print("  %12s %10.4f %10.4f %10.4f | %10.4f %10.4f %10.4f%s" % (
                pname, dm, cm, dm - cm, ds, cs, ratio, warn))

        # 정규화 후
        print("\n  [정규화 후]")
        normalized = (raw - use_mean) / (use_std + 1e-8)
        for pname, (s, e) in parts.items():
            p = normalized[:, s:e]
            over5 = (np.abs(p) > 5).mean() * 100
            over10 = (np.abs(p) > 10).mean() * 100
            print("  %-12s: mean=%7.4f std=%7.4f min=%8.2f max=%8.2f |>5s|=%.2f%% |>10s|=%.2f%%" % (
                pname, p.mean(), p.std(), p.min(), p.max(), over5, over10))

    return raw


# =============================================================================
if __name__ == '__main__':
    # mean/std 파일 로드
    print("=== Loading mean/std ===")
    mean_general = torch.load(os.path.join(SOKE_DIR, 'CSL-Daily/mean_133.pt')).numpy()
    std_general = torch.load(os.path.join(SOKE_DIR, 'CSL-Daily/std_133.pt')).numpy()
    csl_mean = torch.load(os.path.join(SOKE_DIR, 'CSL-Daily/csl_mean_133.pt')).numpy()
    csl_std = torch.load(os.path.join(SOKE_DIR, 'CSL-Daily/csl_std_133.pt')).numpy()

    print("\n[General mean/std]")
    for pname, (s, e) in parts.items():
        gs = std_general[s:e]
        print("  %-12s: mean_avg=%.6f  std_avg=%.6f  std_min=%.8f  std_max=%.6f" % (
            pname, mean_general[s:e].mean(), gs.mean(), gs.min(), gs.max()))

    print("\n[CSL mean/std]")
    for pname, (s, e) in parts.items():
        cs = csl_std[s:e]
        print("  %-12s: mean_avg=%.6f  std_avg=%.6f  std_min=%.8f  std_max=%.6f" % (
            pname, csl_mean[s:e].mean(), cs.mean(), cs.min(), cs.max()))

    # 각 데이터셋 분석
    all_datasets = {}
    for ds_name, ds_label, use_mean, use_std in [
        ('h2s', 'How2Sign', mean_general, std_general),
        ('csl', 'CSL-Daily', csl_mean, csl_std),
        ('phoenix', 'Phoenix', mean_general, std_general),
    ]:
        print("\n" + "=" * 80)
        print("  %s" % ds_label)
        print("=" * 80)
        all_raw, lengths, skip = load_dataset(ds_name)
        print("  Loaded: %d samples, skip: %d" % (len(all_raw), skip))
        if len(all_raw) > 0:
            raw = analyze_dataset(ds_name, all_raw, lengths, use_mean, use_std)
            if raw is not None:
                all_datasets[ds_name] = raw

    # Cross-dataset 비교
    if len(all_datasets) > 1:
        print("\n" + "=" * 80)
        print("  Cross-Dataset 비교 (Raw Std)")
        print("=" * 80)
        print("  %12s" % "", end="")
        for pname in parts:
            print("  %10s" % pname, end="")
        print()
        for ds_name, raw in all_datasets.items():
            print("  %12s" % ds_name, end="")
            for pname, (s, e) in parts.items():
                print("  %10.4f" % raw[:, s:e].std(), end="")
            print()

        print("\n  Cross-Dataset 비교 (Raw Mean)")
        print("  %12s" % "", end="")
        for pname in parts:
            print("  %10s" % pname, end="")
        print()
        for ds_name, raw in all_datasets.items():
            print("  %12s" % ds_name, end="")
            for pname, (s, e) in parts.items():
                print("  %10.4f" % raw[:, s:e].mean(), end="")
            print()

    print("\nDone.")

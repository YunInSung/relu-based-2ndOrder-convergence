import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ——————————————
# 공통 헬퍼: loss 문자열 → float
# ——————————————
def clean_loss_value(val):
    """
    - "tf.Tensor(1.0481111, shape=(), dtype=float32)" → 1.0481111
    - "0.9528497457504272"                → 0.9528497457504272
    """
    s = str(val)
    # 1) tf.Tensor 포맷이면 숫자 부분만 추출
    m = re.search(r"tf\.Tensor\(\s*([0-9eE+.-]+)", s)
    if m:
        return float(m.group(1))
    # 2) 아니면 그냥 float 캐스팅 시도
    try:
        return float(s)
    except:
        return np.nan

# ——————————————
# 개별 CSV 플롯
# ——————————————
def plot_csv(csv_path, loss_col, out_dir, save_interval=5):
    df = pd.read_csv(csv_path)
    df[loss_col] = df[loss_col].apply(clean_loss_value)  # 앞서 만든 helper
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df = df.dropna(subset=['epoch', loss_col])

    name = os.path.basename(csv_path).replace(".csv", "")
    plt.figure(figsize=(6,4))

    # 1) Custom만 먼저 플롯
    if 'custom' in df['optimizer'].values:
        sub_c = df[df['optimizer']=='custom']
        plt.plot(
            sub_c['epoch'],
            sub_c[loss_col],
            label='Custom',
            linestyle='-',
            marker='o',      # 포인트 찍어서 구분
            alpha=0.9
        )

    # 2) 나머지 optimizer들
    for opt, style in [
        ('adam',     {'linestyle':'--', 'alpha':0.7}),
        ('adamw',    {'linestyle':'-.', 'alpha':0.7}),
        ('adabelief',{'linestyle':':',  'alpha':0.7}),
    ]:
        if opt in df['optimizer'].values:
            sub = df[df['optimizer']==opt]
            plt.plot(
                sub['epoch'],
                sub[loss_col],
                label=opt.capitalize(),
                **style
            )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss" if loss_col=="loss" else "Validation Loss")
    plt.title(name)
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name + ".png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")

# ——————————————
# 평균 커브 플롯
# ——————————————
def plot_mean_curves(csv_paths, loss_col, out_dir):
    # 1) 파일명 그룹핑 (_runN 제거)
    pattern = re.compile(r"(.+)_run\d+\.csv$")
    groups = {}
    for p in csv_paths:
        m = pattern.search(os.path.basename(p))
        if not m: continue
        groups.setdefault(m.group(1), []).append(p)

    # 2) 그룹별로
    for name, paths in groups.items():
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            # clean both epoch·loss_col
            df['epoch']  = pd.to_numeric(df['epoch'], errors='coerce')
            df[loss_col] = df[loss_col].apply(clean_loss_value)
            dfs.append(df.dropna(subset=['epoch', loss_col]))
        df_all = pd.concat(dfs, ignore_index=True)

        # 3) 평균 계산
        df_mean = (
            df_all
            .groupby(['optimizer', 'epoch'], as_index=False)[loss_col]
            .mean()
            .rename(columns={loss_col: 'mean_'+loss_col})
        )

        # 4) **표준편차 계산** (추가)
        df_std = (
            df_all
            .groupby(['optimizer', 'epoch'], as_index=False)[loss_col]
            .std()
            .rename(columns={loss_col: 'std_'+loss_col})
        )

        # 5) 평균과 표준편차를 합치기
        df_stats = pd.merge(df_mean, df_std, on=['optimizer','epoch'], how='left')

        # 6) 플롯
        plt.figure(figsize=(6,4))
        styles = {
            'custom':    {'linestyle':'-','marker':'o','alpha':0.9},
            'adam':      {'linestyle':'--','alpha':0.7},
            'adamw':     {'linestyle':'-.','alpha':0.7},
            'adabelief': {'linestyle':':','alpha':0.7},
        }

        for opt, style in styles.items():
            sub = df_stats[df_stats['optimizer']==opt]
            if sub.empty: continue

            # (1) 평균 곡선
            plt.plot(
                sub['epoch'],
                sub['mean_'+loss_col],
                label=f"{opt.capitalize()} (mean)",
                **style
            )

            # (2) 음영 영역: mean ± std
            plt.fill_between(
                sub['epoch'],
                sub['mean_'+loss_col] - sub['std_'+loss_col],
                sub['mean_'+loss_col] + sub['std_'+loss_col],
                alpha=0.2
            )

        plt.xlabel("Epoch")
        plt.ylabel("Training Loss" if loss_col=="loss" else "Validation Loss")
        plt.title(f"{name} — Mean {'Train' if loss_col=='loss' else 'Val'} Loss")
        plt.legend(fontsize=8, loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved mean plot: {out_path}")

# ——————————————
# 실행
# ——————————————
train_csvs = glob.glob("data/train/*.csv")
val_csvs   = glob.glob("data/val/*.csv")

# 개별 플롯
# for csv in train_csvs:
#     plot_csv(csv, loss_col="loss",     out_dir="figures/train")
# for csv in val_csvs:
#     plot_csv(csv, loss_col="val_loss", out_dir="figures/val")

# 평균 플롯
plot_mean_curves(train_csvs, loss_col="loss",     out_dir="figures/train_mean")
plot_mean_curves(val_csvs,   loss_col="val_loss", out_dir="figures/val_mean")

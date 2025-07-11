import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) CSV 로드
df = pd.read_csv("logs/full_sensitivity_summary.csv")

# 2) WineQuality-Red 데이터만 필터 (dataset 컬럼이 "WineQuality")
df = df[df["dataset"].str.contains("WineQuality", case=False)]

# 3) 옵티마이저 리스트 및 라벨
optimizers = ["custom", "adam", "adamw", "adabelief"]
opt_labels = ["Custom", "Adam", "AdamW", "AdaBelief"]

# 4) 2×2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

for ax, opt, label in zip(axes.ravel(), optimizers, opt_labels):
    df_opt = df[df["optimizer"].str.lower() == opt.lower()]
    
    # run 별 val_loss 평균 → pivot
    pivot = (
        df_opt
        .groupby(["dropout_rate", "smooth_alpha"])["val_loss"]
        .mean()
        .unstack()
    )
    
    # 히트맵 그리기 (origin='upper'로 작은 드랍아웃이 위쪽에)
    im = ax.imshow(
        pivot.values,
        origin="upper",
        aspect="auto",
        interpolation="nearest"
    )
    
    # 제목 및 축 레이블
    ax.set_title(label, pad=10)
    ax.set_xlabel("Label Smoothing (α)")
    ax.set_ylabel("Dropout Rate")
    
    # 눈금 위치 및 레이블 (소수점 셋째 자리까지 표시)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])

# 5) 컬러바 (전체 서브플롯 공유)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label("Mean Validation Loss")

# 6) 전체 제목 및 저장
plt.suptitle("Optimizer Sensitivity Heatmaps (Validation Loss)", fontsize=16, y=1.02)
plt.savefig("figures/optimizer_sensitivity_heatmaps.png", dpi=300, bbox_inches="tight")

# plt.show()  # 파일 출력만 하면 된다면 주석 처리하세요.

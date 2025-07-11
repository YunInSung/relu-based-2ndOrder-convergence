import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) CSV 로드
df = pd.read_csv("logs/full_sensitivity_summary.csv")

# 2) 그릴 옵티마이저 목록 및 라벨
optimizers = ['custom', 'adam', 'adamw', 'adabelief']
opt_labels = ['Custom', 'Adam', 'AdamW', 'AdaBelief']

# 3) 2x2 subplot 준비
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

for ax, opt, label in zip(axes.ravel(), optimizers, opt_labels):
    # 해당 옵티마이저 데이터만 추출
    df_opt = df[df['optimizer'] == opt]
    
    # pivot: 행=dropout_rate, 열=label_smoothing, 값=val_loss
    pivot = df_opt.pivot(index='dropout_rate',
                         columns='label_smoothing',
                         values='val_loss')
    
    # 히트맵 그리기
    im = ax.imshow(pivot.values,
                   origin='lower',
                   aspect='auto',
                   interpolation='nearest')
    
    # 축 레이블 및 타이틀
    ax.set_title(label, pad=10)
    ax.set_xlabel('Label Smoothing (α)')
    ax.set_ylabel('Dropout Rate')
    
    # 눈금 설정
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])

# 4) 컬러바
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label('Validation Loss')

# 5) 저장 및 표시
plt.suptitle('Optimizer Sensitivity Heatmaps (Validation Loss)', fontsize=16, y=1.02)
plt.savefig("figures/optimizer_sensitivity_heatmaps.png", dpi=300, bbox_inches='tight')
plt.show()

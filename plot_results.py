import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (이미 로깅된) 결과 CSV 로드
df = pd.read_csv("logs/experiment_results.csv")

# Matplotlib 학술지 스타일
plt.style.use('seaborn-v0_8-paper')
plt.rc('font', size=12)
plt.rc('axes', titlesize=14, labelsize=12)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# 비교할 metric과 표시할 레이블 매핑 (컬럼 순서: Custom, Adam, AdamW, AdaBelief)
metrics = {
    'train': (
        ['custom_mean_train', 'adam_mean_train', 'adamw_mean_train', 'adabelief_mean_train'],
        'Training Loss'
    ),
    'val': (
        ['custom_mean_val', 'adam_mean_val', 'adamw_mean_val', 'adabelief_mean_val'],
        'Validation Loss'
    ),
    'time': (
        ['custom_mean_time', 'adam_mean_time', 'adamw_mean_time', 'adabelief_mean_time'],
        'Time (s)'
    ),
    'acc': (
        ['custom_mean_acc', 'adam_mean_acc', 'adamw_mean_acc', 'adabelief_mean_acc'],
        'Accuracy'
    ),
    'f1': (
        ['custom_mean_f1', 'adam_mean_f1', 'adamw_mean_f1', 'adabelief_mean_f1'],
        'F1 Score'
    ),
}

# 옵티마이저 순서에 대응하는 막대 레이블
opt_labels = ['Custom', 'Adam', 'AdamW', 'AdaBelief']
# 막대 색상 (원하시는 색상을 바꿔도 무방)
opt_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for metric_key, (cols_list, ylabel) in metrics.items():
    fig, ax = plt.subplots(figsize=(8, 5))

    # 데이터프레임에서 dataset과 해당 컬럼들만 추출
    # (dataset을 인덱스로 설정)
    data = df[['dataset'] + cols_list].set_index('dataset')

    # x축 위치 설정
    x = np.arange(len(data))
    total_opts = len(cols_list)
    # 막대 너비: 0.8을 4개로 나눠서 각 막대 간격을 둡니다.
    width = 0.8 / total_opts

    # 각 옵티마이저별 막대 그리기
    for i, col in enumerate(cols_list):
        # x 위치를 살짝씩 이동시켜서 겹치지 않게 함
        x_pos = x + (i - (total_opts - 1) / 2) * width
        ax.bar(
            x_pos,
            data[col],
            width,
            label=opt_labels[i],
            color=opt_colors[i],
            alpha=0.8
        )

    # 축 및 레이블 설정
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Dataset')
    ax.set_title(f'Comparison of {ylabel} across Datasets', pad=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Figure caption을 subplot 아래에 삽입
    fig.text(
        0.5, -0.1,
        f"Figure: {ylabel} comparison among Custom, Adam, AdamW, and AdaBelief on various datasets. "
        "Bars represent mean over repeats.",
        ha='center',
        fontsize=10
    )

    plt.tight_layout()
    fig.savefig(f"figures/comparison_{metric_key}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

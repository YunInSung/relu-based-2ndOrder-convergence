import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from pathlib import Path
import gc
import json
from DNN import DNN        # ← 사용자 정의 옵티마이저/모델
import matplotlib.pyplot as plt
from collections import defaultdict

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ───────────────────────────────────────────────────
# 0) 전역 설정: 재현성 확보
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU 메모리 growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ───────────────────────────────────────────────────
# 1) 하이퍼파라미터 그리드
param_grid = {
    "dropout_rate": [0.002, 0.004, 0.008],
    "smooth_alpha": [0.0125, 0.025, 0.05],
    "optimizer": ["Adam", "AdamW", "AdaBelief", "Custom"],
    "n_hidden_layers": [7]
}

optimizers = {
    "Adam":     lambda lr, wd: tf.keras.optimizers.Adam(lr),
    "AdamW":    lambda lr, wd: tf.keras.optimizers.AdamW(lr, weight_decay=wd),
    "AdaBelief":lambda lr, wd: tfa.optimizers.AdaBelief(lr, weight_decay=wd),
    "Custom":   None
}

# ───────────────────────────────────────────────────
# 2) 데이터 로더

def load_wine_quality(random_state=0):
    base = Path(__file__).parent if '__file__' in globals() else Path(os.getcwd())
    df = pd.read_csv(base/"winequality-red.csv", sep=';')
    X = StandardScaler().fit_transform(df.drop('quality', axis=1).values)
    bins = [2,4,5,6,7,10]
    y = np.digitize(df['quality'].values, bins) - 1
    return train_test_split(X, to_categorical(y, 6),
                            test_size=0.2, stratify=y,
                            random_state=random_state, shuffle=True)

datasets = {
    "WineQuality": load_wine_quality
}

# ───────────────────────────────────────────────────
# 3) 모델 빌더
def build_model(input_dim, output_dim, n_layers, dropout_rate):
    model = Sequential()
    for _ in range(n_layers):
        model.add(Dense(64, kernel_initializer='he_normal'))
        model.add(LeakyReLU(0.01))
        model.add(BatchNormalization())
        if dropout_rate > 0.0:
            model.add(Dropout(rate=dropout_rate))
    model.add(Dense(output_dim, activation='softmax'))
    return model

# ───────────────────────────────────────────────────
# 4) 실험 실행
REPEATS       = 3
EPOCHS        = 100
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

records = []
history_store = []  # epoch별 history 저장용

for dataset_name, loader in datasets.items():
    for params in ParameterGrid(param_grid):
        for run in range(REPEATS):
            # 데이터 준비
            X_train, X_val, y_train, y_val = loader(random_state=SEED + run)
            input_dim, output_dim = X_train.shape[1], y_train.shape[1]

            # 학습 시작
            start = time.time()
            if params['optimizer'] == "Custom":
                dnn = DNN(
                    layer_sizes=[input_dim] + [64]*params['n_hidden_layers'] + [output_dim],
                    batch_size=BATCH_SIZE,
                    dropout_rate=params['dropout_rate'],
                    label_smoothing=params['smooth_alpha']
                )
                dnn.training(
                    tf.convert_to_tensor(X_train, tf.float32),
                    tf.convert_to_tensor(y_train, tf.float32),
                    tf.convert_to_tensor(X_val, tf.float32),
                    tf.convert_to_tensor(y_val, tf.float32),
                    EPOCHS
                )
                elapsed = time.time() - start
                val_loss = dnn.compute_loss(
                    tf.convert_to_tensor(X_val, tf.float32),
                    tf.convert_to_tensor(y_val, tf.float32)
                ).numpy()
                _, _, proba = dnn._forward(tf.transpose(tf.convert_to_tensor(X_val, tf.float32)))
                y_pred = tf.argmax(tf.transpose(proba), axis=1).numpy()
                val_loss_history = dnn.ret_val_loss_list()  # → list of tf.Tensor or float

                # Tensor일 수 있으니 float 변환
                val_loss_history = [float(v) for v in val_loss_history]
                history = {"val_loss": val_loss_history}
                tf.keras.backend.clear_session()
                del dnn
                gc.collect()
            else:
                model = build_model(input_dim, output_dim,
                                    params['n_hidden_layers'],
                                    params['dropout_rate'])
                optimizer = optimizers[params['optimizer']](LEARNING_RATE, WEIGHT_DECAY)
                model.compile(
                    optimizer=optimizer,
                    loss=CategoricalCrossentropy(label_smoothing=params['smooth_alpha']),
                    jit_compile=False
                )
                hist = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )
                elapsed = time.time() - start
                val_loss = hist.history['val_loss'][-1]
                y_pred = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
                history = hist.history
                tf.keras.backend.clear_session()
                del model
                gc.collect()

            # 지표 계산
            y_true = y_val.argmax(axis=1)
            records.append({
                **params,
                "dataset": dataset_name,
                "run": run,
                "val_loss": val_loss,
                "val_acc": accuracy_score(y_true, y_pred),
                "val_f1":  f1_score(y_true, y_pred, average='macro'),
                "time": elapsed
            })
            if history is not None:
                history_store.append({
                    "meta": {**params, "dataset": dataset_name, "run": run},
                    "history": history
                })

# ───────────────────────────────────────────────────
# 5) 결과 저장 포맷 통일 (CSV + Parquet)
df = pd.DataFrame(records)
os.makedirs("logs", exist_ok=True)
df.to_csv("logs/full_sensitivity_summary.csv", index=False)
df.to_parquet("logs/full_sensitivity_summary.parquet", index=False)

with open("logs/full_sensitivity_histories.json", "w") as f:
    json.dump(history_store, f)

# 6) 요약 pivot 테이블 출력
summary = df.pivot_table(
    index=["dataset","dropout_rate","smooth_alpha","n_hidden_layers"],
    columns="optimizer",
    values=["val_loss","time"]
)
print(summary.round(4))


# 1. 로그 디렉터리 생성
os.makedirs("logs/val_loss_plots", exist_ok=True)

# 2. 메타 정보 기반으로 그룹핑
grouped = defaultdict(dict)
for entry in history_store:
    key = (
        entry["meta"]["dataset"],
        entry["meta"]["dropout_rate"],
        entry["meta"]["smooth_alpha"],
        entry["meta"]["n_hidden_layers"],
        entry["meta"]["run"]
    )
    optimizer = entry["meta"]["optimizer"]
    grouped[key][optimizer] = entry["history"]

# 3. 반복(run)별로 하나의 그래프에 4개 옵티마이저 val_loss를 그림
for meta_key, histories in grouped.items():
    dataset, d_rate, alpha, n_layers, run = meta_key
    plt.figure()

    for optimizer_name in ["Custom", "Adam", "AdamW", "AdaBelief"]:
        history = histories.get(optimizer_name)
        if history is not None and "val_loss" in history:
            val_losses = history["val_loss"]
            
            # 🎯 Custom일 경우, x축: [1, 5, 10, 15, ...]
            if optimizer_name == "Custom":
                epochs = [1] + [i for i in range(5, 5 * len(val_losses), 5)]
            else:
                epochs = list(range(1, len(val_losses) + 1))

            plt.plot(epochs, val_losses, label=optimizer_name, marker='o')

    plt.title(f"{dataset} | d={d_rate} α={alpha} L={n_layers} run={run}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)

    fname = (
        f"logs/val_loss_plots/"
        f"{dataset}_d{d_rate}_a{alpha}_l{n_layers}_r{run}.png"
    )
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
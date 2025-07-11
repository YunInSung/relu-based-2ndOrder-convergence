import os, time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import ttest_rel, t
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path          # ← NEW
import tensorflow_addons as tfa
import gc
import shutil

import wfdb
from wfdb import processing

from DNN import DNN        # ← 사용자 정의 옵티마이저/모델

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ────────────────────────────────────────────────
# 0. argparse로 하이퍼파라미터 & 옵션 받기
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Optimizer 비교 실험 스크립트")
parser.add_argument('--num_repeats',  type=int,   default=7,     help='실험 반복 횟수')
parser.add_argument('--epochs',       type=int,   default=250,   help='학습 에폭 수')
parser.add_argument('--batch_size',   type=int,   default=128,    help='배치 크기')
parser.add_argument('--lr',           type=float, default=1e-3,  help='학습률')
parser.add_argument('--weight_decay', type=float, default=1e-4,  help='Weight decay 값')
parser.add_argument('--seed',         type=int,   default=0,     help='랜덤 시드')
args = parser.parse_args()

NUM_REPEATS   = args.num_repeats
EPOCHS        = args.epochs
BATCH_SIZE    = args.batch_size
LR            = args.lr
WEIGHT_DECAY  = args.weight_decay
SEED          = args.seed
SAVE_INTERVAL = 5
DELTA         = 0.02
BATCH_NORM    = True
dropout_rate  = 0.004
label_smoothing = 0.025

# ────────────────────────────────────────────────
# 재현성을 위해 시드 고정
# ────────────────────────────────────────────────
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()

# ────────────────────────────────────────────────
# 폴더 생성
# ────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- 방어적 noninf / tost ---
def noninf(x, y, delta):
    d = x - y
    if len(d) < 2 or d.std(ddof=1) == 0:
        return np.nan
    tstat = (d.mean() - delta) / (d.std(ddof=1) / np.sqrt(len(d)))
    return t.cdf(tstat, df=len(d)-1)

def tost(x, y, delta):
    """Two One-Sided Test equivalence p-value."""
    d = x - y
    se = d.std(ddof=1) / np.sqrt(len(d))
    t_low = (d.mean() + delta) / se       # upper bound
    t_up  = (d.mean() - delta) / se       # lower bound
    p_low = 1 - t.cdf(t_low, df=len(d)-1)
    p_up  =     t.cdf(t_up,  df=len(d)-1)
    return max(p_low, p_up)

class TimeValHistory(tf.keras.callbacks.Callback):
    """Keras 콜백 – 에폭별 소요시간 & val-loss 기록"""
    def on_train_begin(self, logs=None):
        self.times, self.val_losses = [], []
        self.start = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.start)
        self.val_losses.append(logs["val_loss"])

def cohens_d(x, y):
    """
    Compute Cohen's d for two arrays x, y.
    d = (mean(x) - mean(y)) / pooled_sd
    """
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (mx - my) / pooled_sd

# ────────────────────────────────
# 1. 데이터 로더
# ────────────────────────────────
def load_mnist(random_state=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test  = x_test.reshape(-1, 28*28) / 255.0
    return x_train, x_test, to_categorical(y_train), to_categorical(y_test)

def load_cifar10(random_state=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.reshape(-1, 32*32*3) / 255.0
    x_test  = x_test.reshape(-1, 32*32*3) / 255.0
    return x_train, x_test, to_categorical(y_train), to_categorical(y_test)

def load_cifar100(random_state=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.reshape(-1, 32*32*3) / 255.0
    x_test  = x_test.reshape(-1, 32*32*3) / 255.0
    return x_train, x_test, to_categorical(y_train), to_categorical(y_test)

def load_20newsgroups(random_state=42):
    data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
    X = TfidfVectorizer(max_features=2000).fit_transform(data.data).toarray()
    y = to_categorical(data.target, len(data.target_names))
    return train_test_split(X, y, test_size=0.2, stratify=data.target, random_state=random_state)

def load_imbalance(random_state=42):
    X, y = make_classification(n_samples=30000, n_features=20, n_informative=15,
                               n_classes=4, weights=[0.7,0.15,0.1,0.05], random_state=random_state)
    return train_test_split(X, to_categorical(y), test_size=0.2, stratify=y, random_state=random_state)

def load_wine_quality(random_state=42):
    # ■ base 폴더:  (A) 스크립트를 .py 로 실행할 땐 __file__ 기준
    #              (B) Colab/Jupyter 노트북에선 현재 작업 디렉터리 기준
    base = Path(__file__).parent if '__file__' in globals() else Path(os.getcwd())
    
    csv_path = base / 'winequality-red.csv'
    df = pd.read_csv(csv_path, sep=';')

    X = StandardScaler().fit_transform(df.drop('quality', axis=1).values)
    bins = [2,4,5,6,7,10]
    y = np.digitize(df['quality'].values, bins) - 1
    return train_test_split(X, to_categorical(y, 6),
                            test_size=0.2, stratify=y, random_state=random_state)

def load_synthetic_gaussian(n_samples: int = 30000,
                            n_features: int = 12,
                            n_classes: int = 8,
                            class_sep: float = 1.0,
                            n_clusters_per_class: int = 3,
                            flip_y: float = 0.0,
                            random_state: int = 42):
    """
    합성 Gaussian 데이터 로더 (train/test split 포함).
    
    Parameters
    ----------
    n_samples : 전체 샘플 수
    n_features : 피처 수 (informative = n_features)
    n_classes : 클래스 수
    class_sep : 클래스 간 분리도 (0.5, 1.0, 2.0 등)
    n_clusters_per_class : 클래스당 클러스터 수 (1, 3, 5 등)
    flip_y : 레이블 노이즈 비율 (0.0, 0.05, 0.1 등)
    random_state : 시드
    
    Returns
    -------
    X_train, X_test, y_train_cat, y_test_cat
    """
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_features,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               flip_y=flip_y,
                               class_sep=class_sep,
                               random_state=random_state)
    y_cat = to_categorical(y, num_classes=n_classes)
    return train_test_split(X, y_cat,
                            test_size=0.2,
                            stratify=y,
                            random_state=random_state)

def load_fashion_mnist(random_state=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = x_train.reshape(-1, 28*28) / 255.0
    X_test  = x_test.reshape(-1, 28*28) / 255.0
    return X_train, X_test, to_categorical(y_train), to_categorical(y_test)

def load_credit_card_fraud(random_state=42):
    # 이 스크립트(experiment_runner.py) 파일이 있는 폴더
    base = Path(__file__).parent
    csv_path = base / 'creditcard.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")
    df = pd.read_csv(str(csv_path))

    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    y_cat = to_categorical(y, 2)

    return train_test_split(
        X, y_cat,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

def load_har_ucihar(random_state=42):
    # ① 이 스크립트 파일(experiment_runner.py) 위치를 구해서
    base = Path(__file__).parent
    # ② 그 폴더 안에 있는 har_X.npy, har_y.npy 파일 경로를 지정
    X_path = base / 'har_X.npy'
    y_path = base / 'har_y.npy'

    # ③ 파일이 실제로 있는지 체크 (디버깅용)
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected {X_path} and {y_path} in {base}")

    # ④ 로드
    X = np.load(str(X_path))
    y = np.load(str(y_path))

    num_classes = np.unique(y).size
    y_cat = to_categorical(y, num_classes)

    return train_test_split(
        X, y_cat,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )


def load_mitbih_arrhythmia(test_size=0.2, random_state=None):
    records = wfdb.get_record_list('mitdb')
    X, y = [], []

    for rec in records:
        record = wfdb.rdrecord(rec, pn_dir='mitdb')
        ann    = wfdb.rdann(  rec, 'atr',  pn_dir='mitdb')

        sig = record.p_signal[:, 0]
        fs  = record.fs

        # QRS 검출: WFDB 버전에 맞춰 하나 선택
        try:
            qrs_inds = processing.gqrs_detect(sig=sig, fs=fs)
        except AttributeError:
            # 구버전일 경우 xqrs_detect 사용
            qrs_inds = processing.xqrs_detect(sig=sig, fs=fs)

        before, after = int(0.2 * fs), int(0.3 * fs)
        for peak in qrs_inds:
            start, end = peak - before, peak + after
            if start < 0 or end > len(sig): continue

            segment = sig[start:end]
            X.append(segment)

            idx    = np.searchsorted(ann.sample, peak)
            symbol = ann.symbol[idx]
            if symbol == 'N':
                y.append(0)
            elif symbol in ('V','E'):
                y.append(1)
            else:
                y.append(2)

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=len(np.unique(y)))

    return train_test_split(
        X, y_cat,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

# 1) 데이터셋 설정 목록 정의
gauss_configs = [
    # {'name': 'Gauss_sep0.5_clust1', 'class_sep': 0.5, 'n_clusters_per_class': 1, 'flip_y': 0.0},
    # {'name': 'Gauss_sep1.0_clust3', 'class_sep': 1.0, 'n_clusters_per_class': 3, 'flip_y': 0.0},
    {'name': 'Gauss_sep2.0_clust5', 'class_sep': 2.0, 'n_clusters_per_class': 5, 'flip_y': 0.0},
    {'name': 'Gauss_sep1.0_clust3_flip0.05', 'class_sep': 1.0, 'n_clusters_per_class': 3, 'flip_y': 0.05},
]

datasets = {
    # "MNIST":      load_mnist,
    # "CIFAR10":    load_cifar10,
    # "CIFAR100":   load_cifar100,
    # "20NG":       load_20newsgroups,
    # "Imbalance":  load_imbalance,
    # "WineQuality":load_wine_quality,
    # "FashionMNIST":       load_fashion_mnist,
    # "HAR":                load_har_ucihar
}

for cfg in gauss_configs:
    datasets[cfg['name']] = lambda *, random_state=42, cfg=cfg: \
        load_synthetic_gaussian(
            n_samples=30000,
            n_features=10,
            n_classes=8,
            class_sep=cfg['class_sep'],
            n_clusters_per_class=cfg['n_clusters_per_class'],
            flip_y=cfg['flip_y'],
            random_state=random_state
        )

# ────────────────────────────────
# 2. 메인 실험 루프
# ────────────────────────────────
results_list = []
train_curve = {}
val_curve    = {}  # 데이터셋별 val_loss 리스트

for name, loader in datasets.items():
    print(f"\n=== Dataset: {name} ===")

    # ── train_curve, val_curve 초기화
    train_curve.setdefault(name, {
        'custom': [], 'adam': [], 'adamw': [], 'adabelief': []
    })
    val_curve.setdefault(name, {
        'custom': [], 'adam': [], 'adamw': [], 'adabelief': []
    })

    # metric 버퍼
    buf = {k: [] for k in
           ('train_c','val_c','time_c','acc_c','f1_c',
            'train_a','val_a','time_a','acc_a','f1_a',
            # 여기에 아래 8개 키를 추가
            'train_aw','val_aw','time_aw','acc_aw','f1_aw',
            'train_ab','val_ab','time_ab','acc_ab','f1_ab')}

    for r in range(NUM_REPEATS):
        tf.random.set_seed(SEED + r)
        np.random.seed(SEED + r)
        X_train, X_test, y_train, y_test = loader(random_state=r)

        input_dim    = X_train.shape[1]
        num_classes  = y_train.shape[1]
        layer_sizes  = [input_dim] + [64]*10 + [num_classes]

        # ── 2-1. Custom ────────────────────────────────────────
        dnn = DNN(layer_sizes=layer_sizes, batch_size=BATCH_SIZE)
        t0  = time.perf_counter()
        dnn.training(tf.convert_to_tensor(X_train, tf.float32),
                     tf.convert_to_tensor(y_train, tf.float32),
                     tf.convert_to_tensor(X_test, tf.float32),
                     tf.convert_to_tensor(y_test, tf.float32), EPOCHS)
        buf['time_c'].append(time.perf_counter() - t0)

        # loss
        lc = dnn.compute_loss(tf.convert_to_tensor(X_train, tf.float32),
                              tf.convert_to_tensor(y_train, tf.float32)).numpy()
        vc = dnn.compute_loss(tf.convert_to_tensor(X_test, tf.float32),
                              tf.convert_to_tensor(y_test, tf.float32)).numpy()
        buf['train_c'].append(lc); buf['val_c'].append(vc)

        # acc / f1
        X_test_tf = tf.transpose(tf.convert_to_tensor(X_test, tf.float32))
        _, _, proba_c_tf = dnn._forward(X_test_tf)
        proba_c = proba_c_tf.numpy().T      # ← transpose! shape → (batch_size, num_classes)
        y_pred_c = proba_c.argmax(axis=1)   # ← 이제 길이가 batch_size 로 맞음
        y_true = y_test.argmax(1)
        buf['acc_c'].append(accuracy_score(y_true, y_pred_c))
        buf['f1_c'].append(f1_score(y_true, y_pred_c, average='macro'))

        custom_loss_list = dnn.ret_loss_list()            # 5에폭마다 저장된 train loss
        custom_val_loss_list = dnn.ret_val_loss_list()    # 5에폭마다 저장된 val loss

        train_curve[name]['custom'].append(custom_loss_list)
        val_curve[name]['custom'].append(custom_val_loss_list)

        del dnn
        gc.collect()

        # ── 2-2. Adam (Keras) ──────────────────────────────────
        model = Sequential()
        for u in layer_sizes[1:-1]:
            model.add(Dense(u, kernel_initializer='he_normal')); model.add(LeakyReLU(0.01))
            if BATCH_NORM:
                model.add(tf.keras.layers.BatchNormalization())
            if dropout_rate > 0.0:
                model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss=CategoricalCrossentropy(
                        from_logits=False,       # softmax를 이미 거친 출력
                        label_smoothing=label_smoothing    # 스무스 라벨링 계수
                    ), jit_compile=True)

        cb = TimeValHistory()
        t0 = time.perf_counter()
        h  = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                       verbose=0, validation_data=(X_test,y_test), callbacks=[cb])
        buf['time_a'].append(time.perf_counter() - t0)
        buf['train_a'].append(h.history['loss'][-1])
        buf['val_a'].append(  h.history['val_loss'][-1])

        proba_a  = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred_a = proba_a.argmax(axis=1)
        buf['acc_a'].append(accuracy_score(y_true, y_pred_a))
        buf['f1_a'] .append(f1_score(y_true, y_pred_a, average='macro'))

        adam_train_loss_list = h.history['loss']          # length = EPOCHS
        adam_val_loss_list   = h.history['val_loss']      # length = EPOCHS

        train_curve[name]['adam'].append(adam_train_loss_list)
        val_curve[name]['adam'].append(adam_val_loss_list)
        
        tf.keras.backend.clear_session()
        del model
        gc.collect()  # Python GC 강제 호출

        # ── 2-3. AdamW (Keras) ──────────────────────────────────
        model_aw = Sequential()
        for u in layer_sizes[1:-1]:
            model_aw.add(Dense(u, kernel_initializer='he_normal')); model_aw.add(LeakyReLU(0.01))
            if BATCH_NORM:
                model_aw.add(tf.keras.layers.BatchNormalization())
            if dropout_rate > 0.0:
                model_aw.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model_aw.add(Dense(num_classes, activation='softmax'))

        model_aw.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
                         loss=CategoricalCrossentropy(
                                from_logits=False,       # softmax를 이미 거친 출력
                                label_smoothing=label_smoothing    # 스무스 라벨링 계수
                            ), jit_compile=True)

        cb_aw = TimeValHistory()
        t0_aw = time.perf_counter()
        h_aw  = model_aw.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             verbose=0, validation_data=(X_test,y_test), callbacks=[cb_aw])
        buf['time_aw'].append(time.perf_counter() - t0_aw)
        buf['train_aw'].append(h_aw.history['loss'][-1])
        buf['val_aw'].append(  h_aw.history['val_loss'][-1])

        proba_aw  = model_aw.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred_aw = proba_aw.argmax(axis=1)
        buf['acc_aw'].append(accuracy_score(y_true, y_pred_aw))
        buf['f1_aw'] .append(f1_score(y_true, y_pred_aw, average='macro'))

        # Time-vs-Convergence 로그(AdamW만) 저장 (필요하다면)
        aw_train_loss_list = h_aw.history['loss']
        aw_val_loss_list   = h_aw.history['val_loss']

        train_curve[name]['adamw'].append(aw_train_loss_list)
        val_curve[name]['adamw'].append(aw_val_loss_list)
        tf.keras.backend.clear_session()
        del model_aw
        gc.collect()  # Python GC 강제 호출

        # ── 2-4. AdaBelief (TensorFlow Addons) ───────────────────────────────
        # (tensorflow_addons를 미리 import: import tensorflow_addons as tfa)
        model_ab = Sequential()
        for u in layer_sizes[1:-1]:
            model_ab.add(Dense(u, kernel_initializer='he_normal'))
            model_ab.add(LeakyReLU(0.01))
            if BATCH_NORM:
                model_ab.add(tf.keras.layers.BatchNormalization())
            if dropout_rate > 0.0:
                model_ab.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model_ab.add(Dense(num_classes, activation='softmax'))

        # AdaBelief 옵티마이저 인스턴스 생성
        optimizer_ab = tfa.optimizers.AdaBelief(
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            # 사용할 수 있는 추가 하이퍼파라미터(예: epsilon, rectify, amsgrad 등)는
            # tfa.optimizers.AdaBelief 공식 문서 참조
        )

        model_ab.compile(
            optimizer=optimizer_ab,
            loss=CategoricalCrossentropy(
                                from_logits=False,       # softmax를 이미 거친 출력
                                label_smoothing=label_smoothing    # 스무스 라벨링 계수
                            ), jit_compile=True)

        cb_ab = TimeValHistory()
        t0_ab = time.perf_counter()
        h_ab  = model_ab.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            verbose=0, validation_data=(X_test, y_test),
            callbacks=[cb_ab]
        )
        buf['time_ab'].append(time.perf_counter() - t0_ab)
        buf['train_ab'].append(h_ab.history['loss'][-1])
        buf['val_ab'].append(  h_ab.history['val_loss'][-1])

        proba_ab  = model_ab.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred_ab = proba_ab.argmax(axis=1)
        buf['acc_ab'].append(accuracy_score(y_true, y_pred_ab))
        buf['f1_ab'] .append(f1_score(y_true, y_pred_ab, average='macro'))

        # Time vs Validation Loss 로그(원한다면)
        ab_train_loss_list = h_ab.history['loss']
        ab_val_loss_list   = h_ab.history['val_loss']

        train_curve[name]['adabelief'].append(ab_train_loss_list)
        val_curve[name]['adabelief'].append(ab_val_loss_list)
        tf.keras.backend.clear_session()
        del model_ab
        gc.collect()  # Python GC 강제 호출

        del X_train, y_train, X_test, y_test, y_true
        gc.collect()

    # ─────────────────────────────────────────────────────────
    # 3. 통계량 & p-value 계산
    # ─────────────────────────────────────────────────────────
    def arr(key): return np.array(buf[key])

    stats = {}
    for metric in (
        ('train','train_c','train_a','train_aw','train_ab'),
        ('val',  'val_c',  'val_a','val_aw','val_ab'),
        ('time', 'time_c', 'time_a','time_aw','time_ab'),
        ('acc',  'acc_c',  'acc_a','acc_aw','acc_ab'),
        ('f1',   'f1_c',   'f1_a','f1_aw','f1_ab')
    ):
        tag, c_key, a_key, aw_key, ab_key = metric
        c  = arr(c_key)
        a  = arr(a_key)
        aw = arr(aw_key)
        ab = arr(ab_key)

        # 평균
        stats[f'custom_mean_{tag}']   = c.mean()
        stats[f'adam_mean_{tag}']     = a.mean()
        stats[f'adamw_mean_{tag}']    = aw.mean()
        stats[f'adabelief_mean_{tag}'] = ab.mean()

        # 표준편차 (sample std, ddof=1)
        stats[f'custom_std_{tag}']      = c.std(ddof=1)
        stats[f'adam_std_{tag}']        = a.std(ddof=1)
        stats[f'adamw_std_{tag}']       = aw.std(ddof=1)
        stats[f'adabelief_std_{tag}']   = ab.std(ddof=1)

        # 2) 효과 크기 계산
        stats[f'd_c_vs_a_{tag}']  = cohens_d(c,  a)
        stats[f'd_c_vs_aw_{tag}'] = cohens_d(c,  aw)
        stats[f'd_c_vs_ab_{tag}'] = cohens_d(c,  ab)

        if len(c) > 1:
            # 커스텀 vs 각각 비교
            stats[f'p_ttest_c_vs_a_{tag}']   = ttest_rel(c, a).pvalue
            stats[f'p_ttest_c_vs_aw_{tag}']  = ttest_rel(c, aw).pvalue
            stats[f'p_ttest_c_vs_ab_{tag}']  = ttest_rel(c, ab).pvalue
            # Adam vs AdamW, Adam vs AdaBelief 등도 추가 가능
            stats[f'p_ttest_a_vs_aw_{tag}']  = ttest_rel(a, aw).pvalue
            stats[f'p_ttest_a_vs_ab_{tag}']  = ttest_rel(a, ab).pvalue
            stats[f'p_ttest_aw_vs_ab_{tag}'] = ttest_rel(aw, ab).pvalue
        else:
            stats[f'p_ttest_c_vs_a_{tag}']   = np.nan
            stats[f'p_ttest_c_vs_aw_{tag}']  = np.nan
            stats[f'p_ttest_c_vs_ab_{tag}']  = np.nan
            stats[f'p_ttest_a_vs_aw_{tag}']  = np.nan
            stats[f'p_ttest_a_vs_ab_{tag}']  = np.nan
            stats[f'p_ttest_aw_vs_ab_{tag}'] = np.nan

        if tag not in ('time',):
            stats[f'p_noninf_c_vs_a_{tag}']   = noninf(c,  a,  DELTA)
            stats[f'p_noninf_c_vs_aw_{tag}']  = noninf(c,  aw, DELTA)
            stats[f'p_noninf_c_vs_ab_{tag}']  = noninf(c,  ab, DELTA)
            stats[f'p_noninf_a_vs_aw_{tag}']  = noninf(a,  aw, DELTA)
            stats[f'p_noninf_a_vs_ab_{tag}']  = noninf(a,  ab, DELTA)
            stats[f'p_noninf_aw_vs_ab_{tag}'] = noninf(aw, ab, DELTA)

            stats[f'p_equiv_c_vs_a_{tag}']   = tost(c,  a,  DELTA)
            stats[f'p_equiv_c_vs_aw_{tag}']  = tost(c,  aw, DELTA)
            stats[f'p_equiv_c_vs_ab_{tag}']  = tost(c,  ab, DELTA)
            stats[f'p_equiv_a_vs_aw_{tag}']  = tost(a,  aw, DELTA)
            stats[f'p_equiv_a_vs_ab_{tag}']  = tost(a,  ab, DELTA)
            stats[f'p_equiv_aw_vs_ab_{tag}'] = tost(aw, ab, DELTA)

    results_list.append({'dataset': name, **stats})

# ────────────────────────────────
# 4. 결과 저장 & 요약
# ────────────────────────────────
df = pd.DataFrame(results_list)
df.to_csv("logs/experiment_results.csv", index=False)

# 메트릭별 요약 출력
for metric in ['train', 'val', 'acc', 'f1', 'time']:
    print(f"\n=== {metric.upper()} ===")

    if metric == 'time':
        # time은 평균만 표시
        display_cols = [
            'dataset',
            'custom_mean_time', 'custom_std_time',
            'adam_mean_time',   'adam_std_time',
            'adamw_mean_time',  'adamw_std_time',
            'adabelief_mean_time','adabelief_std_time',
        ]
    else:
        # train/val/acc/f1: 평균 + Custom 대비 p-value
        display_cols = [
            'dataset',
            f'custom_mean_{metric}',  f'custom_std_{metric}',
            f'adam_mean_{metric}',    f'adam_std_{metric}',
            f'adamw_mean_{metric}',   f'adamw_std_{metric}',
            f'adabelief_mean_{metric}',f'adabelief_std_{metric}',

            f'p_ttest_c_vs_a_{metric}',
            f'p_ttest_c_vs_aw_{metric}',
            f'p_ttest_c_vs_ab_{metric}',
            # 여기에 추가
            f'd_c_vs_a_{metric}',
            f'd_c_vs_aw_{metric}',
            f'd_c_vs_ab_{metric}',
        ]

    # 실제로 존재하는 컬럼만 선택
    display_cols = [c for c in display_cols if c in df.columns]

    # 원본은 소수점 여러 자리이므로, 보기 좋게 반올림
    sub = df[display_cols].copy()
    # 평균값 컬럼들은 소수점 둘째 자리까지, p-value는 셋째 자리까지 보도록
    for c in sub.columns:
        if c.startswith(('custom_mean_', 'adam_mean_', 'adamw_mean_', 'adabelief_mean_')):
            sub[c] = sub[c].round(4)
        elif c.startswith('p_ttest'):
            sub[c] = sub[c].round(4)
        elif c.startswith('d_'):
            sub[c] = sub[c].round(3)   # 효과 크기는 소수점 셋째 자리까지

    print(sub.to_string(index=False))

print("\n✅ Experiments complete. Results saved to logs/experiment_results.csv")

# 5. Epoch vs Training Loss → CSV 저장
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val",   exist_ok=True)
for name, curves in train_curve.items():
    num_runs = max(
        len(curves.get('custom', [])),
        len(curves.get('adam', [])),
        len(curves.get('adamw', [])),
        len(curves.get('adabelief', [])),
    )

    for run_idx in range(num_runs):
        # 각 optimizer별로 DataFrame 생성
        dfs = []
        save_interval = 5

        # Custom
        if run_idx < len(curves.get('custom', [])):
            loss_list = curves['custom'][run_idx]
            epochs = [1] + list(np.arange(save_interval, save_interval*(len(loss_list)-1)+1, save_interval)) if len(loss_list)>1 else [1]
            dfs.append(pd.DataFrame({
                'optimizer': 'custom',
                'epoch': epochs,
                'loss': loss_list
            }))

        # Adam
        if run_idx < len(curves.get('adam', [])):
            loss_list = curves['adam'][run_idx]
            dfs.append(pd.DataFrame({
                'optimizer': 'adam',
                'epoch': np.arange(1, len(loss_list)+1),
                'loss': loss_list
            }))

        # AdamW
        if run_idx < len(curves.get('adamw', [])):
            loss_list = curves['adamw'][run_idx]
            dfs.append(pd.DataFrame({
                'optimizer': 'adamw',
                'epoch': np.arange(1, len(loss_list)+1),
                'loss': loss_list
            }))

        # AdaBelief
        if run_idx < len(curves.get('adabelief', [])):
            loss_list = curves['adabelief'][run_idx]
            dfs.append(pd.DataFrame({
                'optimizer': 'adabelief',
                'epoch': np.arange(1, len(loss_list)+1),
                'loss': loss_list
            }))

        # 합쳐서 CSV로 저장
        df_all = pd.concat(dfs, ignore_index=True)
        out_path = f"data/train/{name}_train_run{run_idx}.csv"
        df_all.to_csv(out_path, index=False)
        print(f"Saved CSV: {out_path}")


# 6. Epoch vs Validation Loss → CSV 저장
for name, curves in val_curve.items():
    num_runs = max(
        len(curves.get('custom', [])),
        len(curves.get('adam', [])),
        len(curves.get('adamw', [])),
        len(curves.get('adabelief', [])),
    )

    for run_idx in range(num_runs):
        dfs = []
        save_interval = 5

        if run_idx < len(curves.get('custom', [])):
            loss_list = curves['custom'][run_idx]
            epochs = [1] + list(np.arange(save_interval, save_interval*(len(loss_list)-1)+1, save_interval)) if len(loss_list)>1 else [1]
            dfs.append(pd.DataFrame({
                'optimizer': 'custom',
                'epoch': epochs,
                'val_loss': loss_list
            }))

        if run_idx < len(curves.get('adam', [])):
            loss_list = curves['adam'][run_idx]
            dfs.append(pd.DataFrame({
                'optimizer': 'adam',
                'epoch': np.arange(1, len(loss_list)+1),
                'val_loss': loss_list
            }))

        if run_idx < len(curves.get('adamw', [])):
            loss_list = curves['adamw'][run_idx]
            dfs.append(pd.DataFrame({
                'optimizer': 'adamw',
                'epoch': np.arange(1, len(loss_list)+1),
                'val_loss': loss_list
            }))

        if run_idx < len(curves.get('adabelief', [])):
            loss_list = curves['adabelief'][run_idx]
            dfs.append(pd.DataFrame({
                'optimizer': 'adabelief',
                'epoch': np.arange(1, len(loss_list)+1),
                'val_loss': loss_list
            }))

        df_all = pd.concat(dfs, ignore_index=True)
        out_path = f"data/val/{name}_val_run{run_idx}.csv"
        df_all.to_csv(out_path, index=False)
        print(f"Saved CSV: {out_path}")

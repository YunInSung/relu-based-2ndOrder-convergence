# ⚙️ Diagonal Hessian Optimizer Experiments

*Custom Optimizer · Adam · AdamW · AdaBelief*

This repository includes the key scripts required to reproduce all experiments from "A Globally Convergent Second-Order Optimization Method Utilizing the Diagonal Hessian in ReLU-Based Models":

* `experiment_runner.py`: Runs the full suite of experiments
* `prepare_har.py`: Preprocesses the UCI HAR data
* `optimizer_sens_winequality.py`: Performs hyperparameter sensitivity analysis on WineQuality‑Red
* `plot_from_csv.py` / `plot_results.py`: Visualizes experimental results (CSV)
* `plot_optimizer_sensitivity.py`: Generates sensitivity heatmaps for the four optimizers

---

## 📋 System Requirements

* **OS**: Ubuntu 22.04 LTS
* **Python**: 3.9
* **CUDA**: 12.1 (nvcc V12.1.105)
* **cuDNN**: 9.9.0
* **TensorFlow**: 2.15.0 (XLA JIT enabled)
* **Main Libraries**

  ```text
  matplotlib==3.9.4
  numpy==1.26.4
  pandas==2.2.3
  scikit-learn==1.6.1
  tensorflow==2.15.0
  tensorflow-addons==0.22.0
  tensorflow-estimator==2.15.0
  tensorflow-io-gcs-filesystem==0.37.1
  tensorflow-probability==0.25.0
  ```

---

## 🛠 Installation

```bash
git clone https://github.com/YunInSung/relu-based-2ndOrder-convergence.git
cd relu-based-2ndOrder-convergence

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Data Download & Preprocessing

All scripts assume that data files are located at the project root.

| Dataset           | Filename                 | Download/Preparation Method                                                                              |
| ----------------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| WineQuality-Red   | `winequality-red.csv`    | Download manually from the UCI ML Repository and place in the root directory                             |
| Credit Card Fraud | `creditcard.csv`         | Download the ZIP from Kaggle → unzip → copy `creditcard.csv` into the root directory                     |
| UCI HAR           | `UCI_HAR_Dataset.zip`,   | Download the ZIP → unzip → run `prepare_har.py` → generates NumPy files `har_X.npy`, `har_y.npy` in root |
|                   | `har_X.npy`, `har_y.npy` |                                                                                                          |

### 1. WineQuality-Red

```bash
# From the project root
wget -O winequality-red.csv \
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

### 2. Credit Card Fraud

1. Visit the [Credit Card Fraud Kaggle page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** → download `creditcardfraud.zip`
3. Unzip and copy `creditcard.csv` to the project root

### 3. UCI HAR

```bash
# ➊ Download and unzip the original dataset
wget -O UCI_HAR_Dataset.zip \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI_HAR_Dataset.zip -d ./UCI_HAR_Dataset

# ➋ Convert to NumPy
python prepare_har.py  # produces har_X.npy, har_y.npy in the root directory
```

---

## 🚀 Running Experiments

### A. Basic MLP Optimization Experiments

```bash
python experiment_runner.py \
  --epochs 250 \
  --batch_size 128 \
  --num_repeats 7 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --seed 42
```

* **--epochs**: Number of epochs (default: 250)
* **--batch\_size**: Batch size (default: 128)
* **--num\_repeats**: Number of repeats (default: 7)
* **--lr**: Learning rate for baseline optimizers
* **--weight\_decay**: For AdamW/AdaBelief only
* **--seed**: Random seed

### B. WineQuality-Red Sensitivity Analysis

The `optimizer_sens_winequality.py` script automatically evaluates performance across combinations of **dropout rate** and **label smoothing coefficient (α)** on the WineQuality-Red dataset:

```bash
python optimizer_sens_winequality.py
```

All parameter combinations defined in the script’s `ParameterGrid` are executed.

---

## 📊 Result Visualization

| Script                 | Input CSV Path                          | Output PNG Path                                                                                                                                   |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **plot\_from\_csv.py** | `data/train/*.csv` <br>`data/val/*.csv` | `figures/train_mean/*.png` <br>`figures/val_mean/*.png` <br>*Individual curves: `figures/train/`, `figures/val/` — uncomment in script to enable* |
| **plot\_results.py**   | `logs/experiment_results.csv`           | `figures/comparison_*.png` *(per metric)*                                                                                                         |

> Both scripts require no command-line arguments.<br>Simply run `python plot_from_csv.py` or `python plot_results.py`.

| Logs Path                               | Description                                                                                                                                           |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `logs/full_sensitivity_summary.csv`     | Summary of *val\_loss, time, acc, f1* for all parameter combinations × repeats                                                 fileciteturn9file12 |
| `logs/full_sensitivity_summary.parquet` | Same content saved in Parquet format                                                                                                                  |
| `logs/full_sensitivity_histories.json`  | Full Keras history for each experiment run                                                                                                            |
| `logs/val_loss_plots/*.png`             | Validation loss curves for each parameter combination × run                                                                                           |

---

### E. Optimizer Sensitivity Heatmap

Use the `plot_optimizer_sensitivity.py` script to create validation loss heatmaps for the four optimizers (Custom, Adam, AdamW, AdaBelief):

```bash
python plot_optimizer_sensitivity.py
```

* **Required file**: `full_sensitivity_summary.csv` in the `logs/` folder
* **Output**: `figures/optimizer_sensitivity_heatmaps.png`

---

## 📂 Project Structure (Summary)

```
.
├── DNN.py
├── experiment_runner.py
├── plot_from_csv.py
├── plot_results.py
├── plot_optimizer_sensitivity.py
├── optimizer_sens_winequality.py
├── prepare_har.py
├── data/ …
├── figures/ …
├── logs/ …
├── requiredments.txt
└── README.md
```

---

## ⚖️ License

This project is distributed under the **MIT License**.

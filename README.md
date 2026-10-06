
# 🛍 E-Commerce Clothing Reviews Analysis & Sentiment Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://e-commerce-clothing-reviews-4hs7nwvmqcapvrvr5ctfyn.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![uv](https://img.shields.io/badge/uv-Package_Manager-DE5B8D?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

An end-to-end Machine Learning and Natural Language Processing (NLP) system built on the **Women's E-Commerce Clothing Reviews** dataset. This project processes customer text reviews along with structured ratings and categorical metadata to accurately predict customer recommendations and sentiment.

The project features a modular architecture, hyperparameter-tuned Scikit-Learn/Transformer pipelines, an automated retraining script for CI/CD runners, and an interactive **Streamlit** dashboard.

---

## 🌐 Live Demo

🚀 **Try the app live here:** [E-Commerce Clothing Reviews Streamlit App](https://e-commerce-clothing-reviews-4hs7nwvmqcapvrvr5ctfyn.streamlit.app/)

---

## 📌 Features

- **Automated Preprocessing & Pipeline:** Modular feature engineering handling text missing values, categorical encoding (`ONEHOT_COLS`), and numerical scaling.
- **Robust Model Architecture:** Scikit-Learn pipeline paired with hyperparameter tuning (`best_params.joblib`) for optimal ROC-AUC and F1 performance.
- **Interactive Web App:** Real-time prediction dashboard powered by Streamlit for instant text review analysis.
- **Fast Package Management:** Managed via `uv` using locked dependencies (`uv.lock`) for lightning-fast reproducible builds.
- **Production CI/CD Ready:** Automated training script (`train.py`) with quality thresholds suitable for GitHub Actions or cloud runner integration.

---

## 📁 Repository Structure

```text
e-commerce-clothing-reviews/
├── data/                         # Datasets directory (raw & processed)
│   └── Womens Clothing E-Commerce Reviews.csv
├── models/                       # Model artifacts and tuned hyperparameter configs
│   ├── best_params.joblib
│   └── model.joblib
├── src/                          # Source code package
│   ├── __init__.py
│   ├── app.py                    # Streamlit web application
│   ├── config.py                 # Paths, target variables, column mappings
│   └── pipeline.py               # Preprocessing & build_pipeline() definition
├── .gitignore
├── pyproject.toml                # Project configuration & package metadata
├── uv.lock                       # Lockfile for deterministic environments
├── train.py                      # Automated model re-training and validation script
└── README.md

```

---

## 🛠️️ Tech Stack & Dependencies

* **Language:** Python 3.12+
* **Environment & Package Manager:** [`uv`](https://github.com/astral-sh/uv)
* **Data & Modeling:** Pandas, NumPy, Scikit-Learn, LightGBM, PyTorch / Hugging Face Transformers
* **Visualization & Serving:** Streamlit, Matplotlib, Seaborn
* **Serialization:** Joblib

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone [https://github.com/yousssef96/E-Commerce-Clothing-Reviews.git](https://github.com/yousssef96/E-Commerce-Clothing-Reviews.git)
cd E-Commerce-Clothing-Reviews

```

### 2. Set Up the Environment

#### Option A: Using `uv` (Recommended)

```bash
# Create and synchronize environment from uv.lock
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```

#### Option B: Using Standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

```

---

## 🏋️‍♂️ Model Training & Pipeline

To run end-to-end model retraining and validation locally:

```bash
python train.py

```

### What `train.py` does:

1. Loads data specified in `src/config.py`.
2. Splits data using stratified sampling to preserve label balances.
3. Builds an uninitialized pipeline from `src.pipeline.build_pipeline()`.
4. Injects optimal parameters from `models/best_params.joblib`.
5. Fits the model, evaluates test set **ROC-AUC** and **F1-Score**, and saves the fitted model to `models/model.joblib`.

---

## 🖥️ Running the Streamlit Web App Locally

To launch the interactive prediction dashboard locally:

```bash
streamlit run src/app.py

```

Open `http://localhost:8501` in your browser to input review text, ratings, and division parameters to get real-time recommendation predictions.

---

## ☁️ Deployment (Streamlit Community Cloud)

This app is hosted live at: [https://e-commerce-clothing-reviews-4hs7nwvmqcapvrvr5ctfyn.streamlit.app/](https://e-commerce-clothing-reviews-4hs7nwvmqcapvrvr5ctfyn.streamlit.app/)

When deploying to Streamlit Community Cloud:

* Set **Main file path** to `src/app.py`.
* Ensure `sys.path` resolution is handled at the top of `src/app.py`:
```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

```



---

## 📊 Model Evaluation

| Metric | Target Score |
| --- | --- |
| **ROC-AUC** | $\ge 0.92$ |
| **F1-Score** | $\ge 0.93$ |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## 👤 Author

**Youssef Abdelmoneim**

* GitHub: [@yousssef96](https://www.google.com/search?q=https://github.com/yousssef96)

```

```

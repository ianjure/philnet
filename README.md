# **PhiLNet — Phishing Lightweight Network**

**PhiLNet (Phishing Lightweight Network)** is a fusion-based machine learning framework for **real-time phishing detection**, combining **textual analysis** and **heuristic features** to accurately identify phishing websites and messages with low computational overhead.

This repository contains the **experimental, training, and evaluation pipeline** used to develop PhiLNet.
The **production-ready model** is already deployed as a Chrome extension.

---

## 📌 Project Overview

PhiLNet is designed to:

-   Detect phishing attempts in real time
-   Combine **text-based features** and **rule-based heuristics**
-   Remain lightweight for browser-level deployment
-   Support reproducible research and experimentation

⚠️ **Note:**
This repository is **not required** to use PhiLNet in production.
End users can install **PhiLNet Vanguard** directly from the Chrome Web Store.

---

## 📂 Repository Structure

After setup, your project directory should look like this:

```
PhiLNet/
├── data/
│   ├── raw/
│   │   └── dataset/          # Downloaded phishing dataset
│   └── processed/            # Cleaned and feature-engineered data
│
├── models/                   # Trained model artifacts
│
├── notebooks/                # Jupyter notebooks for the pipeline
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   └── 04_model_training_and_evaluation.ipynb
│
├── results/                  # Figures, metrics, and evaluation outputs
│
├── requirements.txt          # Python dependencies
└── setup.txt                 # Environment setup instructions
```

---

## 📊 Dataset

PhiLNet uses a publicly available phishing dataset hosted on Mendeley Data.

### **Download Instructions**

1. Download the dataset from:
   👉 [Phishing Websites Dataset](https://data.mendeley.com/datasets/n96ncsr5g4/1)

2. Extract the downloaded archive.

3. Place the **entire `dataset` folder** inside:

```
data/raw/dataset/
```

The notebooks assume this exact directory structure.

---

## 🛠️ Requirements

Before running the project, make sure you have:

-   **Conda** (Anaconda or Miniconda)
-   **Jupyter Notebook**
-   **Python 3.8** (recommended)

Ensure that:

-   `requirements.txt` is located in the **base project directory**
-   You follow the environment instructions exactly as specified in `setup.txt`

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/PhiLNet.git
cd PhiLNet
```

2. **Follow the environment setup**

```bash
# Read and follow all steps inside:
setup.txt
```

This typically includes:

-   Creating a conda environment
-   Installing dependencies from `requirements.txt`
-   Registering the environment as a Jupyter kernel

---

## ▶️ Running the Project

Once setup is complete, launch Jupyter Notebook:

```bash
jupyter notebook
```

Then run the notebooks **in order**:

1. `01_data_collection.ipynb`

    - Loads and validates the raw phishing dataset

2. `02_data_preprocessing.ipynb`

    - Cleans data and extracts textual and heuristic features

3. `03_exploratory_data_analysis.ipynb`

    - Analyzes phishing patterns, feature distributions, and correlations

4. `04_model_training_and_evaluation.ipynb`

    - Trains the PhiLNet fusion model
    - Evaluates performance and saves results

⚠️ **Important:**
Do not skip notebooks — each step depends on outputs from the previous one.

---

## 🚀 Deployed Version

PhiLNet is already deployed as a Chrome extension:

### **🔐 PhiLNet Vanguard**

-   Real-time phishing detection
-   Runs directly in the browser
-   Optimized for low latency and minimal resource usage

➡️ Available on the **Chrome Web Store**: [PhiLNet Vanguard](https://chromewebstore.google.com/detail/nncjacjbfbidjahpfbcbdngpmflmmpee?utm_source=item-share-cb)

This repository is intended **only for research, experimentation, and model development**.

---

## 📄 License & Usage

-   This project is intended for **academic, research, and educational purposes**
-   Dataset usage follows the original dataset’s license
-   The deployed extension may follow a separate license

---

## 📬 Contact & Contributions

If you would like to:

-   Extend PhiLNet
-   Propose improvements
-   Reproduce or benchmark results

Feel free to open an issue or submit a pull request.

---

**PhiLNet — Built to detect, designed to protect.** 🛡️

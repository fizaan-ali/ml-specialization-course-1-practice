# Machine Learning Specialization — Course 1 Practice Archive

> **Personal study archive.** This is an independent learning archive, not an official DeepLearning.AI or Stanford repository.

This repository preserves my practice notebooks and helper code while studying **Supervised Machine Learning: Regression and Classification**. The notebooks explore concepts from scratch and with NumPy, Matplotlib, and scikit-learn.

## Topics covered

- Linear regression model representation and predictions
- Squared-error cost function and gradient descent
- Univariate and multiple-variable linear regression
- Feature scaling and house-price prediction
- Polynomial features and feature engineering
- Logistic regression, sigmoid, logistic loss, and decision boundaries
- Regularization for linear and logistic regression
- Using scikit-learn estimators
- NumPy vectorization, Boolean masks, and basic Matplotlib plotting

## Repository layout

```text
.
├── data/                         # Small CSV-style practice datasets (no headers)
├── notebooks/
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── regularization/
│   └── numpy_and_plotting/
├── src/                          # Reusable helper modules
├── docs/
│   ├── github-setup.md
│   └── rename-map.md
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer, then from the repository root run:

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
jupyter lab
```

Open a notebook from the `notebooks/` directory. Notebooks that use shared helper modules include a small setup cell to make the `src/` folder importable. Dataset loaders resolve files from `data/`.

## Data notes

The files in `data/` are comma-separated numeric values without header rows. See [`data/README.md`](data/README.md) for the column meanings recorded from the notebooks.

## Notes about the work

These notebooks document learning and experimentation; they are not a standalone, independently evaluated machine-learning project. Some notebooks repeat concepts for practice or compare a from-scratch implementation with scikit-learn. Repetition is intentional in a learning archive.

# Linear Regression Lecture - Environment Setup (Isolated)

This guide sets up a dedicated virtual environment for the Linear Regression lecture only. It won’t affect your shared `ml_lectures_env`.

## Location
- Lecture folder: `Lecture 3 Linear Regression/`
- Env path: `Lecture 3 Linear Regression/linear_regression_env/`
- Jupyter kernel name: `Python (Linear Regression Py39)`

## Prerequisites
- macOS
- Python 3.9 recommended (the venv uses a Python 3.9 interpreter)

## Create and Activate the Environment
```bash
# From the repo root (or cd into the lecture folder)
# Recommended: create the venv using your Python 3.9 interpreter
"/Users/ming/Dropbox/learn-ml-by-building/ml_lectures_env/bin/python" -m venv "Lecture 3 Linear Regression/linear_regression_env"

# Activate (macOS)
source "Lecture 3 Linear Regression/linear_regression_env/bin/activate"

# Upgrade build tools
python -m pip install --upgrade pip setuptools wheel
```

## Install Core Packages
```bash
python -m pip install "numpy<2" "pandas>=1.5" "xarray>=2023.10"
```

## Install ECMWF Packages (required for demo)
```bash
python -m pip install "climetlab>=0.20,<0.21" "climetlab-mltc-surface-observation-postprocessing>=0.3,<0.4"
```

## Jupyter and Kernel Registration
```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name linear-regression-py39 --display-name "Python (Linear Regression Py39)"
```

## Start Jupyter in the Lecture Folder
```bash
# From repo root
"Lecture 3 Linear Regression/linear_regression_env/bin/jupyter" notebook --no-browser --ip=127.0.0.1
```
Open the printed URL (e.g., http://127.0.0.1:8890/tree) and select the kernel: `Python (Linear Regression Py39)`.

## RISE (Slide Show) in this Environment
RISE works with the classic Notebook interface (Notebook 6).
```bash
python -m pip install "notebook==6.5.7" "rise>=5.7"
python -m jupyter nbextension install rise --py --sys-prefix
python -m jupyter nbextension enable rise --py --sys-prefix
```

## Notes
- Your shared env `ml_lectures_env` remains untouched.

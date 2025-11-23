# ml-electrical-fault-detection

Machine learning project for detecting and classifying electrical faults using voltage and current signal data. Includes preprocessing, analysis, model training, and evaluation for power system fault diagnosis.

## Folder structure

```bash
project/
├── data/
│   ├── raw/              # Original classData.csv
│   └── processed/        # Cleaned/processed data
├── notebooks/
│   ├── 01_notebook.ipynb
│   ├── 02_notebook.ipynb
│   ├── 03_notebook.ipynb
│   ├── 04_notebook.ipynb
│   └── 05_notebook.ipynb
├── src/                  # Reusable Python modules
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
├── models/               # Saved model files
├── results/              # Plots, metrics, reports
├── docs/                 # Your current docs
├── requirements.txt      # Dependencies
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

From the project root:

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat

```

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1

```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

---

### 3. Using the virtual environment in Jupyter notebooks inside VS Code

1.  Make sure the virtual environment is activated.
2.  Install `ipykernel` in your venv if not already installed:

```bash
pip install ipykernel

```

3.  Add your venv as a kernel:

```bash
python -m ipykernel install --user --name=venv

```

4.  Open VS Code and your notebook, then select the kernel named `venv` from the top-right kernel selector. This ensures the notebook uses the packages from your virtual environment.

---

This ensures a clean environment and reproducible results for anyone using your project.

# ML Course - Terminal Setup Instructions
**Run these commands BEFORE opening the Jupyter notebook**

## 1. Environment Setup

### Create and activate the virtual environment (if not already created):
```bash
# Create virtual environment at repo root
python3 -m venv ml_lectures_env

# Activate it:
# macOS/Linux:
source ml_lectures_env/bin/activate

# Windows PowerShell:
ml_lectures_env\Scripts\Activate.ps1
```

## 2. Install Required Packages

### Install base requirements:
```bash
# If you have a requirements.txt at repo root:
pip install -r requirements.txt

# Or install directly:
pip install torch>=2.0.0 transformers>=4.36.0 peft>=0.7.0 accelerate>=0.24.0 \
  datasets>=2.14.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 \
  matplotlib>=3.7.0 tqdm>=4.65.0 spacy>=3.0.0 thefuzz>=0.19.0 \
  python-Levenshtein>=0.20.0 rank-bm25>=0.2.2 pyserini>=0.20.0 \
  cleantext>=1.1.4 flask>=3.0.0 flask-cors>=4.0.0 beautifulsoup4>=4.11.0 \
  selenium>=4.0.0 rich>=13.0.0 jupyter>=1.0.0 notebook>=7.0.0 RISE>=5.7.1 \
  sentence-transformers>=2.2.0 seaborn>=0.12.0 psutil>=5.9.0
```

### Quick install (Gemma fine-tuning deps only)
Use this minimal set if you only need to run the Gemma-3 classifier and LoRA fine-tuning steps.
```bash
python3 -m pip install --upgrade pip
pip install torch transformers peft accelerate safetensors numpy
```

### Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

### For CPU-only machines (optional):
```bash
# If you're on a CPU-only machine, install PyTorch CPU version:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

## 3. Create Directory Structure

```bash
# Create necessary directories
mkdir -p "Lecture 1 Overview/data"
mkdir -p "Lecture 1 Overview/models/cat_classifier"
mkdir -p "Lecture 1 Overview/logs"
mkdir -p "Lecture 1 Overview/outputs"
```

## 4. Download WebShop Datasets

```bash
cd "Lecture 1 Overview/data"

# Download small datasets (recommended for course)
# items_shuffle_1000.json — product scraped info
gdown "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib"

# items_ins_v2_1000.json — product attributes
gdown "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu"

# Return to main directory
cd ../..
```

### If gdown is not installed:
```bash
pip install gdown
```

### Alternative: Download manually from browser:
- `items_shuffle_1000.json`: https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
- `items_ins_v2_1000.json`: https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
- Save both files to `Lecture 1 Overview/data/`

## 5. HuggingFace Setup (for Gemma model)

### Accept Gemma-3-270m license:
1. Visit: https://huggingface.co/google/gemma-3-270m
2. Click "Accept License" (requires free HuggingFace account)

### Login to HuggingFace CLI:
```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```

### (Recommended) Store models inside this project
To keep downloads self-contained for the course, set the Hugging Face cache to the project folder (no user-specific paths).

```bash
# macOS/Linux (run at the repo root)
export TRANSFORMERS_CACHE="$(pwd)/Lecture 1 Overview/models"

# Windows PowerShell (run at the repo root)
$env:TRANSFORMERS_CACHE = "$PWD/Lecture 1 Overview/models"
```

Notes:
- You can alternatively set `HF_HOME` to the same path; `TRANSFORMERS_CACHE` is sufficient for Transformers models.
- In Python, you can also pass `cache_dir` to `from_pretrained` if you prefer a per-script setting.

## 6. Java Installation (for PySerini - optional)

### macOS:
```bash
brew install openjdk@11
```

### Ubuntu/Debian:
```bash
sudo apt-get install openjdk-11-jdk
```

---

## Next Steps
1. Ensure your virtual environment is activated
2. Open Jupyter Notebook: `jupyter notebook`
3. Run the verification notebook to test everything works

**Note:** Keep the virtual environment activated when running the verification notebook!
# Quick Start Guide - Brain Tumor Detection

Get your project running in 5 minutes!

---

## Prerequisites

- Python 3.8 or higher installed
- Git installed (for cloning from GitHub)
- 8GB+ RAM recommended

---

## Step 1: Get the Code

### Option A: Clone from GitHub

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

### Option B: Download ZIP

1. Go to your GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal in that folder

---

## Step 2: Set Up Python Environment

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes as it downloads TensorFlow and other libraries.

---

## Step 3: Get the Dataset

### Download from Kaggle

1. Go to: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click "Download" (create free account if needed)
3. Extract the ZIP file
4. Place the extracted folders in the `data/` directory

Your folder structure should look like:

```
brain-tumor-detection/
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
```

---

## Step 4: Run the Code

### Option A: Using Jupyter Notebook (Recommended for Beginners)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/brain_tumor_classification.ipynb
# Run all cells (Cell → Run All)
```

### Option B: Using Python Script

```bash
# Train the model
python src/train.py --epochs 10 --max_trials 5
** This depends on the capabilities of your device**
```

---

## Step 5: View Results

After training completes, check:

- **Model**: Saved in `models/brain_tumor_model/`
- **Confusion Matrix**: `results/confusion_matrix.png`
- **Console Output**: Shows accuracy and performance metrics

---

## Common Issues & Fixes

### Issue 1: "Module not found"

**Fix**: Make sure you installed all requirements and activated your virtual environment.

```bash
pip install -r requirements.txt
```

### Issue 2: "Data path not found"

**Fix**: Check that your data folder structure is correct. See Step 3.

### Issue 3: "CUDA/GPU errors"

**Fix**: TensorFlow will automatically use CPU if GPU is not available. Training will just be slower.

### Issue 4: "Out of memory"

**Fix**: Close other applications. If problem persists, reduce batch size in the code.

---

## What's Next?

### Explore the Code

- **Data Preprocessing**: `src/data_preprocessing.py`
- **Training Script**: `src/train.py`
- **Jupyter Notebook**: `notebooks/brain_tumor_classification.ipynb`

### Improve the Model

Try modifying:
- Number of epochs (increase for better accuracy)
- Image size (larger images may improve accuracy)
- Data augmentation parameters
- Different architectures

### Deploy the Model

- Create a web app with Flask/FastAPI
- Build a mobile app
- Deploy to cloud (AWS, Google Cloud, Azure)

---

## Need Help?

1. **Check the full README**: [README.md](README.md)
2. **Dataset instructions**: [data/README.md](data/README.md)
3. **Open an issue**: On GitHub
4. **Contact me**: Aya.abdelmoniem.afattah@gmail.com

---

## Useful Commands

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Deactivate virtual environment
deactivate

# Update dependencies
pip install -r requirements.txt --upgrade

# Run training with custom settings
python src/train.py --epochs 20 --max_trials 10

# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# List all installed packages
pip list
```

---

## Expected Training Time

On a typical laptop:
- **With GPU**: 30-60 minutes
- **Without GPU**: 2-4 hours

**First run will be slower** as AutoKeras searches for the best architecture.

---

## Minimal Working Example

If you just want to test quickly with a small subset:

```python
# In Python or Jupyter
from src.data_preprocessing import DataPreprocessor

# Initialize
preprocessor = DataPreprocessor('data')

# Load just a few images to test
# (Modify the code to limit number of images loaded)

print("Setup successful!")
```

---

## Congratulations!

You're now ready to work on brain tumor classification with deep learning!

**Remember**:
- ⭐ Star the GitHub repository if you found it useful
- 📝 Read the full documentation for detailed explanations
- 🤝 Contribute improvements via Pull Requests
- 📧 Share your results with the community

---

**Happy Coding!**

*For detailed documentation, see [README.md](README.md)*

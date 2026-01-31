# Brain Tumor Classification Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE](https://img.shields.io/badge/Published-IEEE%20ICCA%202024-brightgreen.svg)](https://doi.org/10.1109/ICCA62237.2024.10927964)

> A deep learning-based framework for automated brain tumor classification using MRI images, achieving **96.5% accuracy** in multi-class tumor detection.

![Brain Tumor Detection Banner](images/banner.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Research Publication](#research-publication)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Brain tumors are among the most serious and life-threatening diseases. Early and accurate detection is crucial for effective treatment planning. This project implements a deep learning solution using **AutoKeras** and **ResNet50** to classify brain MRI images into four categories:

1. **No Tumor** - Healthy brain tissue
2. **Glioma** - Most common malignant brain tumor
3. **Meningioma** - Usually benign tumor
4. **Pituitary Tumor** - Tumor affecting the pituitary gland

The model leverages transfer learning and automated machine learning (AutoML) to achieve high classification accuracy, making it a valuable tool for medical diagnosis support.

---

## ✨ Features

- ✅ **Multi-class Classification**: Accurately classifies 4 types of brain conditions
- ✅ **Deep Learning**: Uses ResNet50 architecture with transfer learning
- ✅ **Automated ML**: Implements AutoKeras for hyperparameter optimization
- ✅ **Image Preprocessing**: Advanced preprocessing pipeline including:
  - Bilateral filtering for noise reduction
  - Color mapping for enhanced visualization
  - Image augmentation for improved generalization
- ✅ **High Accuracy**: Achieves 96.5% classification accuracy
- ✅ **Published Research**: Presented at IEEE ICCA 2024 conference
- ✅ **Reproducible**: Complete code with detailed documentation

---

## 📊 Dataset

### Source
This project uses the **Brain Tumor MRI Dataset** (Masoud Collection) containing MRI scans of brain tumors.

### Dataset Statistics
- **Total Images**: 7,023 MRI scans
- **Training Set**: 5,712 images
- **Testing Set**: 1,311 images
- **Image Size**: 200x200 pixels (RGB)
- **Classes**: 4 (No Tumor, Glioma, Meningioma, Pituitary)

### Class Distribution

| Class | Training Images | Testing Images | Total |
|-------|----------------|----------------|-------|
| No Tumor | 1,595 | 405 | 2,000 |
| Pituitary | 1,457 | 300 | 1,757 |
| Meningioma | 1,339 | 306 | 1,645 |
| Glioma | 1,321 | 300 | 1,621 |

### Downloading the Dataset

Due to size constraints, the dataset is not included in this repository. You can:

1. **Option 1**: Download from [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. **Option 2**: Contact me for access to the preprocessed dataset
3. **Option 3**: Use your own MRI dataset following the same structure

Place the downloaded dataset in the `data/` folder following this structure:
```
data/
├── Training/
│   ├── notumor/
│   ├── pituitary/
│   ├── meningioma/
│   └── glioma/
└── Testing/
    ├── notumor/
    ├── pituitary/
    ├── meningioma/
    └── glioma/
```

## 🏗️ Model Architecture

The model uses a combination of:

1. **ResNet50** (Transfer Learning)
   - Pre-trained on ImageNet
   - Deep residual learning for image classification
   - 50 layers with skip connections

2. **AutoKeras Image Classifier**
   - Automated neural architecture search
   - Automatic hyperparameter tuning
   - Multiple trials to find optimal configuration

3. **Custom Preprocessing Pipeline**
   - Grayscale conversion
   - Bilateral filtering (noise reduction)
   - Color mapping (COLORMAP_BONE)
   - Resize to 200x200
   - Data augmentation (rotation, shifts, flips)

### Model Flow

```
Input (MRI Image)
    ↓
Preprocessing
    ↓
Data Augmentation
    ↓
ResNet50 Base
    ↓
Global Average Pooling
    ↓
Dense Layers
    ↓
Softmax (4 classes)
    ↓
Output (Classification)
```
---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- 8GB+ RAM

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Follow the instructions in the [Dataset](#dataset) section to download and organize the data.

---

## 💻 Usage

### Training the Model

#### Option 1: Using Jupyter Notebook (Recommended for Beginners)

```bash
jupyter notebook notebooks/brain_tumor_classification.ipynb
```

Then run all cells in sequence.

#### Option 2: Using Python Script

```bash
python src/train.py --epochs 10 --batch_size 32
```

### Making Predictions

```python
from src.model import load_trained_model, predict_tumor

# Load model
model = load_trained_model('models/best_model.h5')

# Make prediction
image_path = 'path/to/mri/image.jpg'
prediction = predict_tumor(model, image_path)

print(f"Predicted class: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Evaluating the Model

```bash
python src/evaluate.py --model models/best_model.h5 --test_data data/Testing/
```

---

## 📈 Results

### Classification Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **96.5%** |
| Precision | 96.3% |
| Recall | 96.4% |
| F1-Score | 96.3% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Tumor | 98.1% | 97.5% | 97.8% | 405 |
| Pituitary | 96.7% | 95.3% | 96.0% | 300 |
| Meningioma | 94.8% | 96.1% | 95.4% | 306 |
| Glioma | 95.3% | 96.7% | 96.0% | 300 |


## 📄 Research Publication

This work has been published and presented at:

**Conference**: IEEE International Conference on Control and Automation (ICCA) 2024

**Title**: "Deep Learning Framework for Brain Tumor Classification Using MRI Images"

**Authors**: Aya Abdel Moniem, et al.

**DOI**: [10.1109/ICCA62237.2024.10927964](https://doi.org/10.1109/ICCA62237.2024.10927964)

**Abstract**: 
This paper proposes a deep learning-based framework for classifying brain tumors using MRI images. The model achieves high accuracy in multi-class tumor detection and demonstrates potential for clinical decision support systems.

### Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{abdelmoniem2024brain,
  title={Deep Learning Framework for Brain Tumor Classification Using MRI Images},
  author={Abdel Moniem, Aya and others},
  booktitle={2024 IEEE International Conference on Control and Automation (ICCA)},
  year={2024},
  organization={IEEE},
  doi={10.1109/ICCA62237.2024.10927964}
}
```

---

## 🔮 Future Work

- [ ] **Improve accuracy** to 98%+ using ensemble methods
- [ ] **Implement explainable AI** (Grad-CAM, LIME) to visualize what the model "sees"
- [ ] **Deploy as web application** using Flask/FastAPI
- [ ] **Mobile app** for on-device inference
- [ ] **Multi-modal learning** combining MRI with CT scans
- [ ] **3D MRI classification** using volumetric data
- [ ] **Real-time inference** optimization
- [ ] **Clinical validation** with medical professionals
- [ ] **Edge deployment** for resource-constrained environments
- [ ] **Uncertainty quantification** for prediction confidence

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.14, Keras
- **AutoML**: AutoKeras
- **Computer Vision**: OpenCV
- **Data Science**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Model Architecture**: ResNet50 (Transfer Learning)
- **Development**: Python 3.8+, Jupyter Notebook

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Improving model accuracy
- Adding new architectures (EfficientNet, Vision Transformers)
- Creating web interface
- Improving documentation
- Adding unit tests
- Optimizing inference speed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Aya Abdel Moniem**

- 📧 Email: Aya.abdelmoniem.afattah@gmail.com
- 💼 LinkedIn: [linkedin.com/in/aya-abdel-moniem-737552214](https://linkedin.com/in/aya-abdel-moniem-737552214)
---

## 🙏 Acknowledgments

- Dataset provided by [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar)
- ResNet50 architecture from [Keras Applications](https://keras.io/api/applications/)
- AutoKeras library by [DATA Lab at Texas A&M University](https://autokeras.com/)
- IEEE ICCA 2024 for accepting our research paper
- The British University in Egypt for academic support

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover this work.

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/brain-tumor-detection&type=Date)](https://star-history.com/#YOUR_USERNAME/brain-tumor-detection&Date)

---

## 📊 Project Status

![GitHub last commit](https://img.shields.io/github/last-commit/YOUR_USERNAME/brain-tumor-detection)
![GitHub issues](https://img.shields.io/github/issues/YOUR_USERNAME/brain-tumor-detection)
![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/brain-tumor-detection)
![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/brain-tumor-detection)

---

<p align="center">
  <strong>Made with ❤️ by Aya Abdel Moniem</strong>
</p>

<p align="center">
  <sub>Part of my M.Sc. research in Communication and Information Engineering</sub>
</p>

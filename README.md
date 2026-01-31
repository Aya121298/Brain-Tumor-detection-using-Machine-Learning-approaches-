# Brain Tumor Detection using AutoKeras

## Overview
This project applies **AutoKeras**, an AutoML framework, to detect and classify brain tumors from MRI images.  
It automates model selection and hyperparameter tuning, achieving high accuracy with minimal manual intervention.

## Publication
- https://doi.org/10.1109/ICCA62237.2024.10927964
  
## Dataset
- Source: [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- Classes: Glioma, Meningioma, Pituitary, No Tumor

## Tech Stack
- Python
- AutoKeras
- TensorFlow/Keras
- Matplotlib, Seaborn (visualization)

## Workflow
1. Data preprocessing (resize, normalize, augment images)
2. AutoKeras ImageClassifier for automated model search
3. Model evaluation (accuracy, confusion matrix, ROC curves)
4. Save and deploy trained model

## Results
- Accuracy: ~99% 

## Future Work
- Deploy as a Flask/Django web app
- Integrate with medical imaging pipelines
- Compare AutoKeras with custom CNNs

## Author
Aya Abdel Moniem  
M.Sc. Communication & Information Engineering  
LinkedIn: [Aya Abdel Moniem](https://www.linkedin.com/in/aya-abdel-moniem-737552214)

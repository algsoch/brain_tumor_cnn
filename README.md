# Brain Tumor Classification With Deep Learning

A deep learning pipeline to classify brain MRI images as tumor or healthy using Keras/TensorFlow, with reproducible Colab notebooks and training/testing results.

---

## 🚀 Project Overview

- Task: Automated brain MRI diagnosis — Tumor vs. Healthy
- Dataset: 2,427 tumor images & 2,087 healthy images (provide source/link if public)
- Model: EfficientNetB3 transfer learning, custom augmentation, fine-tuning stages
- Achieved Test Accuracy: **97.94%** (664/678 test images correctly classified)
- All code and results provided in Jupyter/Colab notebooks

---

## 📁 Repository Structure

├── notebooks/
│ └── brain_tumor_classification.ipynb # Main Colab notebook
├── final_brain_tumor_model_97.keras # Saved trained model
├── model_predictions.csv # Test predictions CSV
├── training_history1.csv # Head training history
├── training_history2.csv # Fine-tuning history
├── images/ # Example sample/test images (optional)
├── README.md # This file
---

## ⚡ Quick Start

### In Colab

1. **Open notebook:**  
   [Colab link](https://colab.research.google.com/github/yourusername/yourrepo/blob/main/notebooks/brain_tumor_classification.ipynb)

2. **Run all cells** to reproduce results, train your own model, or use the saved weights for inference.

### Local (VS Code, Python)

1. Clone this repo.
2. Install requirements:
    ```
    pip install -r requirements.txt
    ```
3. Run inference or batch predictions (see notebook or `scripts/inference.py`).

---

## 🧑‍💻 Main Steps

- Data loading and cleaning
- Augmentation and preprocessing
- Train basic CNN, then EfficientNetB3 (with validation)
- Fine-tune best layers of pretrained model
- Save results/predictions as CSV
- Generate all accuracy/loss/diagnostic graphs

---

## 📊 Outputs

- Model prediction CSV, ready for research analysis or dashboard integration
- Training history CSVs for all learning curves/graphs
- Example code for random image sampling and visualization

---

## 📝 How to Use the Files

- **model_predictions.csv**: For evaluation graphs, reporting, sample gallery, and error analysis.
- **training_history1/2.csv**: For epoch-wise training/validation performance plots.
- **final_brain_tumor_model_97.keras**: For inference, API deployment, or frontend integration.

---

## 🔬 Research & Reproducibility

- Includes complete training/testing workflow for reliable replication
- All results and graphs are based on test/held-out split for honest evaluation
- Recommended for medical ML researchers, students, or open-source practitioners

---

## 🤝 Contributors & Credits

- Author: [vicky kumar](https://www.linkedin.com/in/algsoch)
- Affiliation: IIT Madras B.S. Data Science
- Thanks to open-source datasets and TensorFlow/Keras devs

---


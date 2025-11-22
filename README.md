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

<div style="background-color: #1e1e1e; color: #d4d4d4; border-radius: 8px; padding: 20px; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 14px; line-height: 1.5; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #333;">
  <div style="margin-bottom: 15px; font-weight: bold; font-size: 16px; color: #569cd6; border-bottom: 1px solid #333; padding-bottom: 10px;">
    📂 Repository Structure
  </div>
  <pre style="margin: 0; white-space: pre; color: #d4d4d4;">
├── <span style="color: #E8BD36;">📁 notebooks/</span>
│   └── <span style="color: #ce9178;">brain_tumor_classification.ipynb</span>  <span style="color: #6a9955;"># Main Colab notebook</span>
├── <span style="color: #569cd6;">📄 final_brain_tumor_model_97.keras</span>   <span style="color: #6a9955;"># Saved trained model</span>
├── <span style="color: #4ec9b0;">📊 model_predictions.csv</span>              <span style="color: #6a9955;"># Test predictions CSV</span>
├── <span style="color: #4ec9b0;">📈 training_history1.csv</span>              <span style="color: #6a9955;"># Head training history</span>
├── <span style="color: #4ec9b0;">📉 training_history2.csv</span>              <span style="color: #6a9955;"># Fine-tuning history</span>
├── <span style="color: #E8BD36;">🖼️ images/</span>                            <span style="color: #6a9955;"># Example sample/test images (optional)</span>
└── <span style="color: #569cd6;">📝 README.md</span>                          <span style="color: #6a9955;"># This file</span>
  </pre>
</div>


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


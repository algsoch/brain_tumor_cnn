# Brain Tumor Classification with Deep Learning

A reproducible pipeline to classify brain MRI images as "tumor" or "healthy" using Keras / TensorFlow and transfer learning (EfficientNetB3). This repository contains Colab notebooks, saved model weights, training histories, and example predictions to help you reproduce and extend the results.

- Task: Binary classification — Tumor vs Healthy MRI
- Dataset size (used here): 2,427 tumor images & 2,087 healthy images
- Model: EfficientNetB3 (transfer learning) with custom augmentation and fine-tuning
- Reported test accuracy: **97.94%** (664 / 678)

Table of contents
- Project overview
- Results snapshot
- Repository structure
- Quick start (Colab + Local)
- Usage (inference example)
- Reproducibility & training notes
- File descriptions
- Contributing, citation & contact

---

## Project overview

This project demonstrates how to build, train, fine-tune, and evaluate a CNN-based classifier for brain MRI images using transfer learning. The pipeline includes:
- Data loading and cleaning
- On-the-fly augmentation and preprocessing
- Training a head model, then fine-tuning selected pretrained layers
- Saving model, predictions and training histories for analysis

The primary notebook is prepared for Google Colab so you can reproduce experiments without a local GPU.

---

## Results snapshot

- Test accuracy: 97.94% (664/678 correct)
- Saved model: final_brain_tumor_model_97.keras
- Outputs produced: per-image predictions CSV, head training history, fine-tuning history, and diagnostic plots (accuracy / loss curves, confusion matrix)

Note: All reported numbers are from the held-out test split included in the notebook workflow.

---

## Repository structure

Root files
- notebooks/
  - brain_tumor_classification.ipynb — Main Colab notebook with training and evaluation pipeline
- final_brain_tumor_model_97.keras — Saved trained model (Keras format)
- model_predictions.csv — CSV with test predictions and probabilities
- training_history1.csv — Head training history (epochs, loss, acc, val_loss, val_acc)
- training_history2.csv — Fine-tuning history
- images/ — Example sample/test images (optional)
- README.md — This file

---

## Quick start

### Run in Colab (recommended)
1. Open the notebook in Colab:
   https://colab.research.google.com/github/algsoch/brain_tumor_cnn/blob/main/notebooks/brain_tumor_classification.ipynb
2. Follow the notebook cells (mount Drive if needed), run all cells to reproduce training, evaluation, and plots.
3. The notebook is set up to save artifacts (model, histories, predictions) to the runtime or mounted Drive.

### Run locally
1. Clone the repository:
   git clone https://github.com/algsoch/brain_tumor_cnn.git
2. Optional: create a virtual environment (venv / conda)
3. Install dependencies:
   pip install -r requirements.txt
4. Inspect the notebook or run the provided inference script (if present) for batch predictions.

---

## Usage — quick inference example

Below is a minimal example showing how to load the saved model and run a prediction on a single image (adapt paths as needed):

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("final_brain_tumor_model_97.keras")

# Prepare an image (example)
img_path = "images/example_mri.jpg"
img = image.load_img(img_path, target_size=(300, 300))  # adjust size to model's expected input
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)  # output depends on final activation (sigmoid/softmax)
print("Raw model output:", pred)

# Example interpretation for binary sigmoid:
prob_tumor = float(pred[0][0])
print(f"Probability tumor: {prob_tumor:.4f}")
print("Predicted label:", "Tumor" if prob_tumor > 0.5 else "Healthy")
```

Adjust preprocessing (rescaling, normalization, input size) to exactly match what the notebook uses.

---

## Reproducibility & training notes

- The notebook contains the complete training workflow: data splits, augmentations, model building, training schedules (head training then fine-tuning), and saving outputs.
- For deterministic runs, set random seeds for Python, NumPy and TensorFlow, and use a fixed training/validation/test split (the notebook shows how).
- GPU is recommended for training (Colab-pro or local GPU). For inference, CPU is adequate.
- Hyperparameters, augmentation recipes, and learning rate schedules are all included in the notebook for transparency.

---

## File descriptions

- notebooks/brain_tumor_classification.ipynb — Main Colab notebook (data, model, training, eval, plots)
- final_brain_tumor_model_97.keras — Keras model file saved after training/fine-tuning
- model_predictions.csv — Per-sample predictions on the test set (labels, probabilities)
- training_history1.csv — Training history for the initial head training stage
- training_history2.csv — Training history for the fine-tuning stage
- images/ — Optional: example input images used for visualization

---

## Limitations & ethical considerations

- This project is a technical demonstration and not meant for clinical diagnosis. Do not use model outputs as a substitute for professional medical advice.
- Dataset provenance and labeling quality strongly influence model performance. Please verify dataset licensing and patient-consent details if you plan to use or publish derived results.
- Consider fairness, bias and appropriate validation on diverse cohorts before any real-world deployment.

---

## Contributing

Contributions are welcome. Suggested ways to help:
- Improve preprocessing, augmentations, or architectures
- Add unit tests and CI (Actions) to validate notebooks/scripts
- Provide clearer dataset sourcing or scripts to prepare the dataset
- Add a Dockerfile or Binder support for reproducible local runs

Please open issues or PRs against this repository.

---

## Citation / Author

Author: Vicky Kumar — https://www.linkedin.com/in/algsoch  
Affiliation: IIT Madras — B.S. Data Science

If you use this work in research, please cite the repository and include model/training details in your methods.

---

## License

No license file is included in this repository. If you intend to reuse this work, please request permission from the author or add a license to the repo.

---

## Contact

For questions, issues, or collaboration: https://www.linkedin.com/in/algsoch

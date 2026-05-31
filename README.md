
# AI-Based Incident Image and Video Classification

This project classifies uploaded images and short videos as **Incident** or **Non-Incident** using deep learning.

It was developed as an academic decision-support prototype. It does **not** contact emergency services and must not replace human judgement.

## Main Features

- Baseline CNN model
- MobileNetV2 transfer learning model
- ResNet50 transfer learning model
- Model comparison using accuracy, precision, recall, F1-score, confusion matrix, and inference time
- Frame-based video classification using OpenCV
- Grad-CAM explainability for uploaded images
- Grad-CAM explanations for selected video frames
- Streamlit web interface
- Google Drive backup support for trained models and results

## Project Structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── best_incident_model.keras
├── results/
│   ├── model_comparison_results.csv
│   ├── split_summary.csv
│   └── confusion_matrix images
└── notebook.ipynb
```

## How to Run the Streamlit App

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Make sure the trained model is located here:

```text
models/best_incident_model.keras
```

3. Run the app:

```bash
streamlit run app.py
```

## Google Drive Backup in Colab

The notebook saves important project files to:

```text
My Drive/MCS_AI_Project/
```

The best model is backed up to:

```text
My Drive/MCS_AI_Project/models/best_incident_model.keras
```

If Colab disconnects, restore the model by running:

```python
restore_best_model_from_drive()
```

## Important Note

This system is a university prototype for research and development purposes only. It should not be used as a real emergency response, policing, or automated decision-making system.

# Rotten vs Fresh Classification with Grad-CAM

This project is a computer vision application that classifies images of food as **fresh** or **rotten**, and provides visual explanations using **Grad-CAM**. The system is deployed with an interactive interface built using Streamlit.

## Overview

The goal of this project is not only to perform image classification, but also to improve model interpretability. By using Grad-CAM, it is possible to visualize which regions of the image contributed most to the model's decision.

This is particularly useful for:

* Model validation
* Debugging misclassifications
* Increasing trust in predictions

## Features

* Binary classification: Fresh vs Rotten
* Deep learning model implemented in PyTorch
* Visual explanations using Grad-CAM++
* Interactive web interface using Streamlit
* Support for custom image uploads
* Automatic confidence score display

## Project Structure

```bash
.
├── data/
│   └── raw/dataset/
├── outputs/
│   └── gradcam/
├── src/
│   └── dataset.py
│   └── model.py
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ALauraFM/rotten-fresh-classification
cd rotten-fresh-classification
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser at:

```bash
http://localhost:8501
```

## How It Works

1. The user uploads an image via the Streamlit interface
2. The image is preprocessed and passed to the trained model
3. The model predicts whether the item is fresh or rotten
4. Grad-CAM++ generates a heatmap highlighting important regions
5. The result is displayed with:

   * Predicted label
   * Confidence score
   * Visual explanation (heatmap overlay)

## Grad-CAM Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) uses gradients flowing into the final convolutional layer to produce a coarse localization map of important regions in the image.

This helps answer:

> "Why did the model make this prediction?"

## Technologies Used

* Python
* PyTorch
* Torchvision
* Streamlit
* NumPy
* Matplotlib
* PIL
* pytorch-grad-cam

## Notes

* The `.env` and `.vscode` folders are excluded via `.gitignore`
* Generated images (Grad-CAM outputs) are stored in `outputs/`
* The model architecture can be customized or replaced

## Future Improvements

* Support for multi-class classification
* Model performance dashboard
* Deployment to cloud (Streamlit Cloud or Docker)
* Integration with real-time camera input

## Author

Ana Laura

---

This project is intended for academic and research purposes, focusing on explainability in computer vision and machine learning systems.

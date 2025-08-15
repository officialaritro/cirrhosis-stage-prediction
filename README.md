# Cirrhosis Stage Prediction using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning project to predict the stage of cirrhosis in patients based on clinical data. This project uses an XGBoost classifier to achieve high accuracy and provides a detailed analysis of the model's performance.

## Overview

Cirrhosis is a late-stage liver disease, and early detection of its stage is crucial for patient treatment and management. This project addresses this challenge by leveraging a dataset of clinical features to train a predictive model. The primary goal is to build a robust classifier that can accurately determine the stage of cirrhosis, assisting healthcare professionals in making informed decisions.

The project covers the entire machine learning pipeline, from data preprocessing and feature engineering to model training, hyperparameter tuning, and performance evaluation.

## Features

- **Data Preprocessing**: Handles missing values and uses label encoding to prepare categorical data for the model.
- **Feature Engineering**: Selects and engineers over 15 clinical features to improve predictive power.
- **Advanced Model Training**: Implements an **XGBoost Classifier**, a powerful and efficient gradient-boosting algorithm.
- **Hyperparameter Optimization**: Uses **GridSearchCV** to systematically find the best hyperparameters, boosting the model's F1-score and accuracy.
- **Comprehensive Evaluation**: Measures performance using a full suite of metrics, including **Accuracy, Precision, Recall, and F1-Score**.
- **Detailed Analysis**: Generates a **Classification Report** and a **Confusion Matrix** for an in-depth understanding of the model's predictions across different stages.

## Tech Stack

- **Language**: **Python**
- **Core Libraries**:
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  - **Scikit-learn**: For data preprocessing, model evaluation, and hyperparameter tuning (`GridSearchCV`).
  - **XGBoost**: For the core classification model.
- **Visualization**:
  - **Matplotlib** & **Seaborn**: For creating insightful plots and charts.
- **Environment**:
  - **Jupyter Notebook**: For interactive development and analysis.

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. You can generate one using `pip freeze > requirements.txt` in your project environment.)*

## Usage

The primary code is available in the `cirrhosis.ipynb` Jupyter Notebook.

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open `cirrhosis.ipynb`.
3.  You can run the cells sequentially to see the entire process, from data loading to the final model evaluation.

## Model Performance

The trained XGBoost model achieved a **weighted average F1-score of 92%**, demonstrating its effectiveness in classifying the different stages of cirrhosis.

### Key Metrics:

- **Accuracy**: [Enter Your Accuracy Score Here, e.g., 91.5%]
- **Weighted Avg Precision**: [Enter Your Precision Score Here]
- **Weighted Avg Recall**: [Enter Your Recall Score Here]
- **Weighted Avg F1-Score**: 0.92

### Confusion Matrix

The confusion matrix below visualizes the performance of the classifier. The diagonal elements show the number of correct predictions for each stage.


*A visual representation of the model's performance in distinguishing between the different cirrhosis stages.*

## Project Structure

```
.
├── data/
│   └── cirrhosis.csv        # The dataset used for training
├── notebooks/
│   └── cirrhosis.ipynb      # Jupyter Notebook with the full analysis and model
├── README.md                # This README file
├── requirements.txt         # List of Python dependencies
└── .gitignore               # To exclude unnecessary files from Git
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
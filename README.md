# Plagiarism Detection using Machine Learning



## Overview

This repository contains a plagiarism detection system using **Random Forest Classifier** and **LSTM** models. The system identifies whether a given text is paraphrased using numerical vector representations of source text and its potential plagiarized version.



## Features

- **Random Forest Classifier** for traditional ML-based detection

- **LSTM Model with Hyperparameter Tuning** for deep learning-based detection

- **TF-IDF and Word Embeddings** for feature extraction

- **Hyperparameter tuning using Keras Tuner**

- **Early stopping mechanism** for optimal model performance



---



## Installation

### **Clone the Repository**

```bash

git clone https://github.com/your-username/your-repository.git

cd your-repository

```



### **Install Dependencies**

```bash

pip install -r requirements.txt

```



---



## Dataset Preparation

Ensure that your dataset has two vectorized representations:

- **`source_vec`**: Vectorized representation of the original text.

- **`plagiarism_vec`**: Vectorized representation of the potentially plagiarized text.

- **`label`**: Binary label (1 for plagiarism, 0 for non-plagiarism).



---



## Model 1: Random Forest Classifier



### **Training the Model**

- Combines `source_vec` and `plagiarism_vec` into a single feature set.

- Splits the dataset into training and testing sets (80/20 split).

- Trains a **Random Forest Classifier** with 100 estimators.



### **Evaluation Metrics**

- **Accuracy:**  

- **Precision, Recall, F1-score:**  



### **Run the Random Forest Model**

```bash

python random_forest_model.py

```



---



## Model 2: LSTM Model with Hyperparameter Tuning



### **Training the Model**

- Converts `source_vec` and `plagiarism_vec` into a **3D tensor** for LSTM processing.

- Uses **Hyperparameter tuning (Keras Tuner)** to optimize:

  - Number of LSTM layers (1-6)

  - Dropout rates (0.2-0.5)

  - Learning rate (1e-5 to 1e-2)

  - Optimizer (Adam, RMSprop)

- Implements **early stopping** to prevent overfitting.



### **Evaluation Metrics**

- **Accuracy:**  

- **Precision, Recall, F1-score:**  



### **Run the LSTM Model**

```bash

python lstm_model.py

```



---



## Results & Performance



# Model Performance Evaluation

This project focuses on the evaluation of various machine learning models for classification tasks. Below are the performance metrics of the evaluated models:

## Model Evaluation Metrics

### Classification Report

#### Base Model
- **Accuracy:** 54.00%

| Metric          | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|------------------|---------|---------|-----------|--------------|
| Precision        | 0.57    | 0.51    | 0.54      | 0.54         |
| Recall           | 0.51    | 0.57    | 0.54      | 0.54         |
| F1-Score         | 0.54    | 0.54    | 0.54      | 0.54         |
| Support          | 106     | 94      | -         | 200          |

### Random Forest (TBD)
- **Accuracy:** XX%
- **Precision:** XX%
- **Recall:** XX%
- **F1-Score:** XX%

### LSTM (Tuned, TBD)
- **Accuracy:** XX%
- **Precision:** XX%
- **Recall:** XX%
- **F1-Score:** XX%

## Project Overview
This repository includes:
1. Implementation of different machine learning models.
2. Comparison and analysis of their performance based on metrics such as accuracy, precision, recall, and F1-score.

## Getting Started
To reproduce the results, follow the steps below:
1. Clone this repository:



---



## Future Improvements

✅ Improve accuracy by using **Bidirectional LSTM (BiLSTM)**.  

✅ Train on **larger datasets** for better generalization.  

✅ Experiment with **Transformer models** like BERT for better text understanding.  

✅ Deploy as an **API for real-time plagiarism detection**.  



---



## Contact

📌 **Author:** Raju kumar 

📧 **Email:** raju705080@gmail.com 


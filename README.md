# Credit Card Transaction Fraud Detection Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![Status](https://img.shields.io/badge/Status-Completed-green.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
###
![Dashboard](https://github.com/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection-Project/blob/main/Images/cc.png?raw=true)
## 🚀 Project Overview

This project focuses on detecting fraudulent credit card transactions using both traditional Machine Learning models and a Convolutional Neural Network (CNN).  
Due to the dataset's high imbalance, the focus was placed on **Recall**, **F1-score**, and **ROC-AUC** rather than just accuracy.  

---

## 📊 Dataset Overview
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total transactions**: 284,807  
- **Fraudulent transactions**: 492 (0.172%)  
- **Features**: V1 - V28 (PCA transformed), Time, Amount, and Class (0 = Legit, 1 = Fraud)  

---

## 🔎 Project Pipeline
1. Data Loading & Exploration  
2. Preprocessing (Scaling, Balancing, Train-Test Split)  
3. Exploratory Data Analysis (EDA)  
4. Model Training & Evaluation across **10 different models**  
5. Performance Comparison and Conclusion  

---

## ✅ Model Evaluation Summary (All 10 Tests)

| Test | Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------------------|----------|-----------|--------|----------|---------|
| 1    | Logistic Regression  | 99.93%   | 85.71%    | 57.14% | 68.57%   | 0.936   |
| 2    | Decision Tree        | 99.92%   | 75.00%    | 61.90% | 68.75%   | 0.905   |
| 3    | Random Forest        | 99.97%   | 91.67%    | 71.42% | 80.00%   | 0.946   |
| 4    | XGBoost              | 99.97%   | 91.66%    | 76.19% | 83.33%   | 0.971   |
| 5    | SVM                  | 99.93%   | 80.00%    | 66.66% | 72.72%   | 0.920   |
| 6    | K-Nearest Neighbors  | 99.93%   | 76.92%    | 61.90% | 68.75%   | 0.902   |
| 7    | Naive Bayes          | 99.84%   | 16.21%    | 71.42% | 26.45%   | 0.837   |
| 8    | Gradient Boosting    | 99.97%   | 91.66%    | 76.19% | 83.33%   | 0.969   |
| 9    | LightGBM             | 99.97%   | 91.66%    | 76.19% | 83.33%   | 0.968   |
| 10   | CNN (Deep Learning)  | 99.97%   | 90.00%    | 76.19% | 82.35%   | 0.965   |

---

## 🎯 Key Takeaways
- Accuracy alone is misleading for imbalanced datasets; **Recall**, **F1-score**, and **AUC** are critical.  
- **XGBoost**, **Gradient Boosting**, and **LightGBM** consistently performed the best.  
- The custom-built **CNN** model showed excellent results, proving deep learning can complement traditional models.  
- **Naive Bayes** achieved high recall but with excessive false positives.  

---

## 🛠 Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM  
- TensorFlow & Keras (for CNN)  

---

## 📁 Project Structure

---

## ▶️ How to Run Locally
1. Clone the repository:  
2. Open the notebook in Jupyter or Google Colab.  
3. Download the dataset from Kaggle and place it in the project directory.  
4. Run all cells sequentially to reproduce the results.  

---

## 📬 Contact
- **LinkedIn**: [Shaikh Borhan Uddin](https://www.linkedin.com/in/shaikh-borhan-uddin/)  
- **GitHub**: [ShaikhBorhanUddin](https://github.com/ShaikhBorhanUddin)  

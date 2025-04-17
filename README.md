# Credit Card Transaction Fraud Detection Project
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]() [![Status](https://img.shields.io/badge/Status-Completed-green.svg)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

![Model](https://img.shields.io/badge/Model-Logistic%20Regression-blue.svg) 
[![Logistic Regression](https://img.shields.io/badge/-Random%20Forest-green.svg)]() [![Decision Tree](https://img.shields.io/badge/-Decision%20Tree-orange.svg)]() [![KNN](https://img.shields.io/badge/-KNN-purple.svg)]() [![SVM](https://img.shields.io/badge/-SVM-red.svg)]() [![Naive Bayes](https://img.shields.io/badge/-Naive%20Bayes-yellow.svg)]() [![XGBoost](https://img.shields.io/badge/-XGBoost-brightgreen.svg)](https://xgboost.readthedocs.io/) [![LightGBM](https://img.shields.io/badge/-LightGBM-lightgrey.svg)](https://lightgbm.readthedocs.io/) [![CatBoost](https://img.shields.io/badge/-CatBoost-blueviolet.svg)](https://catboost.ai/) [![CNN](https://img.shields.io/badge/-CNN-deeppink.svg)]()

###
![Dashboard](https://github.com/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection-Project/blob/main/Images/credit_card_image.png?raw=true)
## 🚀 Project Overview

Credit card fraud continues to be a major challenge for both consumers and financial institutions, leading to billions of dollars in losses annually. With the growing volume of online transactions, detecting fraudulent activity in real time has become more critical than ever. This project addresses this pressing issue by leveraging machine learning to develop a data-driven solution aimed at identifying fraudulent credit card transactions with greater accuracy and efficiency.

## 📊 Dataset
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total transactions**: 284,807  
- **Fraudulent transactions**: 492 (0.172%)  
- **Features**: V1 - V28 (PCA transformed), Time, Amount, and Class (0 = Legit, 1 = Fraud)  

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It includes transactions from two days, with a total of 284,807 transactions, of which 492 are identified as fraudulent. This dataset is highly unbalanced, as the fraudulent transactions (positive class) represent only 0.172% of all transactions.

The dataset comprises only numerical input variables resulting from a PCA (Principal Component Analysis) transformation. Due to confidentiality issues, we cannot provide the original features or additional background information about the data. The features labeled V1, V2, …, V28 are the principal components obtained through PCA. The features that have not undergone PCA transformation are 'Time' and 'Amount.' The 'Time' feature indicates the seconds elapsed from the first transaction in the dataset, while the 'Amount' feature represents the transaction amount, which can be utilized for example-dependent cost-sensitive learning. The 'Class' feature serves as the response variable, with a value of 1 indicating fraud and 0 indicating a legitimate transaction.
## 📁 Project Structure
```bash
├── Images/
├── dataset/            # Dataset files too large to upload in guithub
├── src/                # Source code
│   ├── images
│   └── Credit_Card_Fraud_Detection.ipynb
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── LICENSE
```
## 🔎 Project Pipeline
1. Data Loading & Exploration  
2. Preprocessing (Scaling, Balancing, Train-Test Split)  
3. Exploratory Data Analysis (EDA)  
4. Model Training & Evaluation across **10 different models**  
5. Performance Comparison and Conclusion  

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

## ROC Curve Comparison
![ROC Curve](https://github.com/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection-Project/blob/main/Images/ROC.png?raw=true)

## 🎯 Key Takeaways
- Accuracy alone is misleading for imbalanced datasets; **Recall**, **F1-score**, and **AUC** are critical.  
- **XGBoost**, **Gradient Boosting**, and **LightGBM** consistently performed the best.  
- The custom-built **CNN** model showed excellent results, proving deep learning can complement traditional models.  
- **Naive Bayes** achieved high recall but with excessive false positives.  

## 🛠 Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost, LightGBM  
- TensorFlow & Keras (for CNN)  

## ▶️ How to Run Locally
1. Clone the repository:  
2. Open the notebook in Jupyter or Google Colab.  
3. Download the dataset from Kaggle and place it in the project directory.  
4. Run all cells sequentially to reproduce the results.  

## 📄 License

This project is licensed under the [MIT License](LICENSE).


## 📬 Contact
- **LinkedIn**: [Shaikh Borhan Uddin](https://www.linkedin.com/in/shaikh-borhan-uddin/)  
- **GitHub**: [ShaikhBorhanUddin](https://github.com/ShaikhBorhanUddin)  

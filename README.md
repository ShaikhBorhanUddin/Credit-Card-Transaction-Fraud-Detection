# Credit Card Transaction Fraud Detection  
<p align="left">
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white&label=Made%20With" alt="Made with Colab">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection" alt="Repo Size">
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection" alt="Issues">
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn-orange?logo=scikitlearn" alt="Framework: Scikit-Learn">
  <img src="https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle" alt="Dataset: Kaggle">
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection?style=social" alt="Forks">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-blue.svg" alt="Model: Logistic Regression">
  <img src="https://img.shields.io/badge/-Random%20Forest-green.svg" alt="Model: Random Forest">
  <img src="https://img.shields.io/badge/-Decision%20Tree-orange.svg" alt="Model: Decision Tree">
  <img src="https://img.shields.io/badge/-KNN-purple.svg" alt="Model: K-Nearest Neighbors">
  <img src="https://img.shields.io/badge/-SVM-red.svg" alt="Model: Support Vector Machine">
  <img src="https://img.shields.io/badge/-Naive%20Bayes-yellow.svg" alt="Model: Naive Bayes">
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/-XGBoost-brightgreen.svg" alt="Model: XGBoost"></a>
  <a href="https://lightgbm.readthedocs.io/"><img src="https://img.shields.io/badge/-LightGBM-lightgrey.svg" alt="Model: LightGBM"></a>
  <a href="https://catboost.ai/"><img src="https://img.shields.io/badge/-CatBoost-blueviolet.svg" alt="Model: CatBoost"></a>
  <img src="https://img.shields.io/badge/-CNN-deeppink.svg" alt="Model: Convolutional Neural Network">
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git&logoColor=white" alt="Version Control: Git">
  <img src="https://img.shields.io/badge/Host-GitHub-black?logo=github&logoColor=white" alt="Host: GitHub">
  <img src="https://img.shields.io/badge/Project-Completed-brightgreen" alt="Project Status">
</p>


![Dashboard](https://github.com/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection-Project/blob/main/Images/credit_card_image.png?raw=true)

## 🚀 Project Overview

Credit card fraud continues to be a major challenge for both consumers and financial institutions, leading to billions of dollars in losses annually. With the growing volume of online transactions, detecting fraudulent activity in real time has become more critical than ever. This project addresses this pressing issue by leveraging machine learning to develop a data-driven solution aimed at identifying fraudulent credit card transactions with greater accuracy and efficiency. 
A total of 10 different models were implemented and compared to ensure thorough experimentation and performance benchmarking. These include:

`Logistic Regression` `Decision Tree` `Random Forest` `K-Nearest Neighbors (KNN)` `Support Vector Machine (SVM)` `Naive Bayes` `XGBoost` `LightGBM` `Gradient Boosting` `CNN`

Each model was evaluated using precision, recall, F1-score, and ROC-AUC, with special focus on handling class imbalance — a key challenge in fraud detection tasks. This comprehensive approach enhances the reliability and practical relevance of the final solution.

## 📊 Dataset

This project leverages an insightful Fraud Detection Dataset from Kartik2112 on Kaggle, which is essential for training and evaluating robust fraud detection models. Spanning real-world credit card transactions from 1/1/2019 to 31/12/2020, this dataset includes a comprehensive mix of both legitimate and fraudulent activities, encompassing transactions from 1,000 customers across 800 merchants. Generated with the advanced Sparkov Data Generation tool created by Brandon Harris, the data provides a realistic simulation for effective analysis. After running the simulation for the designated period, the resulting files were meticulously combined and standardized. No personally identifiable information (PII) is included. All customer IDs are anonymized. Notably, the 'is_fraud' feature acts as the key response variable, with a value of 1 representing fraud and a value of 0 indicating a legitimate transaction, making it a critical asset for any fraud detection effort.

- Total Rows: 1852393
- Total Features: 23 (including target variable)
- Type: Tabular
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

🔍 Features

| Column Name      | Description                                         |
|------------------|-----------------------------------------------------|
| `unnamed`        | Serial of the transactions                          |
| `trans_date_trans_time`           | Exact time of the transactions in DD-MM-YYYY-HH:MM format |
| `cc_num`         | Credit card number                                  |
| `merchant`       | Merchant shop, where the transaction was completed  |
| `category`       | Type of the product purchased                       |
| `amt`            | Amount of the transaction                           |
| `first`          | First name of customer                              |
| `last`           | Last name of customer                               |
| `gender`         | Male (`M`) or Female (`F`)                          |
| `street`         | Residential address of the card holder              |
| `city`           | Residential city of the card holder                 |
| `state`          | Residential state of the card holder                |
| `zip`            | Postal code of the card holder                      |
| `lat`            | Latitude                                            |
| `long`           | Longitude                                           |
| `city_pop`       | Population of customer's residential city           |
| `job`            | Job descriptionof the customer                      |
| `dob`            | Customer's date of birth                            |
| `trans_num`      | unique transaction id generated while purchasing    |
| `unix_time`      | unix timestamp (the numebr of seconds since January 1, 1970)       |
| `merch_lat`      | Merchant's latitude                                 |
| `merch_long`     | Merchant's longitude                                |
| `isFraud`        | Target variable: `1` if fraudulent, else `0`        |

⚠️ Note

The dataset is imbalanced, with a small proportion of fraudulent transactions. This presents a realistic challenge often faced in fraud detection systems.

## 📁 Project Structure
```bash
├── Images/
├── dataset/            # Dataset files too large to upload in repository
├── src/                # Source code
│   ├── images
│   └── Credit_Card_Fraud_Detection.ipynb
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── LICENSE
```
## 🔄 Project Workflow

1. Data Loading & Exploration  
2. Preprocessing (Scaling, Balancing, Train-Test Split)  
3. Exploratory Data Analysis (EDA)  
4. Model Training & Evaluation across **10 different models**  
5. Performance Comparison and Conclusion  

## 📈 Model Evaluation Summary

Performance Matrices of all 10 tested models are included here.
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

To assess the effectiveness of each model in detecting fraudulent transactions, multiple evaluation metrics were considered — with a strong emphasis on Recall, Precision, F1-Score, and ROC-AUC. These metrics are particularly crucial in fraud detection, where minimizing false negatives (i.e., missing fraudulent transactions) is more important than overall accuracy.

Key takeaways from the evaluation:

-  XGBoost and LightGBM consistently delivered the highest performance across most metrics, showcasing their ability to handle class imbalance and capture complex patterns in the data.
-  Random Forest performed well by balancing precision and recall, making it a reliable model for real-world deployment with proper tuning.
-  Logistic Regression and Support Vector Machine showed solid precision but slightly lower recall, which could result in undetected fraud cases.
-  Simpler models like K-Nearest Neighbors (KNN) and Naive Bayes showed relatively lower performance, likely due to their limitations in handling high-dimensional and imbalanced datasets.

Overall, XGBoost, LightGBM, and Random Forest emerged as the top-performing models, with strong ROC-AUC scores and recall — making them well-suited for identifying rare but critical fraud cases in real-world financial systems.

## 🧩 Confusion Matrices
Confusion Matrix of 9 machine learning models are included here. Click image for enlarged view.

<p align="left">
  <img src="Images/cm_ML_Models.png" width="998"/>
</p>
The confusion matrices for Logistic Regression (LR), Support Vector Machine (SVM), and Stochastic Gradient Descent (SGD) Classifier reveal a major flaw: all three models completely failed to identify any positive class (label 1). They predicted every instance as the negative class (label 0), achieving high true negatives (368,526) but zero true positives. Consequently, they have perfect accuracy for the majority class but a recall of 0% for the minority class, making them highly unreliable for imbalanced classification tasks like fraud or anomaly detection. This failure suggests these models are heavily biased toward the dominant class and cannot distinguish meaningful patterns in the minority class due to poor model sensitivity and lack of class balance handling.

In contrast, the other six models — Decision Tree, Random Forest, K-Nearest Neighbors (KNN), AdaBoost, XGBoost, and LightGBM — all show better capability in identifying the positive class to varying degrees. Among these, XGBoost demonstrated the best overall performance, with 1,334 true positives and only 619 false negatives, along with 142 false positives, achieving a solid balance between precision and recall. Random Forest and LightGBM also performed well, with LightGBM producing 1,272 true positives and 424 false positives. AdaBoost and Decision Tree lag slightly behind, especially in recall. KNN performed the worst among these six, misclassifying a large number of positive instances. Overall, XGBoost stands out due to its strong handling of class imbalance and superior precision-recall trade-off, making it the most reliable model for this classification task.
###

<p align="left">
  <img src="Images/cm_CNN.png" width="400"/>
</p>
This experiment revealed that CNN models struggled to accurately identify spam, much like Logistic Regression and other less effective models. This clearly shows that, while deep learning models excel in various applications, they don't perform well when it comes to analyzing financial data in tabular formats.

## 📊 ROC Curve Comparison
![ROC Curve](https://github.com/ShaikhBorhanUddin/Credit-Card-Transaction-Fraud-Detection-Project/blob/main/Images/ROC.png?raw=true)

In this experiment, ROC curves were generated for all models to compare their ability to distinguish between fraudulent and legitimate transactions. Among all models tested, XGBoost achieved a perfect AUC score of 1.00, indicating ideal classification performance. Random Forest followed closely with an AUC of 0.99, also showing excellent discriminative capability. AdaBoost performed similarly well, with an AUC of 0.98, confirming its robustness in handling the class imbalance in the dataset. LightGBM showed strong predictive power with an AUC of 0.91, surpassing most traditional models. The Decision Tree model recorded an AUC of 0.88, offering decent performance but not at par with the ensemble methods. K-Nearest Neighbors (KNN) delivered moderate results with an AUC of 0.73, while Logistic Regression had the lowest AUC score of 0.55, indicating performance close to random guessing.

Overall, the ROC analysis highlights that ensemble models, particularly XGBoost, Random Forest, and AdaBoost, are highly effective in identifying fraudulent transactions in the dataset, significantly outperforming simpler models like Logistic Regression and KNN.


**`⚠️ Limitations of ROC in Imbalanced Datasets`**

While ROC curves are widely used, they can be misleading in highly imbalanced datasets — such as this one, where fraudulent transactions make up a tiny fraction of the total data. Here’s why:
- False Positive Rate can appear low even when the model makes many false predictions, simply because the number of negative (legit) cases is so high.
- This can result in inflated ROC-AUC scores, giving a false sense of performance.
- A model might achieve a high AUC but still fail to catch most fraud cases — which is unacceptable in real-world scenarios.

For this reason, Precision-Recall curves and Recall-focused metrics (like F1-Score and Recall) are often more reliable indicators of model effectiveness in fraud detection tasks.


## 🎯 Key Takeaways
- Accuracy alone is misleading for imbalanced datasets; **Recall**, **F1-score**, and **AUC** play an important role here.  
- **XGBoost**, **Gradient Boosting**, and **LightGBM** consistently performed the best.  
- The custom-built **CNN** model showed excellent results, proving deep learning can complement traditional models.  
- **Naive Bayes** achieved high recall but with excessive false positives.  

## 🛠 Technologies Used
`Python`  `Pandas` ` NumPy` `Matplotlib` ` Seaborn` `Scikit-learn` `TensorFlow` `Keras`

## ▶️ How to Run Locally
1. Clone the repository:  
2. Open the notebook in Jupyter or Google Colab.  
3. Download the dataset from Kaggle and place it in the project directory.  
4. Run all cells sequentially to reproduce the results.  

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🚀 Future Development

- Implement real-time fraud detection using a streaming data pipeline (e.g., Kafka + Spark).
- Explore advanced models like XGBoost, CatBoost, or deep learning architectures.
- Perform feature engineering to improve detection accuracy.
- Address class imbalance with SMOTE, ADASYN, or cost-sensitive learning.
- Integrate model explainability (e.g., SHAP, LIME) to interpret predictions.
- Deploy the model with a REST API using FastAPI or Flask.
- Build an interactive dashboard for monitoring fraud detection performance.


## 📬 Contact
**LinkedIn**: [`Shaikh Borhan Uddin`](https://www.linkedin.com/in/shaikh-borhan-uddin-905566253/)  
**GitHub**: [`ShaikhBorhanUddin`](https://github.com/ShaikhBorhanUddin)  

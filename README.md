# Shopper App – Online Shoppers' Purchase Intention Prediction

## a. Problem Statement

Online shopping websites receive visits from users with different browsing behaviours. Some visitors complete a purchase, while others leave the website without buying anything.

The objective of **Shopper AI** is to build a machine learning classification system that predicts whether an online shopping session will result in a purchase. The project uses browsing behaviour, page interaction, visitor information and session-related features to classify each visit into one of two classes:

- `Revenue = 1` – The visitor completed a purchase.
- `Revenue = 0` – The visitor did not complete a purchase.

This is a **binary classification problem**.

Accurately predicting purchase intention can help e-commerce businesses understand customer behaviour, improve marketing strategies, identify visitors who may need assistance and make better business decisions.

## b. Dataset Description

This project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

- **Dataset source:** [UCI – Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Dataset file:** `online_shoppers_intention.csv`
- **Total records (original):** 12,330
- **Total attributes:** 18 (17 input features and 1 target variable)
- **Target variable:** `Revenue`
- **Problem type:** Binary classification

Each record represents an online shopping session. The dataset contains information about the visitor's browsing activity, time spent on different page categories, page values, visit month, visitor type, browser, operating system, traffic source and whether the session occurred on a weekend.

### Dataset Features

| Feature | Description |
|---|---|
| Administrative | Number of administrative pages visited |
| Administrative_Duration | Time spent on administrative pages |
| Informational | Number of informational pages visited |
| Informational_Duration | Time spent on informational pages |
| ProductRelated | Number of product-related pages visited |
| ProductRelated_Duration | Time spent on product-related pages |
| BounceRates | Average bounce rate of the pages visited |
| ExitRates | Average exit rate of the pages visited |
| PageValues | Page value based on historical transaction information |
| SpecialDay | Closeness of the visit to a special day |
| Month | Month in which the session occurred |
| OperatingSystems | Operating system category code |
| Browser | Browser category code |
| Region | Visitor region category code |
| TrafficType | Traffic source category code |
| VisitorType | New, returning or other visitor category |
| Weekend | Whether the visit occurred on a weekend |
| Revenue | Target indicating whether a purchase was completed |

### Target Distribution

The original dataset contains:

| Class | Meaning | Records | Percentage |
|---|---|---:|---:|
| 0 | No purchase | 10,422 | 84.55% |
| 1 | Purchase | 1,908 | 15.45% |
| **Total** | | **12,330** | **100%** |

The preprocessing pipeline removes 125 duplicate rows. The cleaned dataset therefore contains 12,205 records. Its target distribution is:

- No purchase (`0`): 10,297 records (84.37%)
- Purchase (`1`): 1,908 records (15.63%)

The dataset is imbalanced, as the number of sessions without a purchase is considerably higher than the number of sessions resulting in a purchase. Therefore, accuracy alone is not sufficient to evaluate the models.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/Ashking1956/BITS-ML-Assignment_2.git
**Project Link:** https://bits-ml-assignment2-qsmzvx94cmyjnew3tbc2xe.streamlit.app/

The repository should contain the following project files:

```text
Shopper-AI/
│
├── app.py
├── ml_pipeline.py
├── content.py
├── online_shoppers_intention.csv
├── README.md
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── feature_columns.json
│   ├── raw_columns.json
│   └── metrics.csv
│
└── test_data.csv
```

The `model/` directory and `test_data.csv` are generated when the training pipeline runs. Include them in the repository if you want to distribute the already-trained models and test data.

## d. Models Used and Evaluation

Five machine learning classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Data Preprocessing and Feature Engineering

The following steps are performed before model training:

1. **Duplicate removal:** Exact duplicate records are removed.
2. **Boolean conversion:** Boolean columns are converted into numerical values (`True = 1`, `False = 0`).
3. **Feature engineering:** Seven additional features are created:
   - `TotalPages`: Total administrative, informational and product-related pages visited.
   - `TotalDuration`: Total time spent on the three page categories.
   - `AvgTimePerPage`: Average time spent per page.
   - `HasPageValue`: Indicates whether `PageValues` is greater than zero.
   - `IsReturning`: Indicates whether the visitor is a returning visitor.
   - `IsHolidaySeason`: Indicates whether the visit occurred in November or December.
   - `log_ProductRelated_Duration`: Log-transformed product-related duration.
4. **Categorical encoding:** Text columns are converted into numerical columns using one-hot encoding with `drop_first=True`.
5. **Train-test split:** The data is split into 80% training and 20% testing data using stratified sampling and `random_state=42`.
6. **Feature scaling:** `StandardScaler` is applied to Logistic Regression, KNN, Naive Bayes and SVM. Decision Tree is trained without scaling.

After preprocessing and encoding, the feature matrix contains 33 input features.

### Model Configurations

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=1000, random_state=42)` with StandardScaler |
| Decision Tree | `DecisionTreeClassifier(max_depth=5, random_state=42)` |
| KNN | `KNeighborsClassifier(n_neighbors=5)` with StandardScaler |
| Naive Bayes | `GaussianNB()` with StandardScaler |
| Support Vector Machine | `SVC(kernel='rbf', random_state=42)` with StandardScaler |

### Evaluation Metrics

The models are evaluated using the following six metrics:

- **Accuracy:** Percentage of total predictions classified correctly.
- **AUC (ROC-AUC):** Measures how well the model distinguishes between purchase and no-purchase sessions across classification thresholds.
- **Precision:** Percentage of predicted purchases that are actual purchases.
- **Recall:** Percentage of actual purchases correctly identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **MCC (Matthews Correlation Coefficient):** Measures the quality of binary classifications while considering true and false positives and negatives. It is particularly useful for imbalanced datasets.

### Model Comparison Table

All values below were calculated by executing the uploaded `ml_pipeline.py` on the supplied dataset. Results are from the same stratified hold-out test set of 2,441 records. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9025** | **0.9233** | **0.7156** | 0.6257 | **0.6676** | **0.6127** |
| Decision Tree | 0.8992 | 0.9155 | 0.6954 | **0.6335** | 0.6630 | 0.6048 |
| KNN | 0.8865 | 0.8716 | 0.6596 | 0.5681 | 0.6104 | 0.5465 |
| Naive Bayes | 0.7898 | 0.8719 | 0.4172 | **0.8639** | 0.5627 | 0.4968 |
| Support Vector Machine (SVM) | 0.9005 | 0.9074 | 0.7062 | 0.6230 | 0.6620 | 0.6056 |

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved the best overall performance. It obtained the highest Accuracy (0.9025), AUC (0.9233), Precision (0.7156), F1-score (0.6676) and MCC (0.6127). Its performance indicates a strong balance between identifying purchase sessions and avoiding false purchase predictions. It is also relatively easy to interpret. |
| **Decision Tree** | Decision Tree achieved an Accuracy of 0.8992 and an AUC of 0.9155. Its Recall of 0.6335 was slightly higher than Logistic Regression's recall. It can capture non-linear relationships and feature interactions, while its maximum depth of five helps limit tree complexity. Its overall metrics were slightly lower than Logistic Regression. |
| **KNN** | KNN achieved an Accuracy of 0.8865 and an F1-score of 0.6104. Its AUC of 0.8716 and MCC of 0.5465 were lower than those of Logistic Regression, Decision Tree and SVM. This suggests that the five-neighbour approach was less effective at separating purchasing and non-purchasing sessions in this feature space. |
| **Naive Bayes** | Naive Bayes achieved the highest Recall (0.8639), meaning it identified a large proportion of actual purchase sessions. However, its Precision was only 0.4172, indicating many false purchase predictions. Its Accuracy (0.7898), F1-score (0.5627) and MCC (0.4968) were the lowest among the five models. It may be useful when identifying as many potential buyers as possible is more important than avoiding false alarms. |
| **Support Vector Machine (SVM)** | SVM achieved an Accuracy of 0.9005, Precision of 0.7062, Recall of 0.6230 and F1-score of 0.6620. Its performance was close to Logistic Regression and Decision Tree. The RBF kernel can capture non-linear decision boundaries, although its AUC (0.9074) and MCC (0.6056) were below Logistic Regression's results. |

### Overall Winner for This Dataset

**Winner: Logistic Regression**

Based on the calculated evaluation metrics, Logistic Regression is the overall winner for this dataset.

It achieved the highest Accuracy (90.25%), AUC (0.9233), Precision (0.7156), F1-score (0.6676) and MCC (0.6127) among all five models. Although Decision Tree achieved slightly higher Recall than Logistic Regression, Logistic Regression provided the best overall balance across the evaluation metrics.

Naive Bayes achieved the highest Recall, but its lower Precision indicates that it generates more false positive purchase predictions.

Therefore, **Logistic Regression is the recommended model for this implementation**, while Naive Bayes may be considered when maximising the identification of potential buyers is the primary objective.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Ashking1956/BITS-ML-Assignment_2.git
cd shopper_app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the five models

Make sure `online_shoppers_intention.csv` is in the project root, then run:

```bash
python ml_pipeline.py
```

This generates the trained model files, feature metadata, evaluation metrics and `test_data.csv`.

You can also supply a CSV path:

```bash
python ml_pipeline.py path/to/online_shoppers_intention.csv
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application provides an interactive walkthrough of dataset exploration, preprocessing, feature engineering, model training, evaluation, model comparison and recommendations.

# Credit Risk Scoring Model

Binary classification pipeline that predicts the probability of loan default (`loan_status`) based on borrower demographics, income, and loan terms.

## Dataset

- **Source:** Credit Risk Dataset (Kaggle)
- **Raw size:** 32,581 rows, 12 columns
- **Cleaned size:** 32,408 rows (`0` = paid: 78.13%, `1` = default: 21.87%)

## Data Preparation & Feature Engineering

1. **Data Cleaning:**
   - Removed 165 duplicate records.
   - Filtered out erroneous outliers (`person_age > 90` and `person_emp_length > 60`).
   - Imputed missing values in `person_emp_length` (895 nulls) and `loan_int_rate` (3,116 nulls) inside a `ColumnTransformer` pipeline to avoid data leakage.

2. **Engineered Features:**
   - `annual_interest_payment`: estimated yearly interest cost (`loan_amnt * loan_int_rate / 100`)
   - `interest_to_income`: annual interest payment divided by annual income (`r = 0.460` correlation with default)
   - `emp_to_age_ratio`: employment duration divided by borrower age

## Model Comparison

Evaluated on a stratified 80/20 train-test split (`25,926` train / `6,482` test rows):

| Model                 | Accuracy (%) | F1-score  |  ROC-AUC  |
| :-------------------- | :----------: | :-------: | :-------: |
| Logistic Regression   |     86.6     |   0.649   |   0.871   |
| Random Forest         |     93.3     |   0.824   |   0.929   |
| **Gradient Boosting** |   **93.8**   | **0.839** | **0.948** |

**GradientBoosting Confusion Matrix (Test Set):**

- True Negatives (Paid correctly predicted): `5,028`
- False Positives (Paid flagged as default): `36`
- False Negatives (Missed defaults): `368`
- True Positives (Defaults caught): `1,050`

## Project Structure

```text
credit-risk-scoring/
├── data/
│   └── credit_risk_dataset.csv
├── models/
│   └── credit_risk_model.pkl
├── notebooks/
│   └── credit_risk_analysis.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `credit_risk_dataset.csv` inside the `data/` directory.
3. Open and run `notebooks/credit_risk_analysis.ipynb` to train the pipeline and export `models/credit_risk_model.pkl`.

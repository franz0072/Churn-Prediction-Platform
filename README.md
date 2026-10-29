# Customer Churn Prediction Platform

A machine learning project that predicts whether a telecom customer is likely to leave a service provider based on customer demographics, account information, and service usage.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Random Forest
- **API:** FastAPI, Uvicorn
- **Model Serialization:** Joblib

## Dataset

IBM Telco Customer Churn Dataset

- Original records: 7,043
- Processed records: 7,032
- Input features: 19
- Target: `Churn` (0 = Stay, 1 = Churn)

## Machine Learning Workflow

1. Load the customer dataset.
2. Remove unnecessary identifiers.
3. Convert `TotalCharges` into numeric values.
4. Handle missing values.
5. Encode categorical features using OneHotEncoder.
6. Split data into training (80%) and testing (20%) sets.
7. Train a Random Forest classifier.
8. Evaluate model performance.
9. Save the trained model using Joblib.
10. Serve predictions through a FastAPI endpoint.

## Model Performance

| Metric | Result |
|---|---|
| Accuracy | 76.26% |
| ROC-AUC | 0.8111 |
| Churn Precision | 55% |
| Churn Recall | 63% |
| Churn F1-score | 58% |

These results are from the held-out test dataset.

## Project Structure

```text
Churn Prediction Platform/
├── api/
│   └── main.py
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed_churn.csv
├── models/
│   └── churn_model.joblib
├── notebooks/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

Prepare the dataset:

```powershell
python .\src\prepare_data.py
```

Train the machine learning model:

```powershell
python .\src\train.py
```

Start the API:

```powershell
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

**GET /**

Returns a message confirming the API is running.

**POST /predict**

Accepts customer information and returns:

- `churn_prediction`: 0 or 1
- `churn_label`: Stay or Churn
- `churn_probability`: Estimated probability of churn

## Future Improvements

- Automated API and model tests
- Improved input validation
- Model comparison and hyperparameter tuning
- Docker deployment
- Laravel and MySQL integration
- Monitoring and model versioning

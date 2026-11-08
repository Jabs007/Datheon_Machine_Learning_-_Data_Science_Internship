# Dataset: Telco Customer Churn

File: `Telco-Customer-Churn.csv` — 7,043 rows x 21 columns.

```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn
```

## Column notes

| Column | Type | Notes |
| --- | --- | --- |
| `customerID` | identifier | 7,043 unique values; dropped before modelling |
| `gender` | categorical | Male / Female |
| `SeniorCitizen` | binary numeric | 0 / 1 |
| `Partner`, `Dependents` | categorical | Yes / No |
| `tenure` | numeric | months as a customer, 0-72 |
| `PhoneService` | categorical | Yes / No |
| `MultipleLines` | categorical | No / Yes / No phone service |
| `InternetService` | categorical | Fiber optic / DSL / No |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical | No / Yes / No internet service |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | categorical | Yes / No |
| `PaymentMethod` | categorical | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
| `MonthlyCharges` | numeric | 18.25 - 118.75 |
| `TotalCharges` | text | coerced to numeric; 11 blank strings, all on `tenure == 0` |
| `Churn` | target | Yes (1,869) / No (5,174) - 26.54% positive |

## Handling

- `TotalCharges` -> numeric; the 11 blanks are `tenure == 0` customers and are
  filled with 0.0 (no charges yet), not the median.
- `Churn` -> 1 for `Yes`, 0 for `No`.
- The 15 categorical columns are one-hot encoded with `drop="first"`, giving 26
  indicator columns; together with the 3 numeric columns that is 29 features.

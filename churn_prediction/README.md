# ChurnBank — Customer Churn Prediction Project

## Project Structure
```
churn_prediction/
├── app.py                  # Main Flask web application
├── generate_and_train.py   # Dataset generation + model training
├── requirements.txt        # Python dependencies
├── data/
│   ├── bank_churn_data.csv # 5000-row synthetic bank dataset
│   └── users.json          # User accounts (auto-created on register)
├── models/
│   ├── random_forest.pkl   # Random Forest model (92.3% accuracy)
│   ├── gradient_boost.pkl  # Gradient Boosting model (93% accuracy)
│   ├── logistic_reg.pkl    # Logistic Regression model (85.1% accuracy)
│   ├── scaler.pkl          # Feature scaler
│   ├── le_geography.pkl    # Label encoder - geography
│   ├── le_gender.pkl       # Label encoder - gender
│   ├── le_contract.pkl     # Label encoder - contract
│   └── feature_names.pkl   # Feature list
└── templates/
    ├── base.html           # Base layout with navbar
    ├── register.html       # Registration page
    ├── login.html          # Login page
    ├── dashboard.html      # Home dashboard
    ├── churn.html          # Churn Prediction (main feature)
    ├── deposit.html        # Deposit money
    ├── withdraw.html       # Withdraw money
    ├── transfer.html       # Fund transfer
    ├── loan.html           # Loan application + EMI calculator
    ├── fixed_deposit.html  # Fixed deposit creation
    ├── pay_bills.html      # Bill payments
    ├── history.html        # Transaction history
    └── profile.html        # User profile + password change
```

## Quick Start (3 steps)

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML models (only needed once)
```bash
python generate_and_train.py
```
This will:
- Generate 5,000 synthetic bank customer records
- Train 3 ML models (Random Forest, Gradient Boosting, Logistic Regression)
- Save all models to the /models/ folder
- Print model accuracy scores

### Step 3 — Run the web app
```bash
python app.py
```
Then open your browser at: **http://127.0.0.1:5000**

---

## How to Use

1. **Register** — Create an account with your name, email, any password
2. **Login** — Sign in with your credentials
3. **Banking** — Use Deposit, Withdraw, Transfer, Loan, FD, Pay Bills
4. **Churn Prediction** — Click "Churn Prediction" tab:
   - Fields are auto-filled from your banking activity
   - Adjust any values if needed
   - Click **Predict Churn Risk**
   - See: probability %, risk level, model breakdown, top factors, recommendations

---

## ML Models Used

| Model | Algorithm | Accuracy |
|-------|-----------|----------|
| Random Forest | Ensemble of 200 decision trees | 92.3% |
| Gradient Boosting | Sequential boosting trees | 93.0% |
| Logistic Regression | Linear classifier | 85.1% |
| **Final (Ensemble)** | Weighted average of all 3 | **~93%** |

## Features Used for Prediction (15 total)

| Feature | Description |
|---------|-------------|
| tenure | Months as bank customer |
| age | Customer age |
| balance | Account balance |
| salary | Annual income |
| credit_score | Credit score (300-850) |
| products | Number of bank products |
| has_card | Has credit card |
| is_active | Active bank member |
| monthly_charges | Average monthly fees |
| complaints | Support complaints in 6 months |
| geography | City (Hyderabad/Mumbai/Delhi/Bangalore/Chennai) |
| gender | Male / Female |
| contract | Month-to-Month / One Year / Two Year |
| loan | Has active loan |
| fd | Has fixed deposit |

## Churn Risk Levels

- 🟢 **Low Risk** (0–35%) — Customer likely to stay
- 🟡 **Medium Risk** (35–65%) — Customer at moderate risk
- 🔴 **High Risk** (65–100%) — Customer likely to churn

---
Built with Python, Flask, scikit-learn | Dataset: 5,000 synthetic bank records

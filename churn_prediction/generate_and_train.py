import pandas as pd
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 5000

tenure       = np.random.randint(1, 73, N)
age          = np.random.randint(21, 70, N)
balance      = np.random.uniform(0, 250000, N)
salary       = np.random.uniform(20000, 200000, N)
credit_score = np.random.randint(300, 850, N)
products     = np.random.randint(1, 5, N)
has_card     = np.random.choice([0, 1], N)
is_active    = np.random.choice([0, 1], N, p=[0.3, 0.7])
monthly_charges = np.random.uniform(500, 5000, N)
complaints   = np.random.choice([0,1,2,3,4], N, p=[0.6,0.2,0.1,0.06,0.04])
geography    = np.random.choice(['Hyderabad','Mumbai','Delhi','Bangalore','Chennai'], N)
gender       = np.random.choice(['Male','Female'], N)
contract     = np.random.choice(['Month-to-Month','One Year','Two Year'], N, p=[0.5,0.3,0.2])
loan         = np.random.choice([0,1], N, p=[0.4,0.6])
fd           = np.random.choice([0,1], N, p=[0.5,0.5])

churn_prob = (
    0.30 * (tenure < 12) +
    0.20 * (monthly_charges > 3500) +
    0.15 * (complaints >= 2) +
    0.10 * (contract == 'Month-to-Month') +
    0.08 * (products == 1) +
    0.07 * (balance < 5000) +
    0.05 * (is_active == 0) +
    0.05 * (loan == 0) +
    np.random.uniform(0, 0.15, N)
)
churn = (churn_prob > 0.45).astype(int)

df = pd.DataFrame({
    'tenure': tenure, 'age': age, 'balance': balance, 'salary': salary,
    'credit_score': credit_score, 'products': products, 'has_card': has_card,
    'is_active': is_active, 'monthly_charges': monthly_charges, 'complaints': complaints,
    'geography': geography, 'gender': gender, 'contract': contract,
    'loan': loan, 'fd': fd, 'churn': churn
})

df.to_csv('data/bank_churn_data.csv', index=False)
print(f"Dataset: {len(df)} rows | Churn rate: {df['churn'].mean()*100:.1f}%")

le_geo  = LabelEncoder().fit(df['geography'])
le_gen  = LabelEncoder().fit(df['gender'])
le_con  = LabelEncoder().fit(df['contract'])

df['geography_enc'] = le_geo.transform(df['geography'])
df['gender_enc']    = le_gen.transform(df['gender'])
df['contract_enc']  = le_con.transform(df['contract'])

features = ['tenure','age','balance','salary','credit_score','products',
            'has_card','is_active','monthly_charges','complaints',
            'geography_enc','gender_enc','contract_enc','loan','fd']

X = df[features]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
lr  = LogisticRegression(max_iter=1000)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train_s, y_train)

print(f"Random Forest  accuracy: {accuracy_score(y_test, rf.predict(X_test))*100:.2f}%")
print(f"Gradient Boost accuracy: {accuracy_score(y_test, gb.predict(X_test))*100:.2f}%")
print(f"Logistic Reg   accuracy: {accuracy_score(y_test, lr.predict(X_test_s))*100:.2f}%")

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances.head(8))

os.makedirs('models', exist_ok=True)
pickle.dump(rf,      open('models/random_forest.pkl','wb'))
pickle.dump(gb,      open('models/gradient_boost.pkl','wb'))
pickle.dump(lr,      open('models/logistic_reg.pkl','wb'))
pickle.dump(scaler,  open('models/scaler.pkl','wb'))
pickle.dump(le_geo,  open('models/le_geography.pkl','wb'))
pickle.dump(le_gen,  open('models/le_gender.pkl','wb'))
pickle.dump(le_con,  open('models/le_contract.pkl','wb'))
pickle.dump(features,open('models/feature_names.pkl','wb'))

print("\nAll models saved to /models/")

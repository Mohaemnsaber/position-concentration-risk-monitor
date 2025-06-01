```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\User\Downloads\Anaconda\QDL-FON\QDL-FON.csv")

```


```python
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.describe(include='all'))

```


```python
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

```


```python
df = df[df.isnull().mean(axis=1) < 0.5]

```


```python
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

```


```python
df['total_open_interest'] = df[['total_reportable_longs', 'total_reportable_shorts']].max(axis=1)

# Example threshold-based concentration
df['concentration_risk'] = (
    (df['producer_merchant_processor_user_longs'] > 0.4 * df['total_open_interest']) |
    (df['money_manager_longs'] > 0.4 * df['total_open_interest'])
).astype(int)

```


```python
df.drop(['contract_code', 'type', 'date'], axis=1, inplace=True)

```


```python
X = df.drop('concentration_risk', axis=1)
y = df['concentration_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```


```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=5000)
lr.fit(X_train_scaled, y_train)

```


```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

```


```python
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)

```


```python
# Use scaled versions of both X_train and X_test
lr.fit(X_train_scaled, y_train)

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb
}

for name, model in models.items():
    if name == 'Logistic Regression':
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    print(f"Model: {name}")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("=" * 50)

```


```python
print(y_train.value_counts(normalize=True))

```


```python
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False).head(10)

```


```python
import matplotlib.pyplot as plt
import seaborn as sns

top_features = importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

```


```python

```

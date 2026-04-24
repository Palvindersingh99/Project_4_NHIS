# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# DATA LOADING
# ==============================
file_path = r"E:\pgp\INTERSHIP PROJECT\PROJECT 4\mobile_data.csv"
data = pd.read_csv(file_path)

print("\n✅ Data successfully loaded\n")
print(data.head())

# ==============================
# CLEANING FUNCTION
# ==============================
def clean_column(col):
    return pd.to_numeric(col.astype(str).str.replace(r'\D', '', regex=True), errors='coerce')

# ==============================
# DATA CLEANING
# ==============================

data['RAM'] = clean_column(data['RAM'])
data['Memory'] = clean_column(data['Memory'])
data['Battery'] = clean_column(data['Battery'])
data['Rear Camera'] = clean_column(data['Rear Camera'])
data['Front Camera'] = clean_column(data['Front Camera'])
data['Prize'] = clean_column(data['Prize'])

# Drop unwanted column
data = data.drop(columns=['Unnamed: 0'])

# Drop missing values
data = data.dropna()

print("\n✅ Data Cleaned check\n")

# ==============================
# FEATURE EXTRACTION (CORRELATION)
# ==============================
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# HANDLE CATEGORICAL DATA
# ==============================
categorical_cols = ['Model', 'Colour', 'Processor_']

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

print("\n✅ Categorical Data Encoded done\n")

# ==============================
# FEATURES & TARGET
# ==============================
X = data.drop('Prize', axis=1)
y = data['Prize']

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==============================
# MODEL BUILDING
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model Ban Gya")

# ==============================
# MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
accuracy = model.score(X_test, y_test)

print("\n📊 Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("Accuracy (R2):", accuracy)

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = pd.Series(model.coef_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n🔥 Top Features:\n")
print(importance.head(10))

# Plot feature importance
plt.figure(figsize=(10,5))
importance.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()

# ==============================
# VISUALIZATION
# ==============================
plt.figure()
plt.scatter(data['RAM'], data['Prize'])
plt.xlabel("RAM")
plt.ylabel("Price")
plt.title("RAM vs Price")
plt.show()

# ==============================
# SAMPLE PREDICTION
# ==============================
sample = X.iloc[0:1]
prediction = model.predict(sample)

print("\n💰 Sample Predicted Price:", prediction)
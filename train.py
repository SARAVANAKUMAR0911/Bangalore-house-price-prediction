import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")
print("Original Shape:", df.shape)

# Step 1 - Drop unwanted columns
df.drop(['society', 'availability'], axis=1, inplace=True)

# Step 2 - Missing values handle
df.dropna(subset=['location', 'size'], inplace=True)
df['bath'].fillna(df['bath'].median(), inplace=True)
df['balcony'].fillna(df['balcony'].median(), inplace=True)

# Step 3 - BHK extract pannurom (2 BHK → 2)
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if str(x).split(' ')[0].isdigit() else 0)
df.drop('size', axis=1, inplace=True)

# Step 4 - total_sqft clean pannurom (1000-1200 range handle)
def convert_sqft(x):
    try:
        if '-' in str(x):
            vals = x.split('-')
            return (float(vals[0]) + float(vals[1])) / 2
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df.dropna(subset=['total_sqft'], inplace=True)

# Step 5 - Feature Engineering
df['price_per_sqft'] = df['price'] / df['total_sqft']
df['total_rooms'] = df['bhk'] + df['bath']

# Step 6 - Outlier remove
df = df[df['bhk'] <= 10]
df = df[df['bath'] <= 10]
df = df[df['total_sqft'] / df['bhk'] >= 300]

# Step 7 - Location encode (top 50 locations keep pannurom)
location_counts = df['location'].value_counts()
top_locations = location_counts[location_counts > 10].index
df['location'] = df['location'].apply(lambda x: x if x in top_locations else 'Other')

le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])
df['area_type'] = le.fit_transform(df['area_type'])

print("Clean Shape:", df.shape)
print("\nFeatures:", df.columns.tolist())

# Step 8 - Train Test Split
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9 - LightGBM Model
print("\nTraining LightGBM... ⏳")
model = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Step 10 - Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Results ---")
print("R2 Score:", round(r2 * 100, 2), "%")
print("RMSE:", round(rmse, 2), "Lakhs")

# Save model
with open('bangalore_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save label encoder locations
locations = sorted(df['location'].unique().tolist())
with open('locations.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nModel Saved! ✅")
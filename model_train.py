import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- Ensure model folder exists ---
os.makedirs("model", exist_ok=True)

# --- Load dataset ---
# Create a folder named "data" in your project and put your "salaries.csv" file there
DATA_PATH = "data/salaries.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ Dataset not found! Please create data/salaries.csv first.")

df = pd.read_csv(DATA_PATH)

print("✅ Loaded dataset with", df.shape[0], "rows and", df.shape[1], "columns")

# --- Features and Target ---
numeric_features = ["years_experience", "test_score", "interview_score"]
categorical_features = ["job_title", "education_level", "location", "company_size", "industry"]

X = df[numeric_features + categorical_features]
y = df["salary"]

# --- Preprocessing: scale numeric + encode categorical ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --- Build pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# --- Split and train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:")
print(f"   MAE = {mae:.2f}")
print(f"   R²  = {r2:.3f}")


plt.scatter(
    df['years_experience'],
    df['test_score'],
    color='green',
    s=100,
    alpha=0.7
)

plt.title("Years of Experience vs Test Score", fontsize=14)
plt.xlabel("Years of Experience", fontsize=12)
plt.ylabel("Test Score (%)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

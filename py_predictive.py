
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Dataset

df = pd.read_csv(r"C:\Users\Yashw\Downloads\NFLX.csv")

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 2. Data Preprocessing

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")
df = df.drop_duplicates(subset=["Date"])

cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[cols] = df[cols].ffill()
df = df.dropna(subset=cols).reset_index(drop=True)

# 3. Feature Engineering

df["Return"] = df["Close"].pct_change()

df["Close_Lag1"] = df["Close"].shift(1)
df["Close_Lag2"] = df["Close"].shift(2)
df["Close_Lag3"] = df["Close"].shift(3)

df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["Volatility"] = df["Return"].rolling(5).std()

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month

df["Target"] = df["Close"].shift(-1)

features = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "Return", "Close_Lag1", "Close_Lag2", "Close_Lag3",
    "MA_5", "MA_10", "Volatility", "Day", "Month"
]

future_features = df[features].iloc[[-1]].copy()

data = df.dropna(subset=features + ["Target"])

X = data[features]
y = data["Target"]

# 4. Train Test Split

split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 5. Five Models

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "KNN Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ])
}

# 6. Train and Evaluate Models

results = []
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n", name)
    print("MSE:", round(mse, 4))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    results.append([name, mse, mae, rmse, r2])

# 7. Model Comparison

results_df = pd.DataFrame(
    results,
    columns=["Model", "MSE", "MAE", "RMSE", "R2 Score"]
)

results_df = results_df.sort_values("RMSE").reset_index(drop=True)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]

print("\nBest Model:", best_model_name)

# 8. RMSE Graph

plt.figure(figsize=(10, 5))

plt.bar(results_df["Model"], results_df["RMSE"])

plt.xlabel("Models")
plt.ylabel("RMSE")
plt.title("Comparison of Five Models")

plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 9. Actual vs Predicted Graphs

for name, y_pred in predictions.items():

    plt.figure(figsize=(8, 5))

    plt.scatter(y_test, y_pred, alpha=0.6)

    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Perfect Prediction"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(name)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 10. Best Model Graph

plt.figure(figsize=(12, 5))

plt.plot(y_test.values, label="Actual Price")
plt.plot(
    predictions[best_model_name],
    label="Predicted Price",
    linestyle="--"
)

plt.xlabel("Testing Records")
plt.ylabel("Closing Price")
plt.title("Actual vs Predicted Price")

plt.legend()
plt.tight_layout()
plt.show()

# 11. Feature Importance

forest = models["Random Forest"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": forest.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))

plt.figure(figsize=(10, 5))

plt.barh(importance["Feature"], importance["Importance"])

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 12. Next Day Prediction

best_model = models[best_model_name]

best_model.fit(X, y)

next_price = best_model.predict(future_features)[0]

print("\nFinal Prediction")

print("Best Model:", best_model_name)
print("Last Available Date:", df["Date"].iloc[-1].date())
print("Last Closing Price:", round(df["Close"].iloc[-1], 2))
print("Predicted Next Closing Price:", round(next_price, 2))

print("\nAll Five Models Completed Successfully!")

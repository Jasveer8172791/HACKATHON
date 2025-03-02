import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("Aviation KPI Dashboard")

# ✅ Upload CSV instead of relying on a local file
uploaded_file = st.file_uploader("Aviation_KPIs_Cleaned.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write(df.head())  # Show first few rows
else:
    st.warning("⚠ Please upload a CSV file to proceed.")
    st.stop()  # Stop execution until a file is uploaded

# ✅ Define features & target
features = ['Delay (Minutes)', 'Aircraft Utilization (Hours/Day)', 'Turnaround Time (Minutes)',
            'Load Factor (%)', 'Fleet Availability (%)', 'Maintenance Downtime (Hours)',
            'Fuel Efficiency (ASK)', 'Revenue (USD)', 'Operating Cost (USD)',
            'Net Profit Margin (%)', 'Ancillary Revenue (USD)', 'Debt-to-Equity Ratio',
            'Revenue per ASK', 'Cost per ASK']
target = 'Profit (USD)'

# ✅ Check if columns exist in CSV
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    st.error(f"🚨 Missing columns in CSV: {missing_cols}")
    st.stop()

# ✅ Train-Test Split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train Models
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

linear_model.fit(X_train_scaled, y_train)
random_forest_model.fit(X_train_scaled, y_train)

# ✅ Model Evaluation
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = random_forest_model.predict(X_test_scaled)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# ✅ Sidebar Model Performance
st.sidebar.header("📊 Model Performance")
st.sidebar.write("### Linear Regression")
st.sidebar.write(f"MAE: {mae_linear:.2f}")
st.sidebar.write(f"R²: {r2_linear:.2f}")
st.sidebar.write("### Random Forest")
st.sidebar.write(f"MAE: {mae_rf:.2f}")
st.sidebar.write(f"R²: {r2_rf:.2f}")

# ✅ Feature Importance (Random Forest)
st.subheader("Feature Impact on Profit")
feature_importance = pd.Series(random_forest_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots()
feature_importance.plot(kind='barh', ax=ax, color='royalblue')
st.pyplot(fig)

# ✅ Data Visualizations
st.subheader("📊 Data Visualizations")

# Histogram of Profit
fig1 = plt.figure()
sns.histplot(df['Profit (USD)'], bins=30, kde=True, color='blue')
st.pyplot(fig1)

# Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df[features + [target]].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
st.pyplot(fig2)

# Scatter plot: Revenue vs. Profit
fig3 = plt.figure()
sns.scatterplot(x=df['Revenue (USD)'], y=df['Profit (USD)'], alpha=0.5)
st.pyplot(fig3)

# ✅ Boxplot for Delays
fig4 = plt.figure()
sns.boxplot(x=df['Delay (Minutes)'], color='orange')
st.pyplot(fig4)

# ✅ Line plot: Load Factor vs. Profit
fig5 = plt.figure()
sns.lineplot(x=df['Load Factor (%)'], y=df['Profit (USD)'], marker='o')
st.pyplot(fig5)

# ✅ Pairplot (only when dataset is small)
if len(df) < 1000:  # Prevent memory crash on large datasets
    st.subheader("Pairplot of Key Features")
    selected_features = ['Revenue (USD)', 'Operating Cost (USD)', 'Profit (USD)', 'Load Factor (%)']
    sns.pairplot(df[selected_features])
    st.pyplot()

# ✅ User Input for Prediction
st.subheader("🧑‍💻 Predict Profit")
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.number_input(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

# ✅ Convert input to DataFrame & Scale
user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)
predicted_profit_linear = linear_model.predict(user_df_scaled)[0]
predicted_profit_rf = random_forest_model.predict(user_df_scaled)[0]

st.sidebar.subheader("💰 Predicted Profit")
st.sidebar.write(f"### Linear Regression: ${predicted_profit_linear:,.2f}")
st.sidebar.write(f"### Random Forest: ${predicted_profit_rf:,.2f}")

# ✅ Variance Inflation Factor (VIF) to check multicollinearity
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
st.subheader("📉 Variance Inflation Factor (VIF)")
st.write(vif_data)

# ✅ Function to Remove High-VIF Features
def remove_high_vif_features(X, threshold=5.0):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    while vif_data["VIF"].max() > threshold:
        highest_vif_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]["Feature"]
        X = X.drop(columns=[highest_vif_feature])

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return X

# ✅ Remove High-VIF Features
X_filtered = remove_high_vif_features(X)

st.subheader("✅ Selected Features After Removing High VIF")
st.write(X_filtered.columns.tolist())

import streamlit as st   
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("🚀 Employee Attrition Prediction Dashboard")

# --- Load model and scaler ---
model_path = r"C:\\Users\\LAVANYA\\Desktop\\Project3\\employee_attrition_model.pkl"
with open(model_path, "rb") as f:
    model, scaler = pickle.load(f)

# --- Custom Sidebar with Styling ---
st.sidebar.markdown(
    """
    <style>
        .sidebar-title {
            font-size: 24px;
            color: #4CAF50;
            font-weight: bold;
        }
        .sidebar-option {
            font-size: 18px;
            color: #333;
        }
        .sidebar-option:hover {
            color: #4CAF50;
        }
        .sidebar-radio {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-title">🧭 Navigation</div>', unsafe_allow_html=True)

# Navigation options for sidebar
page = st.sidebar.radio(
    "Choose an option 👇",
    options=["🔍 Single Prediction", "📂 Batch Prediction", "🚨 At-Risk Employees"],
    key="navigation"
)

# --- Mapping dictionaries ---
department_map = {"Research & Development": 0, "Sales": 1, "Human Resources": 2, "Marketing": 3}
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
overtime_map = {"Yes": 1, "No": 0}

# --- Main Pages Logic ---
if page.startswith("🔍"):
    st.header("🔎 Single Employee Prediction")

    # Inputs
    age = st.number_input("Age", 18, 65, 30)
    department = st.selectbox("Department", list(department_map.keys()))
    sat = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4], index=2)
    income = st.number_input("Monthly Income", 0, 100000, 5000)
    years = st.number_input("Years at Company", 0, 40, 5)
    marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
    overtime = st.selectbox("Overtime", list(overtime_map.keys()))

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Department": [department_map[department]],
            "JobSatisfaction": [sat],
            "MonthlyIncome": [income],
            "YearsAtCompany": [years],
            "MaritalStatus": [marital_status_map[marital_status]],
            "OverTime": [overtime_map[overtime]]
        })

        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"💼 Predicted: **Leave** (Probability = {prob:.2f})", icon="🚨")
        else:
            st.success(f"💼 Predicted: **Stay** (Leave Probability = {prob:.2f})", icon="✅")
        
        st.write("Attrition Probability:")
        st.progress(prob)

        # Insights Section
        st.subheader("🔍 Insights for this Employee")
        insight_text = []

        if prob > 0.7:
            insight_text.append("⚠️ High risk of attrition. Immediate engagement recommended.")
        elif 0.4 < prob <= 0.7:
            insight_text.append("⚠️ Moderate risk. Monitor employee satisfaction and workload.")
        else:
            insight_text.append("✅ Low risk. Continue existing HR practices.")

        if sat <= 2:
            insight_text.append("📉 Job Satisfaction is low. Consider role enrichment or motivation programs.")

        if overtime == "Yes":
            insight_text.append("⏳ Working overtime. Risk of burnout - suggest reviewing workload.")

        if years < 3:
            insight_text.append("🧩 New employee. Early engagement is crucial for retention.")

        if income < 30000:
            insight_text.append("💵 Income on lower side. Consider if compensation matches role expectations.")

        for point in insight_text:
            st.info(point)

elif page.startswith("📂"):
    st.header("📂 Batch Prediction via CSV")

    file = st.file_uploader("Upload CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        required_columns = ["Age", "Department", "JobSatisfaction", "MonthlyIncome", "YearsAtCompany", "MaritalStatus", "OverTime"]

        if all(col in df.columns for col in required_columns):
            df["Department"] = df["Department"].map(department_map)
            df["MaritalStatus"] = df["MaritalStatus"].map(marital_status_map)
            df["OverTime"] = df["OverTime"].map(overtime_map)

            X = df[required_columns]
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            df["Attrition_Prediction"] = preds
            df["Attrition_Probability"] = probs

            st.dataframe(df)

            st.subheader("Attrition Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Attrition_Probability"], kde=True, ax=ax)
            ax.set_title("Distribution of Attrition Probabilities")
            st.pyplot(fig)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv_data, file_name="predictions.csv", mime="text/csv")
        else:
            st.error(f"CSV must have columns: {', '.join(required_columns)}")

elif page.startswith("🚨"):
    st.header("🚨 Identify At-Risk Employees")

    file = st.file_uploader("Upload Employee Data CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        required_columns = ["Age", "Department", "JobSatisfaction", "MonthlyIncome", "YearsAtCompany", "MaritalStatus", "OverTime"]

        if all(col in df.columns for col in required_columns):
            df["Department"] = df["Department"].map(department_map)
            df["MaritalStatus"] = df["MaritalStatus"].map(marital_status_map)
            df["OverTime"] = df["OverTime"].map(overtime_map)

            X = df[required_columns]
            X_scaled = scaler.transform(X)

            probs = model.predict_proba(X_scaled)[:, 1]
            df["Attrition_Probability"] = probs

            df_sorted = df.sort_values("Attrition_Probability", ascending=False)

            st.subheader("Top 10 At-Risk Employees")
            st.dataframe(df_sorted.head(10))

            st.subheader("Attrition Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df_sorted["Attrition_Probability"], kde=True, ax=ax)
            ax.set_title("At-Risk Employees Probability Distribution")
            st.pyplot(fig)

            csv_data = df_sorted.to_csv(index=False).encode("utf-8")
            st.download_button("Download At-Risk Employees", data=csv_data, file_name="at_risk_employees.csv", mime="text/csv")
        else:
            st.error(f"CSV must have columns: {', '.join(required_columns)}")

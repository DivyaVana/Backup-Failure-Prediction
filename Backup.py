import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Backup Failure Prediction",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================
st.title("🛡️ Backup Failure Prediction Dashboard")
st.markdown(
    """
    This application analyzes backup data and uses an XGBoost machine
    learning model to predict backup failures.
    """
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload Backup Excel File",
    type=["xlsx", "xls"]
)
if uploaded_file is None:
    st.info("👈 Please upload your backup Excel file from the sidebar.")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================
try:
    df = pd.read_excel("Backup.xlsx")
except Exception as e:
    st.error(f"Unable to read the Excel file: {e}")
    st.stop()

# ============================================================
# REQUIRED COLUMNS
# ============================================================
required_columns = [
    "Date",
    "Server_Name",
    "Backup_Type",
    "Data_Size_GB",
    "Duration_Mins",
    "Error_Code",
    "Backup_Status"
]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]
if missing_columns:
    st.error(
        "The following required columns are missing from the Excel file:"
    )
    for col in missing_columns:
        st.write(f"- {col}")
    st.stop()

# ============================================================
# DATA PREPARATION
# ============================================================
data = df.copy()
# Remove duplicate rows
duplicate_count = data.duplicated().sum()
data = data.drop_duplicates().reset_index(drop=True)

# Convert Date
data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)
invalid_dates = data["Date"].isna().sum()
if invalid_dates > 0:
    st.warning(
        f"{invalid_dates} invalid date values were found "
        "and will be removed."
    )
    data = data.dropna(subset=["Date"]).reset_index(drop=True)

# Handle missing values
missing_values = data[required_columns].isnull().sum()
if missing_values.sum() > 0:
    st.warning(
        "Missing values were found. Rows containing missing "
        "required values will be removed."
    )
    data = data.dropna(subset=required_columns).reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["IsWeekend"] = (
    data["DayOfWeek"] >= 5
).astype(int)

# ============================================================
# TARGET CREATION
# ============================================================
data["Failure"] = data["Backup_Status"].map({
    "Success": 0,
    "Failed": 1
})

# Check for unknown target values
if data["Failure"].isna().any():
    unknown_status = (
        data.loc[
            data["Failure"].isna(),
            "Backup_Status"
        ]
        .unique()
        .tolist()
    )
    st.error(
        f"Unknown Backup_Status values found: {unknown_status}"
    )
    st.stop()

# ============================================================
# DATASET INFORMATION
# ============================================================
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total Backups",
        len(data)
    )
with col2:
    success_count = (
        data["Backup_Status"] == "Success"
    ).sum()
    st.metric(
        "Successful",
        success_count
    )
with col3:
    failure_count = (
        data["Backup_Status"] == "Failed"
    ).sum()
    st.metric(
        "Failed",
        failure_count
    )
with col4:
    failure_rate = (
        failure_count / len(data) * 100
        if len(data) > 0
        else 0
    )
    st.metric(
        "Failure Rate",
        f"{failure_rate:.2f}%"
    )

# ============================================================
# DATA QUALITY
# ============================================================
with st.expander("🔍 Data Quality Information"):
    q1, q2, q3 = st.columns(3)
    with q1:
        st.write("**Original Rows**")
        st.write(len(df))
    with q2:
        st.write("**Duplicate Rows Removed**")
        st.write(duplicate_count)
    with q3:
        st.write("**Invalid Dates**")
        st.write(invalid_dates)
    st.write("### Missing Values")
    st.dataframe(
        missing_values.to_frame("Missing Count"),
        use_container_width=True
    )

# ============================================================
# RAW DATA
# ============================================================
with st.expander("📋 View Backup Data"):
    st.dataframe(
        data,
        use_container_width=True,
        height=400
    )

# ============================================================
# BACKUP STATUS VISUALIZATION
# ============================================================
st.header("📈 Backup Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Backup Status")
    status_counts = data["Backup_Status"].value_counts()
    st.bar_chart(status_counts)

with col2:
    st.subheader("Backup Type")
    backup_counts = data["Backup_Type"].value_counts()
    st.bar_chart(backup_counts)

# ============================================================
# SERVER FILTER
# ============================================================
st.subheader("🔎 Server Analysis")
server_options = sorted(
    data["Server_Name"].unique().tolist()
)
selected_servers = st.multiselect(
    "Select Server(s)",
    server_options,
    default=server_options
)
filtered_data = data[
    data["Server_Name"].isin(selected_servers)
].copy()

if len(filtered_data) > 0:
    server_failure = (
        filtered_data
        .groupby("Server_Name")["Failure"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    st.write("### Failure Rate by Server")
    st.bar_chart(server_failure)

# ============================================================
# MODEL FEATURES
# ============================================================
features = [
    "Server_Name",
    "Backup_Type",
    "Data_Size_GB",
    "Duration_Mins",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "IsWeekend"
]
X = data[features].copy()
y = data["Failure"].copy()

# ============================================================
# ENCODING
# ============================================================
server_encoder = LabelEncoder()
backup_encoder = LabelEncoder()
X["Server_Name"] = server_encoder.fit_transform(
    X["Server_Name"].astype(str)
)
X["Backup_Type"] = backup_encoder.fit_transform(
    X["Backup_Type"].astype(str)
)

# ============================================================
# CHECK TARGET
# ============================================================
if y.nunique() < 2:
    st.error(
        "The dataset must contain both Success and Failed "
        "backup records to train the model."
    )
    st.stop()

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# CLASS WEIGHT
# ============================================================
success_count = (
    y_train == 0
).sum()
failure_count = (
    y_train == 1
).sum()
scale_pos_weight = (
    success_count / failure_count
    if failure_count > 0
    else 1
)

# ============================================================
# XGBOOST MODEL
# ============================================================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ============================================================
# TRAIN MODEL
# ============================================================
with st.spinner("Training XGBoost model..."):
    model.fit(
        X_train,
        y_train
    )

# ============================================================
# PREDICTIONS
# ============================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(
    X_test
)[:, 1]

# ============================================================
# MODEL PERFORMANCE
# ============================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)
try:
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )
except ValueError:
    roc_auc = 0

# ============================================================
# MODEL METRICS
# ============================================================
st.header("🤖 Model Performance")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )
with m2:
    st.metric(
        "Precision",
        f"{precision:.2%}"
    )
with m3:
    st.metric(
        "Recall",
        f"{recall:.2%}"
    )
with m4:
    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )
with m5:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.2%}"
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================
st.subheader("Confusion Matrix")
cm = confusion_matrix(
    y_test,
    y_pred
)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Success", "Actual Failed"],
    columns=["Predicted Success", "Predicted Failed"]
)
st.dataframe(
    cm_df,
    use_container_width=True
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
st.subheader("Classification Report")
report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Success",
        "Failed"
    ],
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(
    report
).transpose()
st.dataframe(
    report_df.round(3),
    use_container_width=True
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.header("⭐ Feature Importance")
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
st.bar_chart(
    importance.set_index("Feature")
)

st.dataframe(
    importance,
    use_container_width=True
)

# ============================================================
# PREDICTION RESULTS
# ============================================================
st.header("🚨 Backup Failure Predictions")

# Make results using original test indexes
results = data.loc[
    X_test.index,
    [
        "Date",
        "Server_Name",
        "Backup_Type",
        "Data_Size_GB",
        "Duration_Mins",
        "Error_Code",
        "Backup_Status"
    ]
].copy()

results["Actual_Failure"] = y_test
results["Predicted_Failure"] = y_pred
results["Failure_Probability"] = y_prob
results["Actual_Status"] = results[
    "Actual_Failure"
].map({
    0: "Success",
    1: "Failed"
})
results["Predicted_Status"] = results[
    "Predicted_Failure"
].map({
    0: "Success",
    1: "Failed"
})

results["Failure_Probability"] = (
    results["Failure_Probability"] * 100
).round(2)

# ============================================================
# RISK LEVEL
# ============================================================
def risk_level(probability):
    if probability >= 70:
        return "🔴 High Risk"
    elif probability >= 40:
        return "🟠 Medium Risk"
    elif probability >= 20:
        return "🟡 Low Risk"
    else:
        return "🟢 Very Low Risk"

results["Risk_Level"] = (
    results["Failure_Probability"]
    .apply(risk_level)
)

# ============================================================
# PREDICTION FILTER
# ============================================================
risk_filter = st.selectbox(
    "Filter Predictions",
    [
        "All",
        "High Risk",
        "Medium Risk",
        "Low Risk",
        "Very Low Risk"
    ]
)

display_results = results.copy()

if risk_filter != "All":
    risk_map = {
        "High Risk": "🔴 High Risk",
        "Medium Risk": "🟠 Medium Risk",
        "Low Risk": "🟡 Low Risk",
        "Very Low Risk": "🟢 Very Low Risk"
    }
    display_results = display_results[
        display_results["Risk_Level"]
        == risk_map[risk_filter]
    ]

# Sort by highest failure probability
display_results = display_results.sort_values(
    "Failure_Probability",
    ascending=False
)

st.dataframe(
    display_results,
    use_container_width=True,
    height=500
)

# ============================================================
# HIGH-RISK BACKUPS
# ============================================================
high_risk = results[
    results["Failure_Probability"] >= 70
].copy()

st.header("🔴 High-Risk Backups")
if len(high_risk) > 0:
    st.warning(
        f"{len(high_risk)} backup(s) have a predicted "
        "failure probability of 70% or higher."
    )
    st.dataframe(
        high_risk,
        use_container_width=True
    )
else:
    st.success(
        "No backups currently have a failure probability "
        "of 70% or higher."
    )

# ============================================================
# FAILURE PROBABILITY DISTRIBUTION
# ============================================================
st.header("📊 Failure Probability Distribution")
fig, ax = plt.subplots()
ax.hist(
    results["Failure_Probability"],
    bins=20
)
ax.set_xlabel(
    "Failure Probability (%)"
)
ax.set_ylabel(
    "Number of Backups"
)
ax.set_title(
    "Backup Failure Probability Distribution"
)
st.pyplot(fig)

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================
st.header("🔄 Actual vs Predicted Status")
comparison = pd.crosstab(
    results["Actual_Status"],
    results["Predicted_Status"]
)
st.dataframe(
    comparison,
    use_container_width=True
)

# ============================================================
# DOWNLOAD RESULTS
# ============================================================
st.header("📥 Download Prediction Results")
csv = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=csv,
    file_name="backup_failure_predictions.csv",
    mime="text/csv"
)

# ============================================================
# MODEL INFORMATION
# ============================================================
with st.expander("ℹ️ Model Information"):
    st.write(
        """
        **Algorithm:** XGBoost Classifier
        **Target:** Backup Failure
        **Failure Mapping:**
        - Success = 0
        - Failed = 1
        **Training/Test Split:** 80% / 20%
        **Random State:** 42
        **Number of Estimators:** 300
        **Maximum Depth:** 5
        **Learning Rate:** 0.05
        **Subsample:** 0.8
        **Column Sampling:** 0.8
        The model uses class weighting through
        `scale_pos_weight` because failed backups are
        less frequent than successful backups.
        """
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Backup Failure Prediction | Python + Streamlit + XGBoost"
)

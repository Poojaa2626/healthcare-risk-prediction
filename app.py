
import os

API_KEY = os.environ.get("API_KEY", "").strip()

if not API_KEY:
    raise ValueError(
        "API key is missing. Please add API_KEY "
        "in Render Environment Variables."
    )

print("API key loaded successfully!")
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore")

# -----------------------------
# LOAD DIABETES DATASET
# -----------------------------

diabetes_url = (
    "https://raw.githubusercontent.com/plotly/datasets/"
    "master/diabetes.csv"
)

diabetes_df = pd.read_csv(diabetes_url)

diabetes_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

diabetes_target = "Outcome"

diabetes_df = diabetes_df[diabetes_features + [diabetes_target]].copy()

# -----------------------------
# LOAD HEART DISEASE DATASET
# -----------------------------

heart_urls = [
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv",

    "https://raw.githubusercontent.com/kb22/"
    "Heart-Disease-Prediction/master/dataset.csv"
]

heart_df = None

for url in heart_urls:
    try:
        temp_df = pd.read_csv(url)
        temp_df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in temp_df.columns
        ]

        target_candidates = [
            "target", "num", "condition", "output", "heartdisease"
        ]

        target_col = next(
            (col for col in target_candidates if col in temp_df.columns),
            None
        )

        if target_col is None:
            raise ValueError("Target column not found.")

        heart_df = temp_df.copy()
        heart_df = heart_df.rename(columns={target_col: "target"})

        # Convert the target into binary values:
        # 0 = No disease, greater than 0 = Disease
        heart_df["target"] = pd.to_numeric(
            heart_df["target"], errors="coerce"
        )

        heart_df = heart_df.dropna(subset=["target"])
        heart_df["target"] = (
            heart_df["target"] > 0
        ).astype(int)

        break

    except Exception as e:
        print("Trying another heart dataset source...")

if heart_df is None:
    raise RuntimeError(
        "Unable to download the heart disease dataset. "
        "Check your internet connection and run this cell again."
    )

# Remove unnecessary ID columns
heart_df = heart_df.loc[
    :, ~heart_df.columns.str.contains("^unnamed|^id$")
]

# Keep numeric columns only
heart_df = heart_df.apply(pd.to_numeric, errors="coerce")

# Remove columns containing only missing values
heart_df = heart_df.dropna(axis=1, how="all")

# Ensure target exists
if "target" not in heart_df.columns:
    raise ValueError("Heart dataset target column is missing.")

heart_features = [
    col for col in heart_df.columns if col != "target"
]

heart_df = heart_df.dropna(subset=["target"])

# Remove rows with no usable feature values
heart_df = heart_df.dropna(
    subset=heart_features, how="all"
)

heart_df["target"] = heart_df["target"].astype(int)

print("Diabetes dataset loaded:", diabetes_df.shape)
print("Heart disease dataset loaded:", heart_df.shape)

print("\nDiabetes columns:")
print(diabetes_df.columns.tolist())

print("\nHeart disease columns:")
print(heart_df.columns.tolist())

print("\nDataset loading completed successfully.")
# -----------------------------
# DIABETES ZERO-VALUE HANDLING
# -----------------------------

diabetes_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

def prepare_features(data, zero_columns=None):
    data = data.copy()

    if zero_columns:
        for col in zero_columns:
            if col in data.columns:
                data[col] = data[col].replace(0, np.nan)

    return data


# -----------------------------
# MODEL TRAINING FUNCTION
# -----------------------------

def train_disease_model(
    dataframe,
    feature_columns,
    target_column,
    zero_columns=None
):
    X = dataframe[feature_columns].copy()
    y = dataframe[target_column].astype(int)

    X = prepare_features(X, zero_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probabilities)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": auc
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    best_model_name = results_df.sort_values(
        "ROC-AUC", ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    return {
        "model": best_model,
        "model_name": best_model_name,
        "features": feature_columns,
        "zero_columns": zero_columns or [],
        "results": results_df,
        "X_test": X_test,
        "y_test": y_test
    }


# -----------------------------
# TRAIN DIABETES MODEL
# -----------------------------

diabetes_bundle = train_disease_model(
    diabetes_df,
    diabetes_features,
    diabetes_target,
    diabetes_zero_columns
)

# -----------------------------
# TRAIN HEART DISEASE MODEL
# -----------------------------

heart_bundle = train_disease_model(
    heart_df,
    heart_features,
    "target"
)

print("Diabetes best model:", diabetes_bundle["model_name"])
print("Heart disease best model:", heart_bundle["model_name"])

print("\nDIABETES MODEL COMPARISON")
print(diabetes_bundle["results"].to_string(index=False))

print("\nHEART DISEASE MODEL COMPARISON")
print(heart_bundle["results"].to_string(index=False))

print("\nBoth machine-learning models trained successfully.")
# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_risk(bundle, values):

    input_df = pd.DataFrame(
        [values],
        columns=bundle["features"]
    )

    input_df = prepare_features(
        input_df,
        bundle["zero_columns"]
    )

    probability = bundle["model"].predict_proba(
        input_df
    )[0][1]

    return float(probability)


# -----------------------------
# RISK CLASSIFICATION
# -----------------------------

def get_risk_category(probability):

    percentage = probability * 100

    if percentage < 30:
        return "Low Risk"

    elif percentage < 60:
        return "Moderate Risk"

    else:
        return "High Risk"


# -----------------------------
# LOCAL HEALTH EXPLANATION
# NO EXTERNAL API REQUIRED
# -----------------------------

def generate_health_explanation(disease, probability):

    percentage = probability * 100
    category = get_risk_category(probability)

    if category == "Low Risk":
        recommendation = (
            "Maintain a balanced diet, exercise regularly, "
            "sleep adequately, and attend routine health checkups."
        )

    elif category == "Moderate Risk":
        recommendation = (
            "Monitor relevant health indicators, maintain a "
            "balanced lifestyle, and discuss your results "
            "with a healthcare professional."
        )

    else:
        recommendation = (
            "Consider seeking evaluation from a qualified "
            "healthcare professional. Do not use this prediction "
            "as a substitute for medical diagnosis."
        )

    explanation = f"""
DISEASE: {disease}

ESTIMATED MODEL RISK SCORE: {percentage:.2f}%

RISK CATEGORY: {category}

HEALTH RECOMMENDATION:
{recommendation}

IMPORTANT:
This result is generated by a machine-learning model
for educational and demonstration purposes only.
It is not a medical diagnosis.
"""

    return explanation


# -----------------------------
# RISK GAUGE
# -----------------------------

def create_gauge(probability):

    percentage = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            title={"text": "Estimated Risk Score"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#fee2e2"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": percentage
                }
            }
        )
    )

    fig.update_layout(height=300)

    return fig


# -----------------------------
# DIABETES PREDICTION
# -----------------------------

def diabetes_prediction(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    pedigree,
    age
):

    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age
    ]

    if any(value is None for value in values):
        raise gr.Error("Please enter all diabetes input values.")

    probability = predict_risk(
        diabetes_bundle,
        values
    )

    category = get_risk_category(probability)

    explanation = generate_health_explanation(
        "Diabetes",
        probability
    )

    result = (
        f"Diabetes Risk Category: {category}\n"
        f"Estimated Risk Score: {probability * 100:.2f}%\n"
        f"Selected Model: {diabetes_bundle['model_name']}"
    )

    return result, explanation, create_gauge(probability)


# -----------------------------
# HEART DISEASE PREDICTION
# -----------------------------

def heart_prediction(*values):

    if any(value is None for value in values):
        raise gr.Error("Please enter all heart disease input values.")

    probability = predict_risk(
        heart_bundle,
        values
    )

    category = get_risk_category(probability)

    explanation = generate_health_explanation(
        "Heart Disease",
        probability
    )

    result = (
        f"Heart Disease Risk Category: {category}\n"
        f"Estimated Risk Score: {probability * 100:.2f}%\n"
        f"Selected Model: {heart_bundle['model_name']}"
    )

    return result, explanation, create_gauge(probability)


print("Prediction and explanation functions are ready.")
# -----------------------------
# HELPER: DATA EXPLORER
# -----------------------------

def explore_data(dataset_name):

    if dataset_name == "Diabetes":
        data = diabetes_df
        bundle = diabetes_bundle

    else:
        data = heart_df
        bundle = heart_bundle

    description = data.describe().round(2)

    return (
        data.head(20),
        description,
        bundle["results"].round(3)
    )


# -----------------------------
# HELPER: PATIENT MONITORING
# -----------------------------

def patient_monitoring(
    temperature,
    heart_rate,
    glucose,
    spo2
):

    readings = [temperature, heart_rate, glucose, spo2]

    if any(value is None for value in readings):
        raise gr.Error("Please enter all monitoring values.")

    alerts = []

    if temperature >= 38:
        alerts.append("High temperature detected.")

    if heart_rate > 100 or heart_rate < 60:
        alerts.append("Heart rate is outside the illustrative range.")

    if glucose >= 200:
        alerts.append("High glucose reading. Seek professional advice.")

    if spo2 < 95:
        alerts.append("Oxygen saturation is below the illustrative threshold.")

    if not alerts:
        status = "No threshold alerts from the entered readings."
    else:
        status = "Monitoring alerts:\n" + "\n".join(
            "- " + alert for alert in alerts
        )

    monitoring_data = pd.DataFrame({
        "Parameter": [
            "Temperature (°C)",
            "Heart Rate (bpm)",
            "Glucose (mg/dL)",
            "SpO2 (%)"
        ],
        "Reading": readings
    })

    fig = px.bar(
        monitoring_data,
        x="Parameter",
        y="Reading",
        title="Patient Monitoring Readings"
    )

    fig.update_layout(height=350)

    return status, monitoring_data, fig


# -----------------------------
# HELPER: PDF REPORT
# -----------------------------

def generate_pdf_report(
    patient_name,
    disease,
    risk_score,
    risk_category
):

    if not patient_name or not patient_name.strip():
        raise gr.Error("Please enter a patient name.")

    if risk_score is None or not 0 <= risk_score <= 100:
        raise gr.Error("Enter a risk score between 0 and 100.")

    if disease == "Diabetes":
        model_name = diabetes_bundle["model_name"]
    else:
        model_name = heart_bundle["model_name"]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )
    temp_file.close()

    pdf = canvas.Canvas(temp_file.name, pagesize=A4)

    width, height = A4

    y = height - 70

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, y, "Healthcare Risk Prediction Report")

    y -= 50

    pdf.setFont("Helvetica", 12)

    report_lines = [
        f"Patient Name: {patient_name}",
        f"Disease Type: {disease}",
        f"Estimated Risk Score: {risk_score:.2f}%",
        f"Risk Category: {risk_category}",
        f"Machine Learning Model: {model_name}",
        "",
        "Recommendation:",
        "Please consult a qualified healthcare professional",
        "for interpretation and further evaluation.",
        "",
        "Disclaimer:",
        "This report is generated by an educational machine",
        "learning application. It is not a medical diagnosis."
    ]

    for line in report_lines:
        pdf.drawString(50, y, line)
        y -= 25

    pdf.save()

    return temp_file.name


# -----------------------------
# CREATE GRADIO WEBSITE
# -----------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AI-Powered Healthcare Risk Prediction"
) as demo:

    gr.Markdown(
        """
        # 🏥 AI-Powered Healthcare Risk Prediction

        ### Diabetes | Heart Disease | Health Explanation | Patient Monitoring

        An educational machine-learning healthcare application.
        """
    )

    # -------------------------
    # DIABETES TAB
    # -------------------------

    with gr.Tab("Diabetes Prediction"):

        gr.Markdown("## Diabetes Risk Assessment")

        with gr.Row():

            with gr.Column():

                d_pregnancies = gr.Number(
                    label="Pregnancies",
                    value=1
                )

                d_glucose = gr.Number(
                    label="Glucose",
                    value=120
                )

                d_bp = gr.Number(
                    label="Blood Pressure",
                    value=70
                )

                d_skin = gr.Number(
                    label="Skin Thickness",
                    value=20
                )

            with gr.Column():

                d_insulin = gr.Number(
                    label="Insulin",
                    value=80
                )

                d_bmi = gr.Number(
                    label="BMI",
                    value=25
                )

                d_pedigree = gr.Number(
                    label="Diabetes Pedigree Function",
                    value=0.5
                )

                d_age = gr.Number(
                    label="Age",
                    value=30
                )

        diabetes_button = gr.Button(
            "Predict Diabetes Risk",
            variant="primary"
        )

        diabetes_result = gr.Textbox(
            label="Prediction Result",
            lines=3
        )

        diabetes_explanation = gr.Textbox(
            label="Health Risk Explanation",
            lines=8
        )

        diabetes_gauge = gr.Plot(
            label="Diabetes Risk Visualization"
        )

        diabetes_button.click(
            fn=diabetes_prediction,
            inputs=[
                d_pregnancies,
                d_glucose,
                d_bp,
                d_skin,
                d_insulin,
                d_bmi,
                d_pedigree,
                d_age
            ],
            outputs=[
                diabetes_result,
                diabetes_explanation,
                diabetes_gauge
            ]
        )

    # -------------------------
    # HEART DISEASE TAB
    # -------------------------

    with gr.Tab("Heart Disease Prediction"):

        gr.Markdown(
            "## Heart Disease Risk Assessment"
        )

        gr.Markdown(
            "Enter values corresponding to the heart dataset features."
        )

        heart_inputs = []

        with gr.Row():

            with gr.Column():

                for feature in heart_features[:7]:

                    component = gr.Number(
                        label=feature.replace("_", " ").title()
                    )

                    heart_inputs.append(component)

            with gr.Column():

                for feature in heart_features[7:]:

                    component = gr.Number(
                        label=feature.replace("_", " ").title()
                    )

                    heart_inputs.append(component)

        heart_button = gr.Button(
            "Predict Heart Disease Risk",
            variant="primary"
        )

        heart_result = gr.Textbox(
            label="Prediction Result",
            lines=3
        )

        heart_explanation = gr.Textbox(
            label="Health Risk Explanation",
            lines=8
        )

        heart_gauge = gr.Plot(
            label="Heart Disease Risk Visualization"
        )

        heart_button.click(
            fn=heart_prediction,
            inputs=heart_inputs,
            outputs=[
                heart_result,
                heart_explanation,
                heart_gauge
            ]
        )

    # -------------------------
    # DATA EXPLORER TAB
    # -------------------------

    with gr.Tab("Data Explorer"):

        gr.Markdown("## Dataset Analysis")

        dataset_choice = gr.Radio(
            choices=["Diabetes", "Heart Disease"],
            value="Diabetes",
            label="Select Dataset"
        )

        explore_button = gr.Button(
            "Explore Dataset",
            variant="primary"
        )

        dataset_preview = gr.Dataframe(
            label="Dataset Preview"
        )

        dataset_description = gr.Dataframe(
            label="Statistical Description"
        )

        model_comparison = gr.Dataframe(
            label="Machine Learning Model Comparison"
        )

        explore_button.click(
            fn=explore_data,
            inputs=dataset_choice,
            outputs=[
                dataset_preview,
                dataset_description,
                model_comparison
            ]
        )

    # -------------------------
    # PATIENT MONITORING TAB
    # -------------------------

    with gr.Tab("Patient Monitoring"):

        gr.Markdown(
            """
            ## Patient Health Monitoring

            Enter sample readings to check for illustrative
            threshold alerts. This is not a live medical device.
            """
        )

        with gr.Row():

            temperature_input = gr.Number(
                label="Temperature (°C)",
                value=36.8
            )

            heart_rate_input = gr.Number(
                label="Heart Rate (bpm)",
                value=75
            )

        with gr.Row():

            glucose_input = gr.Number(
                label="Glucose (mg/dL)",
                value=110
            )

            spo2_input = gr.Number(
                label="SpO2 (%)",
                value=98
            )

        monitor_button = gr.Button(
            "Monitor Patient",
            variant="primary"
        )

        monitoring_status = gr.Textbox(
            label="Monitoring Status",
            lines=5
        )

        monitoring_table = gr.Dataframe(
            label="Patient Readings"
        )

        monitoring_chart = gr.Plot(
            label="Monitoring Visualization"
        )

        monitor_button.click(
            fn=patient_monitoring,
            inputs=[
                temperature_input,
                heart_rate_input,
                glucose_input,
                spo2_input
            ],
            outputs=[
                monitoring_status,
                monitoring_table,
                monitoring_chart
            ]
        )

    # -------------------------
    # PDF REPORT TAB
    # -------------------------

    with gr.Tab("PDF Report"):

        gr.Markdown(
            "## Generate Healthcare Risk Report"
        )

        patient_name = gr.Textbox(
            label="Patient Name",
            placeholder="Enter patient name"
        )

        report_disease = gr.Dropdown(
            choices=["Diabetes", "Heart Disease"],
            value="Diabetes",
            label="Disease Type"
        )

        report_risk_score = gr.Number(
            label="Risk Score (%)",
            value=25
        )

        report_category = gr.Dropdown(
            choices=[
                "Low Risk",
                "Moderate Risk",
                "High Risk"
            ],
            value="Low Risk",
            label="Risk Category"
        )

        pdf_button = gr.Button(
            "Generate PDF Report",
            variant="primary"
        )

        pdf_output = gr.File(
            label="Download Healthcare Report"
        )

        pdf_button.click(
            fn=generate_pdf_report,
            inputs=[
                patient_name,
                report_disease,
                report_risk_score,
                report_category
            ],
            outputs=pdf_output
        )

    gr.Markdown(
        """
        ---
        **Disclaimer:** This application is an academic
        machine-learning prototype. Predictions and monitoring
        alerts are not a substitute for professional medical advice.
        """
    )

print("Website interface created successfully.")
import os

demo.queue()

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=False
)

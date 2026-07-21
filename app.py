
import joblib
import gradio as gr
import numpy as np

model = joblib.load("loan_prediction_model.pkl")

def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):
    try:
        features = [
            int(no_of_dependents),
            1 if education == "Graduate" else 0,
            1 if self_employed == "Yes" else 0,
            float(income_annum),
            float(loan_amount),
            int(loan_term),
            int(cibil_score),
            float(residential_assets_value),
            float(commercial_assets_value),
            float(luxury_assets_value),
            float(bank_asset_value),
        ]
    except Exception:
        return "❌ Please enter valid numeric values."

    if any(x < 0 for x in [
        features[0],features[3],features[4],features[5],
        features[6],features[7],features[8],features[9],features[10]
    ]):
        return "❌ Negative values are not allowed."

    if not 300 <= features[6] <= 900:
        return "❌ CIBIL score must be between 300 and 900."

    pred = model.predict(np.array(features).reshape(1,-1))[0]

    if pred == 1:
        return (
            "## ✅ Loan Approved\n\n"
            "Congratulations! Based on the provided details, the model predicts "
            "that the loan is likely to be approved."
        )
    else:
        return (
            "## ❌ Loan Rejected\n\n"
            "Based on the provided details, the model predicts that the loan is "
            "unlikely to be approved."
        )

with gr.Blocks(title="Loan Approval Prediction System") as demo:
    gr.Markdown("""
# 🏦 Loan Approval Prediction System

Predict whether a loan application is likely to be **Approved** or **Rejected**
using a trained **Random Forest Classifier**.
""")

    with gr.Row():
        dep = gr.Number(label="Number of Dependents", value=0)
        edu = gr.Radio(
            choices=["Graduate", "Not Graduate"],
            label="🎓 Education",
            value="Graduate"
        )

        emp = gr.Radio(
            choices=["Yes", "No"],
            label="💼 Self Employed",
            value="No"
        )

    income = gr.Slider(
    minimum=0,
    maximum=50000000,
    step=10000,
    value=500000,
    label="Annual Income")
    loan = gr.Slider(
    minimum=0,
    maximum=60000000,
    step=10000,
    value=1000000,
    label="Loan Amount")
    loan_term = gr.Slider(
    minimum=2,
    maximum=20,
    step=1,
    value=10,
    label="Loan Term (Years)")
    cibil = gr.Slider(
    minimum=300,
    maximum=900,
    step=1,
    value=650,
    label="CIBIL Score")
    residential = gr.Slider(
    minimum=0,
    maximum=29100000,
    step=10000,
    value=0,
    label="🏠 Residential Assets Value"
)

    commercial = gr.Slider(
        minimum=0,
        maximum=19400000,
        step=10000,
        value=0,
        label="🏢 Commercial Assets Value"
    )

    luxury = gr.Slider(
        minimum=300000,
        maximum=39200000,
        step=10000,
        value=300000,
        label="💎 Luxury Assets Value"
    )

    bank = gr.Slider(
        minimum=0,
        maximum=14700000,
        step=10000,
        value=0,
        label="🏦 Bank Asset Value")
    output = gr.Markdown()

    btn = gr.Button("Predict Loan Status", variant="primary")

    btn.click(
        predict_loan,
        inputs=[
    dep,
    edu,
    emp,
    income,
    loan,
    loan_term,
    cibil,
    residential,
    commercial,
    luxury,
    bank
],
        outputs=output
    )

    gr.Markdown("""
---
## 👩‍💻 Developer

**Sneha Verma**

**Roll No.:** 28240091

**Branch:** B.Tech CSE Core [B]

🔗 **LinkedIn:** [Visit My LinkedIn](https://linkedin.com/in/sneha-verma-071b04322/)

💻 **GitHub:** [Visit My GitHub](https://github.com/SnehaVerma16)


---
### Tech Stack
- Python
- Gradio
- Scikit-learn
- Random Forest Classifier
- NumPy
- Pandas
- Joblib

> **Disclaimer:** This application predicts loan approval using a trained Machine Learning model. Predictions are for educational purposes only and should not be considered as financial advice or an actual bank decision.
""")

import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

# app.py
from flask import Flask, request, jsonify, render_template_string
import joblib, pandas as pd, numpy as np

# -------- Load artifacts (pipeline + tuned threshold) --------
ART_PATH = r"C:\Vodaphone_Customer_Churn\churn_pipeline_artifacts.joblib"
art = joblib.load(ART_PATH)
pipe = art["model"]              # full fitted Pipeline (preprocess + model)
thr  = float(art["threshold"])   # tuned decision threshold

# Get the raw feature columns that the ColumnTransformer expects
pre = pipe.named_steps["preprocess"]          # ColumnTransformer
num_cols = list(pre.transformers_[0][2]) if len(pre.transformers_) > 0 else []
cat_cols = list(pre.transformers_[1][2]) if len(pre.transformers_) > 1 else []
expected_cols = num_cols + cat_cols          # order matters

app = Flask(__name__)

# ---------- Home ----------
@app.route("/", methods=["GET"])
def home():
    return "<h3>Vodafone Churn API</h3><p>Use <a href='/form'>/form</a> or POST JSON to <code>/predict</code>.</p>"

# ---------- JSON API ----------
@app.route("/predict", methods=["POST"])
def predict():
    # Accept JSON OR form submit
    if request.is_json:
        payload = request.get_json()
        X = pd.DataFrame([payload]) if isinstance(payload, dict) else pd.DataFrame(payload)
    else:
        payload = request.form.to_dict()
        X = pd.DataFrame([payload])

    # Coerce numeric fields (safe if columns are missing)
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                "numAdminTickets", "numTechTickets"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # === IMPORTANT: align to training schema ===
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan          # add missing raw columns as NaN
    X = X[expected_cols]             # keep the same order as training

    # Predict
    proba = pipe.predict_proba(X)[:, 1]
    pred  = (proba >= thr).astype(int)

    # If this was a browser form, show a friendly message
    if not request.is_json:
        msg = "Churn" if pred[0] == 1 else "No Churn"
        return f"<h3>Prediction: {msg}</h3><p>Probability: {proba[0]:.3f} &nbsp;&nbsp; Threshold: {thr}</p><p><a href='/form'>Back to form</a></p>"

    # Otherwise return JSON
    return jsonify({
        "prediction": pred.tolist(),
        "probability": [float(p) for p in proba],
        "threshold": thr
    })

# ---------- Simple HTML form with dropdowns ----------
FORM_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Vodafone Churn Form</title>
  <style>
    body { font-family: system-ui, Arial; max-width: 860px; margin: 2rem auto; }
    fieldset { margin-bottom: 1rem; }
    label { display:block; margin:.4rem 0 .2rem; }
    input, select { width: 100%; padding:.45rem; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .btn { margin-top:1rem; padding:.6rem 1rem; }
    small{color:#666}
  </style>
</head>
<body>
  <h2>Vodafone Churn — Quick Prediction</h2>
  <form method="post" action="/predict">
    <fieldset class="row">
      <div><label>tenure</label><input name="tenure" type="number" step="1" value="12"></div>
      <div><label>MonthlyCharges</label><input name="MonthlyCharges" type="number" step="0.01" value="75"></div>
      <div><label>TotalCharges</label><input name="TotalCharges" type="number" step="0.01" value="900"></div>
      <div><label>SeniorCitizen (0/1)</label>
        <select name="SeniorCitizen"><option value="0">0</option><option value="1">1</option></select>
      </div>
    </fieldset>

    <fieldset class="row">
      <div><label>Contract</label>
        <select name="Contract">
          <option>Month-to-month</option>
          <option>One year</option>
          <option>Two year</option>
        </select>
      </div>
      <div><label>InternetService</label>
        <select name="InternetService">
          <option>Fiber optic</option>
          <option>DSL</option>
          <option>No</option>
        </select>
      </div>
      <div><label>PaymentMethod</label>
        <select name="PaymentMethod">
          <option>Electronic check</option>
          <option>Mailed check</option>
          <option>Bank transfer (automatic)</option>
          <option>Credit card (automatic)</option>
        </select>
      </div>
      <div><label>PaperlessBilling</label>
        <select name="PaperlessBilling"><option>Yes</option><option>No</option></select>
      </div>
    </fieldset>

    <fieldset class="row">
      <div><label>TechSupport</label><select name="TechSupport"><option>No</option><option>Yes</option></select></div>
      <div><label>OnlineSecurity</label><select name="OnlineSecurity"><option>No</option><option>Yes</option></select></div>
      <div><label>OnlineBackup</label><select name="OnlineBackup"><option>No</option><option>Yes</option></select></div>
      <div><label>DeviceProtection</label><select name="DeviceProtection"><option>No</option><option>Yes</option></select></div>
      <div><label>StreamingTV</label><select name="StreamingTV"><option>No</option><option>Yes</option></select></div>
      <div><label>StreamingMovies</label><select name="StreamingMovies"><option>No</option><option>Yes</option></select></div>
    </fieldset>

    <fieldset class="row">
      <div><label>PhoneService</label><select name="PhoneService"><option>Yes</option><option>No</option></select></div>
      <div><label>MultipleLines</label><select name="MultipleLines"><option>No</option><option>Yes</option></select></div>
      <div><label>Dependents</label><select name="Dependents"><option>No</option><option>Yes</option></select></div>
      <div><label>gender</label><select name="gender"><option>Male</option><option>Female</option></select></div>
      <div><label>Location</label><input name="Location" value="East - Illinois"></div>
      <div><label>numAdminTickets</label><input name="numAdminTickets" type="number" step="1" value="0"></div>
      <div><label>numTechTickets</label><input name="numTechTickets" type="number" step="1" value="2"></div>
    </fieldset>

    <button class="btn" type="submit">Predict</button>
    <p><small>You can leave fields at defaults; missing ones are handled automatically.</small></p>
  </form>
</body>
</html>
"""

@app.route("/form", methods=["GET"])
def form_page():
    return render_template_string(FORM_HTML)

if __name__ == "__main__":
    app.run(debug=True)

# ============================================================
# EL NIÑO PREDICTOR — FLASK WEB APP
# Routes:
#   /          → home page with prediction form
#   /predict   → handles form POST, returns result
#   /history   → shows all past predictions from SQLite
# ============================================================

import os, json, sqlite3
from datetime import datetime
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# ── Load model artifacts once at startup ──────────────────────
MODEL    = joblib.load('models/model.pkl')
SCALER   = joblib.load('models/scaler.pkl')
FEATURES = joblib.load('models/features.pkl')

with open('models/metadata.json') as f:
    META = json.load(f)

# ── SQLite setup ──────────────────────────────────────────────
DB_PATH = 'predictions.db'

def init_db():
    """Create predictions table if it doesn't exist."""
    con = sqlite3.connect(DB_PATH)
    con.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            oni         REAL,
            mei         REAL,
            month       INTEGER,
            probability REAL,
            prediction  TEXT,
            risk_level  TEXT
        )
    ''')
    con.commit()
    con.close()

def save_prediction(oni, mei, month, prob, pred, risk):
    """Save one prediction row to SQLite."""
    con = sqlite3.connect(DB_PATH)
    con.execute('''
        INSERT INTO predictions
        (timestamp, oni, mei, month, probability, prediction, risk_level)
        VALUES (?,?,?,?,?,?,?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          oni, mei, month, round(prob, 4), pred, risk))
    con.commit()
    con.close()

def build_features(oni, mei, month):
    """
    Build the full 22-feature vector from 3 user inputs.
    We use the oni value for all lag/rolling features as a
    reasonable approximation when historical context is unknown.
    For a production app you'd fetch real historical ONI series.
    """
    oni_approx_prev = oni * 0.85   # rough estimate of prior months

    feat = {
        'oni'              : oni,
        'mei_lag1'         : mei,
        'mei_lag3'         : mei * 0.7,
        'oni_lag1'         : oni_approx_prev,
        'oni_lag2'         : oni_approx_prev * 0.95,
        'oni_lag3'         : oni_approx_prev * 0.90,
        'oni_lag6'         : oni_approx_prev * 0.75,
        'oni_lag12'        : oni_approx_prev * 0.50,
        'oni_roll_mean_3'  : oni_approx_prev,
        'oni_roll_mean_6'  : oni_approx_prev * 0.90,
        'oni_roll_mean_12' : oni_approx_prev * 0.75,
        'oni_roll_std_3'   : abs(oni - oni_approx_prev) * 0.5,
        'oni_roll_std_6'   : abs(oni - oni_approx_prev) * 0.7,
        'oni_roll_max_3'   : max(oni, oni_approx_prev),
        'oni_roll_max_6'   : max(oni, oni_approx_prev * 1.1),
        'oni_trend_6m'     : (oni - oni_approx_prev) / 6,
        'oni_diff1'        : oni - oni_approx_prev,
        'oni_diff3'        : oni - oni_approx_prev * 0.90,
        'consec_above'     : max(0, int(oni / 0.5)) if oni >= 0.5 else 0,
        'month_sin'        : np.sin(2 * np.pi * month / 12),
        'month_cos'        : np.cos(2 * np.pi * month / 12),
    }
    # Return as ordered array matching training feature order
    return np.array([[feat[f] for f in FEATURES]])

# ── Month names for display ───────────────────────────────────
MONTH_NAMES = ['','January','February','March','April','May','June',
               'July','August','September','October','November','December']

# ============================================================
# ROUTE 1: Home — prediction form
# ============================================================
@app.route('/')
def index():
    return render_template('index.html', meta=META, month_names=MONTH_NAMES[1:])

# ============================================================
# ROUTE 2: Predict — handle form POST
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        oni   = float(request.form['oni'])
        mei   = float(request.form['mei'])
        month = int(request.form['month'])

        # Build features and scale
        X     = build_features(oni, mei, month)
        X_s   = SCALER.transform(X)
        prob  = MODEL.predict_proba(X_s)[0][1]
        pred  = 'El Niño' if prob >= 0.5 else 'Not El Niño'
        risk  = 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.4 else 'LOW'

        # ENSO phase label
        if oni >= 0.5:
            phase = 'El Niño'
        elif oni <= -0.5:
            phase = 'La Niña'
        else:
            phase = 'Neutral'

        # Save to DB
        save_prediction(oni, mei, month, prob, pred, risk)

        result = {
            'oni'        : oni,
            'mei'        : mei,
            'month_name' : MONTH_NAMES[month],
            'probability': round(prob * 100, 1),
            'prediction' : pred,
            'risk'       : risk,
            'phase'      : phase,
            'is_elnino'  : pred == 'El Niño',
        }
        return render_template('result.html', r=result, meta=META)

    except Exception as e:
        return f"<h3>Error: {e}</h3><a href='/'>← Back</a>", 400

# ============================================================
# ROUTE 3: History — show all predictions from SQLite
# ============================================================
@app.route('/history')
def history():
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT 50'
    ).fetchall()
    con.close()
    cols = ['id','timestamp','oni','mei','month',
            'probability','prediction','risk_level']
    records = [dict(zip(cols, row)) for row in rows]
    # Add month name
    for rec in records:
        rec['month_name'] = MONTH_NAMES[rec['month']]
        rec['probability_pct'] = round(rec['probability'] * 100, 1)
    return render_template('history.html', records=records, meta=META)

# ============================================================
# ROUTE 4: Clear history (bonus utility route)
# ============================================================
@app.route('/clear', methods=['POST'])
def clear():
    con = sqlite3.connect(DB_PATH)
    con.execute('DELETE FROM predictions')
    con.commit()
    con.close()
    return redirect(url_for('history'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

# 🌊 El Niño Predictor

A machine learning web app that predicts El Niño occurrence using 70+ years of NOAA climate data.

**[🚀 Live Demo →](http://YOUR-EC2-IP)**

---

## 📊 Model Performance

| Model | ROC-AUC | Avg Precision | CV AUC |
|-------|---------|---------------|--------|
| Logistic Regression | **1.0000** | 0.9993 | — |
| SVM (RBF kernel) | 0.9997 | 0.9993 | 0.9994 |
| **Best (LR)** | **1.0000** | — | — |

> ROC-AUC of 1.00 on test period (2010–present) — model perfectly separates El Niño from non-El Niño months.

---

## 🛠 Tech Stack

`Python` `scikit-learn` `Flask` `SQLite3` `Gunicorn` `Nginx` `AWS EC2`

---

## 🔍 Project Highlights

- **Two data sources**: NOAA ONI index (1950–present) + MEI v2 (1979–present)  
- **22 engineered features**: lag values (t-1 to t-12), rolling mean/std/max, trend slope, cyclical month encoding, consecutive months above threshold  
- **Class imbalance handled**: `class_weight='balanced'` + ROC-AUC evaluation (28% El Niño months)  
- **Time-based train/test split**: train 1950–2009, test 2010–present (no data leakage)  
- **Flask REST app**: 3 routes — prediction form, results page, SQLite prediction log  
- **Deployed**: AWS EC2 t2.micro (Free Tier) with Gunicorn + Nginx  

---

## 📁 Structure

```
el-nino-predictor/
├── day1_train.py          # Training pipeline (EDA → models)
├── app.py                 # Flask web app
├── requirements.txt
├── data/
│   ├── oni_historical_data.csv
│   ├── mei_v2_historical_data.csv
│   └── processed.csv
├── models/
│   ├── model.pkl          # Trained Logistic Regression
│   ├── scaler.pkl
│   ├── features.pkl
│   └── metadata.json
├── templates/
│   ├── index.html         # Prediction form
│   ├── result.html        # Output + risk gauge
│   └── history.html       # SQLite prediction log
└── deploy/
    ├── setup.sh           # EC2 install script
    └── nginx.conf         # Reverse proxy config
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/el-nino-predictor
cd el-nino-predictor
pip install -r requirements.txt

# Train the model first
python day1_train.py

# Launch Flask app
python app.py
# Open http://localhost:5000
```

---

## ☁️ Deploy on AWS Free Tier

```bash
# 1. Launch EC2 t2.micro (Ubuntu 22.04) — Free Tier
# 2. Open ports 22 (SSH) and 80 (HTTP) in Security Group
# 3. SSH into instance:
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 4. Run setup script:
bash deploy/setup.sh

# 5. Configure nginx:
sudo cp deploy/nginx.conf /etc/nginx/sites-available/elnino
sudo ln -s /etc/nginx/sites-available/elnino /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

---

*Built to demonstrate end-to-end ML: data wrangling → feature engineering → classification → Flask deployment.*

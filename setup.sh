#!/bin/bash
# ── Run this on your EC2 t2.micro after SSH-ing in ──
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx -y

# Clone your repo
git clone https://github.com/YOUR_USERNAME/el-nino-predictor.git
cd el-nino-predictor

# Setup venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# Init DB
python3 -c "from app import init_db; init_db()"

# Run with gunicorn (4 workers, port 8000)
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app --daemon

echo "Flask app running on port 8000"
echo "Now configure nginx to proxy port 80 → 8000"

# 🏠 Bangalore House Price Predictor

A machine learning web app to predict house prices in Bangalore using LightGBM.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green) ![LightGBM](https://img.shields.io/badge/LightGBM-95%25_Accuracy-orange)

---

## 📊 Model Performance

| Model | R² Score |
|-------|----------|
| Linear Regression | 57.58% |
| Random Forest | 80.20% |
| XGBoost | 84.09% |
| **LightGBM (Final)** | **95.06%** ✅ |

---

## 🚀 Features

- 🔮 Predict house prices in 50+ Bangalore locations
- 🗺️ View best places & shopping malls in Bangalore
- 🎓 Explore top schools & colleges
- 📰 Real estate news & market updates
- 📩 Contact form

---

## 🛠️ Tech Stack

- **ML Model** - LightGBM
- **Backend** - FastAPI
- **Frontend** - HTML, CSS, JavaScript
- **Dataset** - Bengaluru Housing Dataset (Kaggle)

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/SARAVANAKUMAR0911/Bangalore-house-price-prediction.git
cd Bangalore-house-price-prediction

# Install dependencies
pip install fastapi uvicorn lightgbm scikit-learn pandas numpy

# Run the app
uvicorn app:app --reload
```

Open browser → `http://127.0.0.1:8000`

---

## 📁 Project Structure

```
├── app.py               # FastAPI backend
├── train.py             # Model training code
├── bangalore_model.pkl  # Trained LightGBM model
├── locations.pkl        # Label encoder
├── templates/
│   └── index.html       # Frontend UI
└── Bengaluru_House_Data.csv
```

---

## 👨‍💻 Author

**Saravana Kumar** - [@SARAVANAKUMAR0911](https://github.com/SARAVANAKUMAR0911)

---

⭐ Star this repo if you found it useful!

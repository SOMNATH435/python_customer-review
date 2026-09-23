# ⭐ Customer Reviews Analytics

A full-stack ML application for analysing customer reviews — sentiment classification,
KPI dashboards, trend analysis, live text prediction, and filterable review browsing.

---

## Tech Stack

| Layer     | Technology                                      |
|-----------|-------------------------------------------------|
| Data      | `Customer_Reviews.csv` (20,000 reviews, 11 cols)|
| ML Model  | TF-IDF + Logistic Regression (scikit-learn)     |
| Backend   | Flask REST API (Python)                         |
| Frontend  | Streamlit (Python)                              |
| Charts    | Plotly                                          |

---

## Project Structure

```
customer_reviews/
├── data/
│   └── Customer_Reviews.csv      ← source dataset
├── model/
│   ├── train_model.py            ← train & evaluate sentiment classifier
│   ├── sentiment_model.pkl       ← saved model pipeline (auto-generated)
│   ├── metadata.json             ← metrics + top terms (auto-generated)
│   ├── confusion_matrix.png      ← test set confusion matrix (auto-generated)
│   └── top_terms.png             ← per-class discriminative terms (auto-generated)
├── backend/
│   └── app.py                    ← Flask REST API (port 5050)
├── frontend/
│   └── streamlit_app.py          ← Streamlit UI (port 8501)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create virtual environment & install dependencies

```bash
cd customer_reviews

# Using uv (recommended)
uv venv .venv --python 3.13
uv pip install -r requirements.txt --python .venv/Scripts/python.exe

# OR using pip
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

### 2. Train the model

```bash
.venv/Scripts/python model/train_model.py
```

Output:
- Compares Logistic Regression, Naive Bayes, Random Forest with 5-fold CV
- Selects best model by F1-macro
- Saves `sentiment_model.pkl`, `metadata.json`, charts

---

## Running the App

### Terminal 1 — Flask backend

```bash
cd customer_reviews
.venv/Scripts/python backend/app.py
# API live at http://localhost:5050
```

### Terminal 2 — Streamlit frontend

```bash
cd customer_reviews
.venv/Scripts/streamlit run frontend/streamlit_app.py
# UI live at http://localhost:8501
```

---

## API Reference

| Method | Endpoint                    | Description                              |
|--------|-----------------------------|------------------------------------------|
| GET    | `/health`                   | Liveness check + model name + record count|
| GET    | `/metadata`                 | Model info, metrics, top terms, CV results|
| POST   | `/predict`                  | Predict sentiment for one review text    |
| POST   | `/predict/batch`            | Predict sentiment for a list of texts    |
| GET    | `/analytics/summary`        | Overall KPIs (total, avg rating, etc.)   |
| GET    | `/analytics/sentiment`      | Sentiment counts + percentages           |
| GET    | `/analytics/rating`         | Rating distribution (1–5 stars)          |
| GET    | `/analytics/category`       | Per-category sentiment breakdown         |
| GET    | `/analytics/trend`          | Monthly sentiment trend                  |
| GET    | `/analytics/topwords`       | Top TF-IDF terms per sentiment class     |
| GET    | `/reviews`                  | Paginated review list (with filters)     |
| GET    | `/search?q=<keyword>`       | Full-text keyword search                 |

### Example — Predict Sentiment

```bash
curl -X POST http://localhost:5050/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing product, super fast delivery!"}'
```

Response:
```json
{
  "sentiment": "Positive",
  "confidence": 0.9823,
  "probabilities": {
    "Negative": 0.0041,
    "Neutral": 0.0136,
    "Positive": 0.9823
  }
}
```

### Example — Filter Reviews

```bash
# Negative reviews in Electronics, rating <= 2
curl "http://localhost:5050/reviews?sentiment=Negative&category=Electronics&max_rating=2&per_page=10"
```

---

## Frontend Pages

| Page | Description |
|---|---|
| **📊 Dashboard** | KPI cards, sentiment donut chart, rating bar chart, monthly trend line |
| **🤖 Sentiment Analyzer** | Live text input → AI prediction + confidence bars; batch multi-line analyzer |
| **🔍 EDA Explorer** | Category stacked bars, Rating × Sentiment heatmap, trend area chart, top TF-IDF terms |
| **📋 Review Browser** | Paginated & filterable review table with keyword search + CSV download |
| **ℹ️ Model Info** | Algorithm details, CV comparison chart, system architecture, dataset schema |

---

## Dataset Schema

| Column           | Type   | Description                          |
|------------------|--------|--------------------------------------|
| Review_ID        | string | Unique review identifier             |
| Order_ID         | string | Associated order identifier          |
| Customer_ID      | string | Anonymised customer ID               |
| Product          | string | Product name                         |
| Category         | string | Product category                     |
| Review_Date      | date   | YYYY-MM-DD                           |
| Rating           | int    | Star rating 1–5                      |
| Verified_Purchase| string | Yes / No                             |
| Helpful_Votes    | int    | Helpful vote count                   |
| Review_Text      | string | Raw review text (model input)        |
| Sentiment        | string | Target: Positive / Neutral / Negative|

---

## Model Details

- **Pipeline**: `TfidfVectorizer(max_features=15000, ngram_range=(1,2))` → `LogisticRegression`
- **Train/Test Split**: 80% / 20%, stratified
- **Evaluation**: 5-fold stratified cross-validation (F1-macro)
- **Test Accuracy**: 1.0000 | **F1 Macro**: 1.0000

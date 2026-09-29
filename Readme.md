# Enterprise Customer Churn Intelligence Platform

End-to-end Machine Learning and Analytics system that profiles customer attrition risk and exposes an ultra-low latency inference microservice.

## Pipeline Walkthrough

1. **Synthetic Data Engine**: Generates non-linear behavior profiles (tenure, ticket frequency, contract commitments).
2. **Analytics & Aggregations**: Extracts business metrics, monthly lost revenue, and cohort-level trends.
3. **Feature Preprocessor**: Modular transformers adding domain indicators (`charge_ratio`, `ticket_intensity`) and handling missing values without leakage.
4. **Machine Learning Model**: XGBoost trained with Stratified Cross-Validation and calibrated via optimal F1 decision boundary cutoffs.
5. **FastAPI Serving Engine**: Validates incoming payloads with Pydantic v2 and returns risk tiers (`LOW`, `MEDIUM`, `HIGH`).

## Quickstart

```bash
# 1. Setup Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate Dataset
python src/data_generator.py

# 3. Run EDA & Analytics Aggregation
python src/eda_analytics.py

# 4. Train Model & Save Bundle
python src/train.py

# 5. Run Test Suite
pytest tests/

# 6. Start API Server
uvicorn src.app:app --reload --port 8000
```

## API Sample Request

```bash
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure_months": 2,
       "contract_type": "Month-to-Month",
       "internet_service": "Fiber Optic",
       "payment_method": "Electronic Check",
       "monthly_charges": 92.50,
       "total_charges": 185.00,
       "support_tickets": 4,
       "paperless_billing": 1
     }'
```
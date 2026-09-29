"""
Synthetic data generator for telecom / SaaS churn intelligence.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_customer_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    customer_ids = [f"CUST-{100000 + i}" for i in range(n_samples)]
    tenure_months = np.random.exponential(scale=24, size=n_samples).clip(1, 72).astype(int)
    contract_type = np.random.choice(
        ["Month-to-Month", "One-Year", "Two-Year"], 
        size=n_samples, 
        p=[0.55, 0.25, 0.20]
    )
    internet_service = np.random.choice(
        ["Fiber Optic", "DSL", "No"], 
        size=n_samples, 
        p=[0.45, 0.35, 0.20]
    )
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n_samples,
        p=[0.35, 0.15, 0.25, 0.25]
    )
    
    # Monthly charges depend on service tier
    base_charges = {"Fiber Optic": 75.0, "DSL": 45.0, "No": 20.0}
    noise = np.random.normal(0, 5, n_samples)
    monthly_charges = np.array([base_charges[s] for s in internet_service]) + noise
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0).round(2)

    # Total charges with historical tenure effect
    total_charges = (monthly_charges * tenure_months * np.random.uniform(0.95, 1.05, n_samples)).round(2)
    
    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    paperless_billing = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # Probability of churn (ground truth non-linear logic)
    churn_logits = (
        -1.5
        - 0.05 * tenure_months
        + 0.03 * (monthly_charges - 50)
        + 0.45 * support_tickets
        + (contract_type == "Month-to-Month") * 0.9
        - (contract_type == "Two-Year") * 1.2
        + (payment_method == "Electronic Check") * 0.4
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logits))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "paperless_billing": paperless_billing,
        "churn": churn
    })

    # Inject realistic real-world missingness in 0.5% of total charges
    mask = np.random.rand(n_samples) < 0.005
    df.loc[mask, "total_charges"] = np.nan

    return df


if __name__ == "__main__":
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_customer_data(10000)
    file_path = out_dir / "customers.csv"
    df.to_csv(file_path, index=False)
    print(f"Generated {len(df)} customer records saved to {file_path}")
"""
Data Analytics: Statistical breakdown, cohort KPI analysis, and segmentation.
"""

from pathlib import Path
import pandas as pd


def compute_business_analytics(df: pd.DataFrame) -> dict:
    total_customers = len(df)
    total_churned = int(df["churn"].sum())
    overall_churn_rate = total_churned / total_customers
    total_mrr = df["monthly_charges"].sum()
    lost_mrr = df[df["churn"] == 1]["monthly_charges"].sum()

    print("=" * 60)
    print("EXECUTIVE BUSINESS ANALYTICS REPORT")
    print("=" * 60)
    print(f"Total Base:              {total_customers:,}")
    print(f"Churned Customers:       {total_churned:,} ({overall_churn_rate:.2%})")
    print(f"Total Monthly Run-rate:  ${total_mrr:,.2f}")
    print(f"Monthly Lost to Churn:   ${lost_mrr:,.2f} ({(lost_mrr/total_mrr):.2%})")
    print("-" * 60)

    # 1. Churn by Contract Type
    contract_kpis = df.groupby("contract_type").agg(
        customers=("customer_id", "count"),
        churn_rate=("churn", "mean"),
        avg_monthly=("monthly_charges", "mean"),
        avg_tickets=("support_tickets", "mean")
    ).reset_index()
    print("\n--- Churn by Contract Type ---")
    print(contract_kpis.to_string(index=False))

    # 2. Churn by Support Ticket Volume
    ticket_kpis = df.groupby("support_tickets").agg(
        volume=("customer_id", "count"),
        churn_rate=("churn", "mean")
    ).reset_index()
    print("\n--- Churn by Support Ticket Frequency ---")
    print(ticket_kpis.head(6).to_string(index=False))

    # 3. Tenure Cohorts
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]
    df["tenure_cohort"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)
    tenure_kpis = df.groupby("tenure_cohort", observed=True).agg(
        customers=("customer_id", "count"),
        churn_rate=("churn", "mean")
    ).reset_index()
    print("\n--- Churn by Tenure Cohort ---")
    print(tenure_kpis.to_string(index=False))

    return {
        "overall_churn_rate": overall_churn_rate,
        "lost_mrr": lost_mrr,
        "contract_summary": contract_kpis.to_dict(orient="records")
    }


if __name__ == "__main__":
    raw_path = Path("data/raw/customers.csv")
    if not raw_path.exists():
        raise FileNotFoundError("Run data_generator.py first to create raw dataset.")
    data = pd.read_csv(raw_path)
    compute_business_analytics(data)
# QuantFlow Financial Data Platform

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Apache Airflow](https://img.shields.io/badge/Apache_Airflow-017CEE?style=flat&logo=apacheairflow&logoColor=white)](https://airflow.apache.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

End-to-end financial analytics platform that automates the full Discounted Cash Flow (DCF) valuation pipeline — from real-time API ingestion through financial modeling, sensitivity analysis, peer benchmarking, and investor-ready Power BI dashboards.

---

## Key Results

| Metric | Result |
|---|---|
| Manual modeling eliminated | ~10 hours per valuation cycle |
| Adobe (ADBE) fair value estimate | $194 vs $416 market price |
| Implied signal | −53% downside (SELL) |
| Forecast horizon | 5-year NOPAT-based DCF + terminal value |

---

## Pipeline Architecture

```
yfinance + Alpha Vantage APIs
           │
           ▼ (Apache Airflow DAG)
┌─────────────────────────┐
│   Data Ingestion Layer  │  Income statement, balance sheet,
│                         │  cash flow, market prices
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Transformation Layer   │  NOPAT calculation, WACC derivation,
│                         │  free cash flow normalization
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   DCF Modeling Engine   │  5-year forecast + terminal value
│                         │  Peer benchmarking + multiples
│                         │  Sensitivity: WACC vs growth rate
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   PostgreSQL Storage    │  Versioned valuation snapshots
└───────────┬─────────────┘
            │
            ▼
     Power BI Dashboards
     Buy / Sell / Hold signal
```

---

## Features

**Automated Data Ingestion**
- Real-time financial data via yfinance and Alpha Vantage APIs
- Historical financials: income statement, balance sheet, cash flow
- Airflow DAG orchestrates ingestion on a scheduled basis

**DCF Modeling Engine**
- NOPAT-based free cash flow calculation
- WACC derivation from market data
- 5-year revenue and margin forecast with configurable assumptions
- Terminal value calculation using Gordon Growth Model
- Full sensitivity analysis: WACC vs terminal growth rate heatmaps

**Peer Benchmarking**
- Valuation multiples comparison across sector peers (EV/EBITDA, P/E, P/FCF)
- Relative positioning chart for buy/sell context

**Power BI Dashboards**
- Executive summary with fair value vs market price
- Valuation bridge showing value drivers
- Sensitivity matrix for scenario modeling
- Peer comparison panel

---

## Example Output — Adobe (ADBE)

| Item | Value |
|---|---|
| Fair value estimate | $194 |
| Market price (at model date) | $416 |
| Implied upside/downside | −53% |
| Signal | SELL |
| WACC used | 9.2% |
| Terminal growth rate | 3.0% |

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | Apache Airflow |
| Data APIs | yfinance, Alpha Vantage |
| Modeling | Python (Pandas, NumPy) |
| Database | PostgreSQL |
| Visualization | Power BI |
| AI Narrative | OpenAI GPT |

---

## Project Structure

```
Quantflow-Financial-Suite/
├── airflow/
│   └── dags/
│       ├── ingest_financials.py
│       └── run_valuation.py
├── modeling/
│   ├── dcf_engine.py         # Core DCF calculations
│   ├── wacc.py               # Cost of capital derivation
│   ├── sensitivity.py        # Scenario and sensitivity analysis
│   └── peer_benchmarking.py  # Multiples comparison
├── ingestion/
│   ├── yfinance_loader.py
│   └── alpha_vantage_loader.py
├── storage/
│   ├── schema.sql
│   └── db_utils.py
├── dashboard/
│   └── quantflow_dashboard.pbix
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/mukuldesai/Quantflow-Financial-Suite
cd Quantflow-Financial-Suite
pip install -r requirements.txt
cp .env.example .env          # Add API keys and DB credentials
python modeling/dcf_engine.py --ticker ADBE
```

---

## Disclaimer

This project is for educational and demonstration purposes only. It does not constitute investment advice. Consult a licensed financial professional before making investment decisions.

---

## Author

**Mukul Desai** — Data Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mukuldesai-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/mukuldesai)
[![Portfolio](https://img.shields.io/badge/Portfolio-mukuldesai.vercel.app-000000?style=flat&logo=vercel&logoColor=white)](https://mukuldesai.vercel.app)

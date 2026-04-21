# This file loads documents from multiple sources
# In production this would load from S3, databases, APIs
# We simulate with:
# - Fake company reports (generated with Faker)
# - Real Wikipedia articles (scraped with requests)
# - Sample financial data

import requests
import pandas as pd
import numpy as np
import os
import logging
from faker import Faker
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

fake = Faker()
random.seed(42)
Faker.seed(42)


class DocumentLoader:

    def __init__(self, output_dir: str = "data/documents"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)


    # This METHOD generates fake but realistic company reports
    # In real world: these would be SEC 10-K filings, earnings reports
    def generate_company_reports(self, n_companies: int = 10) -> list:

        logger.info(f"Generating {n_companies} company reports...")

        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Retail"]
        documents = []

        for i in range(n_companies):
            company  = fake.company()
            sector   = random.choice(sectors)
            revenue  = random.randint(100, 50000)
            growth   = round(random.uniform(-20, 50), 1)
            employees = random.randint(100, 100000)
            founded  = random.randint(1950, 2015)

            # Generate a realistic company report
            content = f"""
ANNUAL REPORT — {company}
Sector: {sector}
Founded: {founded}

EXECUTIVE SUMMARY
{company} is a leading {sector.lower()} company with operations across
multiple markets. In the fiscal year, the company reported revenue of
${revenue}M, representing a {growth}% year-over-year change.

FINANCIAL HIGHLIGHTS
- Annual Revenue: ${revenue}M
- Revenue Growth: {growth}%
- Total Employees: {employees:,}
- Markets: {', '.join([fake.country() for _ in range(3)])}

BUSINESS OVERVIEW
The company operates in the {sector} sector and has established
a strong market position through innovation and strategic acquisitions.
Key products include {fake.bs()} and {fake.bs()}.

RISK FACTORS
The company faces risks including market competition, regulatory changes,
and macroeconomic conditions. The {sector} sector is experiencing
{random.choice(['rapid growth', 'consolidation', 'disruption', 'steady expansion'])}.

OUTLOOK
Management expects {random.choice(['continued growth', 'margin expansion',
'market share gains', 'operational efficiency improvements'])} in the
coming fiscal year driven by {fake.bs()}.
            """.strip()

            doc = {
                "id":      f"company_{i+1}",
                "title":   f"{company} Annual Report",
                "content": content,
                "metadata": {
                    "company":   company,
                    "sector":    sector,
                    "revenue":   revenue,
                    "growth":    growth,
                    "employees": employees,
                    "type":      "annual_report"
                }
            }
            documents.append(doc)

        logger.info(f"Generated {len(documents)} company reports")
        return documents


    # This METHOD fetches real Wikipedia articles
    # Good source of real text for RAG testing
    def fetch_wikipedia_articles(self, topics: list) -> list:

        logger.info(f"Fetching {len(topics)} Wikipedia articles...")
        documents = []

        for topic in topics:
            try:
                # Wikipedia API — completely free, no key needed
                url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    doc = {
                        "id":      f"wiki_{topic.lower().replace(' ', '_')}",
                        "title":   data.get("title", topic),
                        "content": data.get("extract", ""),
                        "metadata": {
                            "source": "wikipedia",
                            "topic":  topic,
                            "type":   "encyclopedia"
                        }
                    }
                    documents.append(doc)
                    logger.info(f"Fetched: {topic}")

            except Exception as e:
                logger.warning(f"Failed to fetch {topic}: {e}")

        return documents


    # This METHOD generates financial Q&A pairs for testing RAG
    def generate_financial_qa(self) -> list:

        qa_pairs = [
            {
                "id": "qa_1",
                "title": "What is a P/E ratio?",
                "content": """
The Price-to-Earnings (P/E) ratio is a valuation metric that compares
a company's stock price to its earnings per share (EPS).
Formula: P/E = Stock Price / Earnings Per Share
A high P/E suggests investors expect high growth. A low P/E may indicate
undervaluation or slow growth expectations. The average S&P 500 P/E is
around 15-20x historically.
                """.strip(),
                "metadata": {"type": "financial_education", "topic": "valuation"}
            },
            {
                "id": "qa_2",
                "title": "What is the Sharpe Ratio?",
                "content": """
The Sharpe Ratio measures risk-adjusted return of an investment.
Formula: Sharpe = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
A Sharpe ratio above 1.0 is considered good, above 2.0 is excellent.
It was developed by Nobel laureate William F. Sharpe in 1966.
Higher Sharpe ratios indicate better risk-adjusted performance.
                """.strip(),
                "metadata": {"type": "financial_education", "topic": "risk_metrics"}
            },
            {
                "id": "qa_3",
                "title": "What is momentum investing?",
                "content": """
Momentum investing is a strategy that buys securities that have shown
upward price trends and sells those with downward trends.
Based on the momentum factor first documented by Jegadeesh and Titman (1993).
Typical lookback periods: 3-12 months, skipping the most recent month.
Works across stocks, bonds, commodities, and currencies.
Risk: momentum crashes can occur during market reversals.
                """.strip(),
                "metadata": {"type": "financial_education", "topic": "investment_strategy"}
            },
            {
                "id": "qa_4",
                "title": "What is machine learning in finance?",
                "content": """
Machine learning in finance applies algorithms to financial data to:
- Predict stock returns and market movements
- Detect fraud and anomalies in transactions
- Assess credit risk for loans
- Optimize trading strategies
- Automate customer service with NLP
Common algorithms: XGBoost, Random Forest, LSTM neural networks, transformers.
Key challenge: non-stationarity of financial time series data.
                """.strip(),
                "metadata": {"type": "financial_education", "topic": "machine_learning"}
            },
            {
                "id": "qa_5",
                "title": "What is a data warehouse?",
                "content": """
A data warehouse is a centralized repository for structured data from
multiple sources, optimized for analytical queries.
Key components: ETL/ELT pipelines, dimensional modeling, OLAP cubes.
Modern stack: dbt for transformation, Snowflake/BigQuery as warehouse,
Airflow for orchestration, Metabase/Looker for visualization.
Different from databases: optimized for reads not writes, historical data.
                """.strip(),
                "metadata": {"type": "technical_education", "topic": "data_engineering"}
            }
        ]
        return qa_pairs


    # This METHOD loads all documents from all sources
    def load_all(self) -> list:

        all_docs = []

        # Company reports
        all_docs.extend(self.generate_company_reports(10))

        # Wikipedia articles on finance topics
        topics = [
            "Quantitative_finance",
            "Machine_learning",
            "Data_engineering",
            "Stock_market",
            "Algorithmic_trading"
        ]
        all_docs.extend(self.fetch_wikipedia_articles(topics))

        # Financial Q&A
        all_docs.extend(self.generate_financial_qa())

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs


if __name__ == "__main__":
    loader = DocumentLoader()
    docs   = loader.load_all()
    print(f"\nLoaded {len(docs)} documents")
    for doc in docs[:3]:
        print(f"  - {doc['title']} ({len(doc['content'])} chars)")
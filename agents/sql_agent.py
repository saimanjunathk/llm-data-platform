# AI SQL AGENT
# Converts natural language questions into SQL queries
# Then executes the SQL and returns results
#
# Example:
# User: "Show me companies with revenue above $5000M"
# Agent: generates → SELECT * FROM companies WHERE revenue > 5000
# Agent: executes  → returns actual data
# Agent: explains  → "Found 3 companies with revenue above $5000M..."

import anthropic
import sqlite3
import pandas as pd
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SQLAgent:

    def __init__(self, db_path: str = "data/company_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        try:
            import streamlit as st
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key )
        self.model = "claude-haiku-4-5"

        # Create database with sample data
        self._setup_database()
        logger.info(f"SQL Agent ready | DB: {db_path}")


    def _setup_database(self):

        from faker import Faker
        import random

        fake = Faker()
        Faker.seed(42)
        random.seed(42)

        conn = sqlite3.connect(self.db_path)

        # Create companies table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id          INTEGER PRIMARY KEY,
                name        TEXT,
                sector      TEXT,
                revenue     REAL,
                growth_pct  REAL,
                employees   INTEGER,
                founded     INTEGER,
                country     TEXT
            )
        """)

        # Create financials table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                id          INTEGER PRIMARY KEY,
                company_id  INTEGER,
                year        INTEGER,
                revenue     REAL,
                profit      REAL,
                debt        REAL,
                FOREIGN KEY (company_id) REFERENCES companies(id)
            )
        """)

        # Only insert if empty
        count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        if count == 0:
            sectors = ["Technology", "Healthcare", "Finance", "Energy", "Retail"]
            companies = []
            for i in range(1, 51):
                companies.append((
                    i,
                    fake.company(),
                    random.choice(sectors),
                    round(random.uniform(100, 50000), 2),
                    round(random.uniform(-20, 50), 1),
                    random.randint(100, 100000),
                    random.randint(1950, 2015),
                    fake.country()
                ))

            conn.executemany("""
                INSERT INTO companies VALUES (?,?,?,?,?,?,?,?)
            """, companies)

            # Add financial history for each company
            for company_id in range(1, 51):
                base_rev = companies[company_id-1][3]
                for year in range(2019, 2025):
                    growth   = random.uniform(0.8, 1.3)
                    revenue  = round(base_rev * growth, 2)
                    profit   = round(revenue * random.uniform(0.05, 0.25), 2)
                    debt     = round(revenue * random.uniform(0.1, 0.5), 2)
                    conn.execute("""
                        INSERT INTO financials (company_id, year, revenue, profit, debt)
                        VALUES (?,?,?,?,?)
                    """, (company_id, year, revenue, profit, debt))

            conn.commit()
            logger.info("Database populated with sample data")

        conn.close()


    # This METHOD returns database schema for the LLM to understand
    def get_schema(self) -> str:
        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        schema = []
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for (table,) in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            col_info = ", ".join([f"{col[1]} {col[2]}" for col in columns])
            schema.append(f"Table: {table} ({col_info})")

        conn.close()
        return "\n".join(schema)


    # This METHOD generates SQL from natural language using Claude
    def generate_sql(self, question: str) -> str:

        schema = self.get_schema()

        prompt = f"""You are a SQL expert. Generate a SQLite SQL query to answer the question.

Database Schema:
{schema}

Rules:
- Return ONLY the SQL query, no explanation
- Use proper SQLite syntax
- Limit results to 20 rows maximum
- Use ORDER BY when appropriate

Question: {question}

SQL Query:"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        sql = response.content[0].text.strip()

        # Clean up SQL (remove markdown code blocks if present)
        sql = sql.replace("```sql", "").replace("```", "").strip()

        logger.info(f"Generated SQL: {sql[:100]}...")
        return sql


    # This METHOD executes SQL and returns results as DataFrame
    def execute_sql(self, sql: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            df   = pd.read_sql(sql, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return pd.DataFrame({"error": [str(e)]})


    # This METHOD explains results in natural language
    def explain_results(self, question: str, sql: str, df: pd.DataFrame) -> str:

        if df.empty or "error" in df.columns:
            return "I encountered an error executing the query."

        prompt = f"""Given this question and SQL results, provide a brief natural language summary.

Question: {question}
SQL Used: {sql}
Results (first 5 rows): {df.head().to_string()}
Total rows: {len(df)}

Provide a 2-3 sentence summary of the results."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()


    # This METHOD runs the full pipeline: question → SQL → results → explanation
    def ask(self, question: str) -> dict:

        # Generate SQL
        sql = self.generate_sql(question)

        # Execute SQL
        df = self.execute_sql(sql)

        # Explain results
        explanation = self.explain_results(question, sql, df)

        return {
            "question":    question,
            "sql":         sql,
            "results":     df,
            "explanation": explanation
        }
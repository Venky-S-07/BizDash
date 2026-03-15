# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import os
import io
import json

import pandas as pd
import plistlib
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

# ─────────────────────────────────────────────
# Load environment variables and init client
# ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# ─────────────────────────────────────────────
# Schema Extraction
# ─────────────────────────────────────────────
def extract_schema(df: pd.DataFrame) -> dict:
    """
    Extracts a summary schema from a DataFrame.

    Returns a dict containing row/column counts and
    per-column metadata (dtype, null counts, unique values).
    """
    schema = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_details": []
    }

    for col in df.columns:
        schema["column_details"].append({
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notnull().sum()),
            "null": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique())
        })

    return schema


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────
def build_prompt(schema: dict, user_query: str) -> str:
    """
    Builds a structured prompt for the Gemini model.

    Includes the dataset schema and user query, and instructs
    the model to return a JSON object with pandas code + chart config.
    """
    # Build a readable schema summary block
    schema_context = (
        f"Dataset Summary\n"
        f"Total Rows: {schema['rows']}\n"
        f"Total Columns: {schema['columns']}\n"
        f"Columns:\n"
    )
    for col in schema["column_details"]:
        schema_context += (
            f"  - {col['name']}\n"
            f"      type: {col['dtype']}\n"
            f"      unique values: {col['unique_values']}\n"
            f"      null values: {col['null']}\n"
        )

    prompt = f"""
You are a Python Data Analyst bot.

DATASET SCHEMA:
{schema_context}

USER REQUEST: "{user_query}"

OUTPUT FORMAT:
Return ONLY a valid JSON object. Do not include markdown formatting like ```json.
The JSON must follow this structure:
{{
    "reasoning": "Brief explanation of the logic used.",
    "pandas_code": "The exact Python code to transform the dataframe 'df' into the required result set.",
    "chart_type": "one of ['bar', 'line', 'pie', 'scatter']",
    "chart_config": {{
        "x": "column_name_for_x_axis",
        "y": "column_name_for_y_axis",
        "color": "column_name_for_grouping_or_breakdown (return null if not applicable)",
        "title": "A descriptive title"
    }}
}}

CONSTRAINTS:
- Use only the dataframe named 'df'.
- Ensure 'pandas_code' results in a variable named 'result_df'.
- If the X-axis represents time (months, days, years), the pandas_code MUST sort the
  dataframe chronologically. If grouping by month name, sort by calendar order, not alphabetically.
- If the query cannot be answered, return {{"error": "reason"}}.
- Explicitly filter the dataframe first if the user specifies a timeframe (e.g., 'H1', 'Q4', '2023').
- If calculating averages, counts, or growth, rename the resulting column so the Y-axis label
  accurately reflects the metric (e.g., 'average_order_value' instead of 'total_revenue').
- If the user asks a direct question (e.g., "Which region is worst?"), state the answer
  clearly in the "reasoning" field.
"""
    return prompt


# ─────────────────────────────────────────────
# Model Inference
# ─────────────────────────────────────────────
def model_r(dataframe: pd.DataFrame, user_query: str = "What is the Q3 revenue each month") -> dict:
    """
    Sends the dataset schema + user query to Gemini and returns
    a parsed JSON response containing pandas code and chart config.

    Args:
        dataframe:   The input pandas DataFrame to analyse.
        user_query:  Natural language question about the data.

    Returns:
        A dict with keys: reasoning, pandas_code, chart_type, chart_config.
        Returns {"error": "reason"} if the query can't be answered.
    """
    schema = extract_schema(dataframe)
    prompt = build_prompt(schema, user_query)

    # Call the Gemini model
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt
    )

    text = response.text.strip()

    # Strip markdown code fences if the model wraps its response in them
    if text.startswith("```"):
        text = text.split("```")[1]
    text = text.replace("json", "").strip()

    return json.loads(text)

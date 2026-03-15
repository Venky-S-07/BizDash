from google import genai
import pandas as pd
import plistlib
import json
from bs4 import BeautifulSoup
import io
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def extract_schema(df):
    schema = {}

    schema["rows"] = len(df)
    schema["columns"] = len(df.columns)

    column_info = []

    for col in df.columns:
        info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notnull().sum()),
                "null": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique())
        }

        column_info.append(info)

    schema["column_details"] = column_info

    return schema


def build_prompt(schema, user_query):
    schema_context = f"""
    Dataset Summary

    Total Rows: {schema['rows']}
    Total Columns: {schema['columns']}

    Columns:
    """

    for col in schema["column_details"]:
        schema_context += f"""
        - {col['name']}
            type: {col['dtype']}
            unique values: {col['unique_values']}
            null values: {col['null']}
        """

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
        "pandas_code": "The exact python code to transform the dataframe 'df' into the required result set.",
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
    - If the X-axis represents time (months, days, years), the pandas_code MUST sort the dataframe chronologically before returning result_df and If you group by month names, you must ensure the final table is sorted by chronological calendar order, not alphabetically.
    - If the query cannot be answered, return {{"error": "reason"}}.
    - Explicitly filter the dataframe first if the user asks for a specific timeframe (e.g., 'H1', 'Q4', '2023').
    - If calculating averages, counts, or growth, rename the resulting column in pandas_code so the chart's Y-axis label accurately reflects the metric (e.g., 'average_order_value' instead of 'total_revenue').
    - If the user asks a direct question (e.g., "Which region is worst?"), ensure the answer is clearly stated in the "reasoning" JSON field.
    """

    return prompt


def model_r(dataframe, user_query="What is the q3 revenue each month"):
    schema = extract_schema(dataframe)
    prompt = build_prompt(schema,user_query)

    response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview", 
            contents=prompt
    )

    text = response.text.strip()

    # remove markdown if present
    if text.startswith("```"):
        text = text.split("```")[1]

    text = text.replace("json", "").strip()

    return json.loads(text)

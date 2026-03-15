import pandas as pd
import plotly.express as px

def execute_ai_code(ai_response, df):

    if "pandas_code" not in ai_response:
        raise ValueError(f"AI Response does not contain 'pandas_code'. Error: {ai_response.get('error', 'Unknown')}")

    code = ai_response["pandas_code"]

    blocked = ["import os", "import sys", "open(", "exec(", "eval("]

    for word in blocked:
        if word in code:
            raise ValueError("Unsafe code detected")

    local_env = {
            "df": df,
            "pd": pd
    }

    exec(code, {}, local_env)

    result_df = local_env.get("result_df")

    if result_df is None:
        raise ValueError("AI code must create a variable named result_df")

    return result_df

def create_chart(result_df, ai_response):

    chart_type = ai_response["chart_type"].lower().strip()
    config = ai_response["chart_config"]

    x = config.get("x")
    y = config.get("y")
    color = config.get("color")
    title = config.get("title")

    if color == "null" or color == "":
        color = None

    if chart_type == "bar":
        fig = px.bar(result_df, x=x, y=y, color=color, title=title, barmode = 'group')

    elif chart_type == "line":
        fig = px.line(result_df, x=x, y=y, color=color, title=title)

    elif chart_type == "scatter":
        fig = px.scatter(result_df, x=x, y=y, color=color, title=title)

    elif chart_type == "pie":
        fig = px.pie(result_df, names=x, values=y, title=title)

    else:
        raise ValueError("Unsupported chart type")

    return fig

if __name__ == "__main__":
    from main import model_r
    
    test_df = pd.DataFrame({"Month": ["Jan", "Feb"], "Revenue": [100, 200]})
    try:
        response = model_r(test_df, "What is the revenue by month?")
        res_df = execute_ai_code(response, test_df)
        print(res_df)
        f = create_chart(res_df, response)
        f.show()
    except Exception as e:
        print(f"Test execution failed: {e}")

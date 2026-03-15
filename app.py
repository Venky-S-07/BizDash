import streamlit as st
import pandas as pd
import plistlib
import io
from bs4 import BeautifulSoup
from main import extract_schema, build_prompt, model_r
from exec import execute_ai_code, create_chart
import plotly.express as px

st.set_page_config(page_title="InsightAI | Business Intelligence", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Overall Background and Text */
    .stApp {
        background-color: #0d1117; 
        color: #e6edf3;
    }

    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input, .stChatInputContainer textarea {
        background-color: #0d1117;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stChatInputContainer textarea:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #0d1117;
        border: 1px dashed #30363d;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }

    /* Titles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .title-highlight {
        background: -webkit-linear-gradient(45deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

</style>
""", unsafe_allow_html=True)

# Custom Title
st.markdown("<h1 style='text-align: center;'><span class='title-highlight'>Visual</span> Business Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>Conversational insights, powered by Gemini.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3>Data Configuration</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Business Data (CSV)", type="csv")

    if uploaded_file:
        @st.cache_data
        def load_data(file):
            try:
                # First try treating it as a raw CSV file
                return pd.read_csv(file)
            except Exception:
                # Reset file pointer for the secondary parsing logic
                file.seek(0)
                file_bytes = file.read()
                
                # Check if it's the specific plist/webarchive format previously exported
                try:
                    data = plistlib.loads(file_bytes)
                    html_bytes = data["WebMainResource"]["WebResourceData"]
                    html = html_bytes.decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    csv_text = soup.find("pre").text
                    return pd.read_csv(io.StringIO(csv_text))
                except Exception as e:
                    # If all parsing fails, raise an exception or allow the app to handle it gracefully
                    st.error(f"Failed to parse uploaded data. Ensure it's a valid CSV or WebArchive Plist. Error: {e}")
                    st.stop()

        df = load_data(uploaded_file)
        if df is not None:
            st.success("Data successfully loaded")
            
            with st.expander("Data Preview", expanded=True):
                st.dataframe(df.head(5), use_container_width=True)
    else:
        df = None

# Main Query Interface
if df is None:
    st.info("Welcome! Please upload a CSV dataset in the **sidebar** to begin generating insights.")
    query = st.chat_input("Upload a CSV dataset first to ask questions...", disabled=True)
else:
    query = st.chat_input("Ask a question about your data (e.g., 'What is the highest revenue by month?')")


if query and df is not None:
    st.markdown(f"**You asked:** {query}")

    with st.spinner("Analyzing data and verifying insights..."):
        max_retries = 3
        current_query = query
        success = False
        ai_response = {}
        error_msg = ""
        
        for attempt in range(max_retries):
            # UI Updates so the user knows it's "thinking deeply"
            if attempt == 0:
                st.toast("AI drafting initial logic...")
            elif attempt == 1:
                st.toast("AI reviewing and verifying data...")
            else:
                st.toast("AI applying final corrections...")
                
            ai_response = model_r(df, current_query)

            if "error" in ai_response:
                st.error(f"AI declined to answer: {ai_response['error']}")
                break 

            try:
                # Execute the AI's code
                result_df = execute_ai_code(ai_response, df)
                fig = create_chart(result_df, ai_response)
                
                # --- THE CRITIC STEP (Only happens on the first try) ---
                if attempt == 0:
                    data_preview = result_df.head(5).to_markdown()
                    
                    # Force the AI to look at its own output
                    current_query = f"""
                    Original Query: {query}
                    
                    You just wrote code that generated a dataframe with shape {result_df.shape}.
                    Here are the first 5 rows of your output:
                    {data_preview}
                    
                    CRITIC STEP: Review this output carefully. Does it perfectly answer the original query? 
                    Are the items sorted logically? Is the data properly aggregated, or are there duplicate categories? 
                    If it is perfect, return the exact same JSON format with the same code. 
                    If there is ANY flaw, update the 'pandas_code', 'reasoning', and 'chart_config' to fix it and return the improved JSON.
                    """
                    continue # Skips the rest of this loop and forces Attempt 1
                
                # If we get here (attempt > 0), the code works and the AI has verified it!
                success = True
                break 
                
            except Exception as e:
                # --- THE ERROR HEALING STEP ---
                error_msg = str(e)
                bad_code = ai_response.get('pandas_code', '# No code')
                
                current_query = f"""
                Original Query: {query}
                Your previous python code: {bad_code}
                Failed with this error: {error_msg}
                Please fix the pandas_code to resolve this error. Return ONLY the valid JSON format.
                """

        # --- DISPLAY THE RESULTS ---
        if success:
             st.info(f"**AI Reasoning:** {ai_response.get('reasoning', 'No reasoning provided.')}")
             st.markdown("<br>", unsafe_allow_html=True)
             col1, col2 = st.columns([3, 2], gap="large")

             with col1:
                 st.markdown("### Visualization")
                 fig.update_layout(
                     template="plotly_dark",
                     plot_bgcolor="rgba(0,0,0,0)",
                     paper_bgcolor="rgba(0,0,0,0)",
                     margin=dict(t=40, b=0, l=0, r=0),
                     font=dict(family="Inter", color="#c9d1d9")
                 )
                 st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

             with col2:
                 st.markdown("### Data Summary")
                 st.dataframe(result_df, use_container_width=True)

             with st.expander("View Pipeline Code"):
                 st.code(ai_response.get('pandas_code', '# No code generated'), language='python')
                 
        elif not success and "error" not in ai_response:
             st.error(f"Execution failed after multiple attempts. Last error: {error_msg}")










# if query and df is not None:
#     st.markdown(f"**You asked:** {query}")
#     with st.spinner("Analyzing data and generating visualization..."):
#         try:
#             ai_response = model_r(df, query)
#
#             if "error" in ai_response:
#                  st.error(f"AI declined to answer: {ai_response['error']}")
#             else:
#                  st.info(f"**AI Reasoning:** {ai_response.get('reasoning', 'No reasoning provided.')}")
#
#                  # Card-like layout
#                  st.markdown("<br>", unsafe_allow_html=True)
#                  col1, col2 = st.columns([3, 2], gap="large")
#
#                  with col1:
#                      st.markdown("### Visualization")
#                      result_df = execute_ai_code(ai_response, df)
#                      fig = create_chart(result_df, ai_response)
#
#                      # Improve Plotly Theme integration
#                      fig.update_layout(
#                          template="plotly_dark",
#                          plot_bgcolor="rgba(0,0,0,0)",
#                          paper_bgcolor="rgba(0,0,0,0)",
#                          margin=dict(t=40, b=0, l=0, r=0),
#                          font=dict(family="Inter", color="#c9d1d9")
#                      )
#
#                      st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
#
#                  with col2:
#                      st.markdown("### Data Summary")
#                      st.dataframe(result_df, use_container_width=True)
#
#                  with st.expander("View Pipeline Code"):
#                      st.code(ai_response.get('pandas_code', '# No code generated'), language='python')
#
#         except Exception as e:
#             st.error(f"Execution Error: {e}")
#             st.warning("Please try rephrasing your query or checking your dataset format.")

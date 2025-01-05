import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import google.generativeai as genai

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Streamlit Page Configuration
st.set_page_config(page_title="Titanic Q&A Explorer with Gemini", layout="wide")

# Configure Gemini API
API_KEY = "AIzaSyD5OLJgDUvIpushd9jza_6DVM8hOBNL_MY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load Titanic Dataset or Upload Custom Dataset
@st.cache_data
def load_data(default=True, uploaded_file=None):
    if default:
        return sns.load_dataset("titanic")
    else:
        return pd.read_csv(uploaded_file)

# Provide Dataset Description to Gemini
def get_dataset_description(dataset):
    """
    Generate a dataset description that explains the schema to Gemini.
    """
    description = f"The dataset contains the following columns:\n"
    for col in dataset.columns:
        dtype = dataset[col].dtype
        description += f"- {col}: {dtype}\n"
    
    # Optionally, provide a few rows as an example for clarity
    sample_data = dataset.head(3).to_string(index=False)
    description += f"\nExample data:\n{sample_data}\n"
    
    return description

# Gemini API Integration for Code Generation
def generate_code(query, dataset):
    """
    Generate Python code for a user query, based on dataset description.
    """
    dataset_description = get_dataset_description(dataset)
    
    prompt = f"""
    You are given a dataset named 'dataset' which is a Pandas DataFrame.
    The dataset contains the following columns: {', '.join(dataset.columns)}
    It is already loaded and ready to be used.

    The task is to generate Python code to answer the following question using the 'dataset':

    Query: {query}

    Write Python code only (without description) that will use the 'dataset' DataFrame to answer the question
    above. if you plot then that plot should also be stored in fig variable. The answer of the query should be               
    stored in result variable with proper text in context of query. the answer can be figure as well. please dont       
    use plotly and mention legends in plots.
    The generated code should assume that 'dataset' is already defined. please preprocess the data if necessary.
    """
    
    # Generate code using Gemini
    response = model.generate_content(prompt)
    response_text = response.text.strip()

    # Remove any unwanted Python keywords or context
    cleaned_response = response_text.replace("```python", "").strip()
    cleaned_response = cleaned_response.replace("```", "").strip()
    return cleaned_response

# Execute the Generated Code and Return Results
def execute_generated_code(generated_code, dataset):
    """
    Execute the generated code safely and return the results or figures.
    """
    exec_locals = {}
    try:
        exec(generated_code, {"dataset": dataset}, exec_locals)
        
        # Ensure result or fig exists in the executed code
        result = exec_locals.get("result", None)
        fig = exec_locals.get("fig", None)

        return result, fig
    except Exception as e:
        return f"Error executing the code: {e}", None

# App Title
st.title("🚢 Titanic Data Explorer with Gemini")
st.write("Ask questions about the Titanic dataset, explore visualizations, and analyze custom data!")

# Sidebar for Dataset Selection
with st.sidebar:
    st.header("Dataset Options")
    default_data = st.radio("Select Dataset:", ["Default Titanic Dataset", "Upload Custom Dataset"])
    uploaded_file = None
    if default_data == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load Selected Dataset
data = load_data(default=(default_data == "Default Titanic Dataset"), uploaded_file=uploaded_file)

# Display Dataset
if st.checkbox("Preview Dataset"):
    st.dataframe(data.head())

# User Input for Query
query = st.text_input("Enter your question:", "How many males survived?")
tabs = st.tabs(["Results", "Generated Code", "Dataset Insights"])

if st.button("Generate Answer"):
    with st.spinner("Processing your query..."):
        # Generate the code from Gemini API
        generated_code = generate_code(query, data)
        
        with tabs[1]:
            st.subheader("Generated Code")
            st.code(generated_code, language="python")
        
        # Execute the code and display results
        result, fig = execute_generated_code(generated_code, data)
        
        with tabs[0]:
            st.subheader("Results")
            if result:
                st.write(result)  # Show the result (text output)
            if fig:
                st.plotly_chart(fig, use_container_width=True)  # Show the Plotly figure
            
        with tabs[2]:
            st.subheader("Dataset Insights")
            st.write("Basic Statistics:")
            st.write(data.describe(include="all"))
            st.write("Missing Values:")
            st.write(data.isnull().sum())

# Additional Widgets for User Interaction
with st.sidebar:
    st.header("App Settings")
    font_size = st.slider("Font Size for Results", 12, 24, 16)
    dark_mode = st.checkbox("Enable Dark Mode")
    st.info("Powered by Gemini API")

# Apply Custom Styling
st.markdown(
    f"""
    <style>
    .reportview-container {{
        font-size: {font_size}px;
        background-color: {'#1e1e1e' if dark_mode else '#ffffff'};
        color: {'#ffffff' if dark_mode else '#000000'};
    }}
    .sidebar .sidebar-content {{
        background-color: {'#333333' if dark_mode else '#f4f4f4'};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
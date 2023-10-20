from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
#from pandasai.llm import AzureOpenAI
import matplotlib.pyplot as plt


#load_dotenv()

#API_KEY = os.environ['OPENAI_API_KEY']

from pandasai.llm import AzureOpenAI

llm = AzureOpenAI(
    api_token="f769445c82844edda56668cb92806c21",
    api_base="https://aoiaipsi.openai.azure.com",
    api_version="2023-07-01-preview",
    deployment_name="gpt-35-turbo-0613"
)

#llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)


st.title("Data visualization app Using PandasAI")
uploaded_file = st.file_uploader("upload a files", accept_multiple_files=True, type=['.csv','.db'])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))
    
    prompt = st.text_area("Enter your prompt:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating answer....."):
                st.write(pandas_ai.run(df,prompt=prompt))
        else:
            st.warning("please enter a prompt")
    
    
    
    
    chart_type = st.sidebar.selectbox("Choose a chart type", ["Bar Chart", "Line Chart", "Pie Chart"])
    if chart_type == "Bar Chart":

        st.bar_chart(df.head(5))

    elif chart_type == "Line Chart":

        st.line_chart(df.head(5))

    elif chart_type == "Pie Chart":

        st.pie_chart(df)

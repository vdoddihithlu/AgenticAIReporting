import os
import pandas as pd
import plotly.express as px
from pandasql import sqldf

from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



# Initialize the HuggingFaceEndpoint LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",   # 👈 works on Inference API
    task="text2text-generation", 
    temperature=0.7,
    max_new_tokens=50
)

response = llm.invoke("What is the capital of France?")
print(response)
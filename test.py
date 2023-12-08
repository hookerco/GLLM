import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_token"]


template = """Question: {question}

please answer the question above using one short sentence consisting of 10 words. """

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = 'codellama/CodeLlama-7b-Instruct-hf'
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = 'Generate code to create an array and print each element'
print(llm_chain.run(question))


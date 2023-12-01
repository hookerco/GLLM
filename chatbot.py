import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_token"]

MODELS = {'bloom': 'bigscience/bloom',
          'gpt2': 'gpt2',
          'flan-t5-xxl': 'google/flan-t5-xxl',
          'starchat': 'HuggingFaceH4/starchat-beta',
          'zephyr-7b-beta': 'HuggingFaceH4/zephyr-7b-beta', ##### good reply, finetuned version of mistral
          'vicuna-7b': 'TheBloke/vicuna-7B-v1.5-GPTQ',  ### understands context, answers are cut short
          'guanaco-33b': 'timdettmers/guanaco-33b-merged',
          'CodeLlama-34b': 'victor/CodeLlama-34b-Instruct-hf',  ## understands context, very short incomplete answer
          'falcon-7b-instruct': 'tiiuae/falcon-7b-instruct'
          #'Mistral 7B': 'mistralai/Mistral-7B-v0.1' error
          }

with st.sidebar:
    model = st.selectbox('Which model would you like to use?', tuple(MODELS.keys()))

# Define LLM, prompt template and chain
llm = HuggingFaceHub(
    repo_id=MODELS[model], model_kwargs={"temperature": 0.9, "max_length": 500}
)

template = """
You are a helpful chatbot that wants to help people use industrial machines easier. Your purpose is to 
serve people and cater to their needs.

Question: {question}.

IMPORTANT INSTRUCTION: respond below this line.

"""

prompt = PromptTemplate(template=template, input_variables=["question"])

st.title("💬 Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# React to user input
if msg := st.chat_input("How can I help you?"):

    client = LLMChain(prompt=prompt, llm=llm)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": msg})
    # Display user message in chat message container
    st.chat_message("user").write(msg)

    response = client.run(msg)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

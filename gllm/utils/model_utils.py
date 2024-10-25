"""
Description of this file:

This file contains utility functions for setting up and managing machine learning models in a Langchain application. 
The models are used to generate G-codes from natural language instructions for CNC machines. 
Various models including Zephyr-7b, Fine-tuned StarCoder, GPT-3.5, and CodeLlama are supported, with configurations tailored for text generation tasks.

The utilities are implemented in Python and utilize libraries such as Transformers, Langchain, and Hugging Face APIs 
to ensure seamless integration and execution within the application.

Authors: Mohamed Abdelaal, Samuel Lokadjaja

This work was done at Software AG, Darmstadt, Germany in 2023-2024 and is published under the Apache License 2.0.
"""

import os
import toml
import openai
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, pipeline 
from langchain_openai import ChatOpenAI
from utils.prompts_utils import SYSTEM_MESSAGE
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers import AutoTokenizer


# Define the path to the secrets.toml file
secrets_file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.streamlit', 'secrets.toml'))
# Load the secrets
secrets = toml.load(secrets_file_path)
# Set your OpenAI API key
openai.api_key = secrets["openai_token"]


def setup_model(model:str):
    if model == "Zephyr-7b":
        ENDPOINT_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        llm = HuggingFaceEndpoint(
                endpoint_url=ENDPOINT_URL,
                task="text-generation",
                max_new_tokens=512,
                top_k=50,
                temperature=0.1,
                repetition_penalty=1.03,)
        
    elif model == "Fine-tuned StarCoder":
        # load the base model
        config = PeftConfig.from_pretrained("ArneKreuz/starcoderbase-finetuned-thestack")
        base_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-3b")
        # Load the fine tuned model
        llm = PeftModel.from_pretrained(base_model, "ArneKreuz/starcoderbase-finetuned-thestack", force_download=True)

    elif model == "GPT-3.5":
        #llm = OpenAI(api_key=openai.api_key)
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7, api_key=openai.api_key)

    elif model == 'CodeLlama':
        model_name = "codellama/CodeLlama-7b-hf"
        llm = AutoModelForCausalLM.from_pretrained(model_name)
        ## llm = pipeline("text-generation", model="codellama/CodeLlama-7b-hf")
        

    return llm

def setup_langchain_without_rag(model):
    # create a prompt
    prompt = ChatPromptTemplate.from_messages(
            [
                (
                 "system", SYSTEM_MESSAGE),
                ("human", "{input}"),
            ])
    model_chain =  prompt | model
    
    # Here we assume the model name is compatible with Hugging Face's interfac
    #model_chain = ChatHuggingFace(llm=model) 

    return model_chain
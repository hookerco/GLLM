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
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from langchain_openai import ChatOpenAI
from utils.prompts_utils import SYSTEM_MESSAGE
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.chat_models.huggingface import ChatHuggingFace

# Define the path to the secrets.toml file
secrets_file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.streamlit', 'secrets.toml'))
# Load the secrets
secrets = toml.load(secrets_file_path)
# Set your OpenAI API key
openai.api_key = secrets["openai_token"]
# Optionally set your OpenRouter API key
openrouter_api_key = secrets.get("openrouter_token")


def setup_model(model: str):
    if model == "Zephyr-7b":
        ENDPOINT_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        llm = HuggingFaceEndpoint(
            endpoint_url=ENDPOINT_URL,
            task="text-generation",
            max_new_tokens=512,
            top_k=50,
            temperature=0.1,
            repetition_penalty=1.03,
        )
    elif model == "Fine-tuned StarCoder":
        # Load the fine tuned model and wrap it in a pipeline so it can be used
        # directly with LangChain runnables.
        PeftConfig.from_pretrained("ArneKreuz/starcoderbase-finetuned-thestack")
        base_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-3b")
        peft_model = PeftModel.from_pretrained(base_model, "ArneKreuz/starcoderbase-finetuned-thestack", force_download=True)
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-3b")
        hf_pipeline = pipeline(
            "text-generation",
            model=peft_model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.03,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
    elif model == "GPT-3.5":
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7, api_key=openai.api_key)
    elif model == "OpenRouter":
        openrouter_model = secrets.get("openrouter_model", "openai/gpt-3.5-turbo")
        llm = ChatOpenAI(
            model=openrouter_model,
            temperature=0.7,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/mohamedyd/GLLM",
                "X-Title": "GLLM",
            },
        )
    elif model == 'CodeLlama':
        # Wrap the model inside a transformers pipeline to ensure text input works with LangChain.
        model_name = "codellama/CodeLlama-7b-hf"
        model_llm = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_pipeline = pipeline(
            "text-generation",
            model=model_llm,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.03,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        raise ValueError(f"Unsupported model: {model}")

    return llm


def setup_langchain_without_rag(model):
    # create a prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            ("human", "{input}"),
        ]
    )
    model_chain = prompt | model

    # Here we assume the model name is compatible with Hugging Face's interface
    # model_chain = ChatHuggingFace(llm=model)

    return model_chain

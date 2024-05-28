import openai
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from utils.prompts_utils import SYSTEM_MESSAGE
from langchain_community.chat_models.huggingface import ChatHuggingFace

def setup_model(model:str):
    if model == "finetuned_starcoder":
        llm = HuggingFaceHub(
                            repo_id="HuggingFaceH4/zephyr-7b-beta", ################# TODO change to StarCoder
                            task="text-generation",
                            model_kwargs={
                                "max_new_tokens": 512,
                                "top_k": 30,
                                "temperature": 0.1,
                                "repetition_penalty": 1.03,
                            }, )
    elif model == "GPT-3.5":
        #llm = OpenAI(api_key=openai.api_key)
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7, api_key=openai.api_key)

    return llm

def setup_langchain_without_rag(is_openai, model):
    if is_openai:
        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system", SYSTEM_MESSAGE),
                    ("human", "{input}"),
                ])
        model_chain =  prompt | model
    else:
        # Here we assume the model name is compatible with Hugging Face's interfac
        model_chain = ChatHuggingFace(llm=model) 

    return model_chain
import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer

import numpy as np
import evaluate
from glob import glob


os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_token"]

# template = """Question: {question}
#
# please answer the question above using one short sentence consisting of 10 words. """
#
# prompt = PromptTemplate(template=template, input_variables=["question"])
#
# repo_id = 'codellama/CodeLlama-7b-Instruct-hf'
# llm = HuggingFaceHub(
#     repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
# )
# llm_chain = LLMChain(prompt=prompt, llm=llm)
#
# question = 'Generate code to create an array and print each element'
# print(llm_chain.run(question))

################################################################################################


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
# trainer.train()

################################################################################################

# collect all pdf files recursively
# my_path = 'pdfs'
# files = glob(my_path + '/**/*.pdf', recursive=True)
# print(len(files))


# links
# https://huggingface.co/docs/transformers/tasks/language_modeling
# https://github.com/moabdelmoez/llm-projects/blob/main/talk-to-your-data/run.py
# https://python.langchain.com/docs/expression_language/cookbook/retrieval

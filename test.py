
# This file is just used for playing around

import os
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer, \
    AutoModelForCausalLM, pipeline
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


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# dataset = load_dataset("yelp_review_full")
# tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#
# training_args = TrainingArguments(output_dir="test_trainer")
# metric = evaluate.load("accuracy")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
# trainer.train()

################################################################################################

# collect all pdf files recursively
# my_path = 'pdfs'
# files = glob(my_path + '/**/*.pdf', recursive=True)
# print(len(files))

################################################################################################

# Testing question-answer pair generation from text

# from pipelines import pipeline
# import nltk
#
# nltk.download('punkt')
# nlp = pipeline("question-generation", model="GermanT5/t5-base-german-3e", qg_format="prepend")
# qa = nlp("Die Grundidee der HELLER Lernfabrik ist, den Bau und die Fertigung einer Werkzeugmaschine durch den tatsächlichen Bau und die Fertigung einer solchen zu erlernen. Dabei werden alle Kompetenzen, Abläufe und Zusammenhänge von der Entwicklung über die Produktion und Montage bis hin zur Auslieferung geschult und am realen Objekt umgesetzt.")
# print(qa)

# try only using huggingface model straight
# ml6team/mt5-small-german-query-generation
# dehio/german-qg-t5-e2e-quad

# llm = HuggingFaceHub(repo_id="ml6team/mt5-small-german-query-generation", model_kwargs={"temperature": 0.9, "max_length": 500})

################################################################################################

# Testing finetuned model
# https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu#inference
# def get_code_completion(prefix, suffix):
#     text = prompt = f"""<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"""
#     model.eval()
#     outputs = model.generate(
#         input_ids=tokenizer(text, return_tensors="pt").input_ids.cuda(),
#         max_new_tokens=1024,
#         temperature=0.2,
#         top_k=50,
#         top_p=0.95,
#         do_sample=True,
#         repetition_penalty=1.0,
#     )
#     return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#
# from peft import PeftModel
# base_model = 'WizardLM/WizardCoder-3B-V1.0'
# finetuned_params = "finetuned_model"
#
# tokenizer = AutoTokenizer.from_pretrained(base_model)
# model = AutoModelForCausalLM.from_pretrained(base_model)
#
# finetuned_model = PeftModel.from_pretrained(model, finetuned_params)
# finetuned_model.merge_and_unload()
#
#
# text = "Generate G-code for me for milling"
# model.eval()
# outputs = model.generate(
#     input_ids=tokenizer(text, return_tensors="pt").input_ids,
#     max_new_tokens=1024,
#     temperature=0.2,
#     top_k=50,
#     top_p=0.95,
#     do_sample=True,
#     repetition_penalty=1.0,
# )
# outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
# print(outputs_text)

################################################################################################
# Testing the stack dataset

def group_texts(examples, block_size=128):
    # Method to split sequence into smaller chunks
    # Taken from huggingface: https://huggingface.co/docs/transformers/tasks/language_modeling

    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

#dataset = load_dataset('txt')
#dataset = dataset['train'].train_test_split(test_size=0.2)
#tokenizer = AutoTokenizer.from_pretrained('WizardLM/WizardCoder-3B-V1.0', use_fast=True)
#tokenized_dataset = dataset.map(lambda x: tokenizer(x['text']), remove_columns=dataset['train'].column_names)    # DatasetDict containing train test split, each have attribute input_ids and attention_mask

#print(tokenized_dataset['train']['input_ids'])  # list of list of token id ([1045, 32850, 3280], [34, 280], [], [225], [2166, 279, 519, 225, 37, 44]), len = 123888
#lm_dataset = tokenized_dataset.map(group_texts, batched=True)   # same format as tokenized_dataset, additional
# print(lm_dataset)
# print(lm_dataset["train"]['input_ids'])


#print("-----------")
#dataset = load_dataset("bigcode/the-stack", data_dir="data/g-code", split="train[:1000]")  # to load less examples, use "train[:1000]"
#print(dataset)
#dataset = dataset.train_test_split(test_size=0.2)

# Tokenize data and split into chunks
#tokenizer = AutoTokenizer.from_pretrained('WizardLM/WizardCoder-3B-V1.0', use_fast=True)
#tokenized_dataset = dataset.map(lambda x: tokenizer(x['content']), remove_columns=dataset['train'].column_names)    # remove all columns
#print(tokenized_dataset)    # datasetdict -> train, test -> features (including content)
#print(tokenized_dataset['train'][0])
#print(tokenized_dataset['train']['input_ids'])
#lm_dataset = tokenized_dataset.map(group_texts, batched=True)
#print(np.array(lm_dataset['train']['input_ids']).shape)

model = AutoModelForCausalLM.from_pretrained('WizardLM/WizardCoder-3B-V1.0', device_map="cpu")

print(model)
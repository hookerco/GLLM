from pdfminer.high_level import extract_text
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from glob import glob
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import torch
import os
import math
import argparse


def group_texts(examples, block_size=256):
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


# User picks which dataset to use
parser = argparse.ArgumentParser(description='Pick the dataset that will be used for finetuning')
parser.add_argument("--dataset", type=str, default='pdf', help='Pick either "pdf" or "thestack"')
args = parser.parse_args()

# Check if GPU can be used
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'
#torch.set_default_device(device)
print(f"GPU is available: {cuda_available}")

# todo test different models, adjust lora_modules
#################################### Define parameters
params = SimpleNamespace(
    block_len=128,
    test_split=0.1,

    # directories
    data_directory_pdf='pdfs',
    data_directory_txt='txt',  # destination folder for text files taken from pdfs
    trained_model_dir="finetuned_model",

    # model path from huggingface
    model_path='WizardLM/WizardCoder-3B-V1.0',

    # lora parameters
    lora_r=16,
    # lora modules depend on specific model,
    # see https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models
    lora_modules=["c_proj", "c_attn"],
    lora_alpha=32,
    lora_dropout=0.05,

    # training parameters
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True if cuda_available else False
)

#################################### Loading data
if args.dataset == 'pdf':
    # Convert pdf files (stored in dir specified by params.data_directory_pdf) into txt files (stored in dir
    # specified by params.data_directory_txt), which will be used for finetuning the LLM

    if not os.path.isdir(params.data_directory_txt):
        os.makedirs(params.data_directory_txt)

    # Search recursively files in data directory
    # To test finetuning, we can also only use subset of files
    files = glob(params.data_directory_pdf + '/**/*.pdf', recursive=True)

    # Convert pdf files to txt
    for idx, file in enumerate(files):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):
            # file name without .pdf extension and complete path
            filename_no_ext = os.path.basename(filename)[:-4]
            txt_filepath = os.path.join(params.data_directory_txt, f"{filename_no_ext}.txt")

            # Extract text from pdf file, if not yet done
            if not os.path.exists(txt_filepath):
                print(f"Extracting text from file: {filename_no_ext}, {idx}/{len(files)} done")
                text = extract_text(Path(filename))
                text = text.encode('ascii', errors='ignore').decode()

                # load to txt file
                with open(txt_filepath, "w", encoding="utf-8") \
                        as text_file:
                    text_file.write(text)

            # Need to remove excess spaces in txt file?
            # No, group_texts method concatenates all texts together and then splits the result in small chunks
            #   according to block_size, so removing spaces makes no difference

        else:
            continue
    print("Done extracting text")

    # only train split is available
    dataset = load_dataset(params.data_directory_txt, split="train")
    data_col = 'text'
elif args.dataset == 'thestack':
    # only train split is available
    # Training using the whole dataset (16020 rows) takes 85 hours on the NVIDIA GeForce RTX 4090
    # to load less examples, it is possible to slice the array, e.g. use split="train[:500]" instead of split="train"
    dataset = load_dataset("bigcode/the-stack", data_dir="data/g-code", split="train[:1000]")
    data_col = 'content'
else:
    raise Exception("Invalid dataset choice")

#################################### Finetuning

print("Spliting and tokenizing data")
dataset = dataset.train_test_split(test_size=params.test_split)

# Tokenize data and split into chunks
tokenizer = AutoTokenizer.from_pretrained(params.model_path, use_fast=True)
tokenized_dataset = dataset.map(lambda x: tokenizer(x[data_col]), remove_columns=dataset['train'].column_names)
lm_dataset = tokenized_dataset.map(partial(group_texts, block_size=params.block_len), batched=True)
print("Done splitting and tokenizing data")

# Use LoRA for parameter efficient fine tuning
lora_config = LoraConfig(
    r=params.lora_r,
    target_modules=params.lora_modules,
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=params.lora_alpha,
    lora_dropout=params.lora_dropout
)

# Define model
model = get_peft_model(AutoModelForCausalLM.from_pretrained(params.model_path), lora_config)
model.print_trainable_parameters()

# Build batches, use the inputs as labels shifted to the right by one element:
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Finetune model
training_args = TrainingArguments(
    output_dir=params.trained_model_dir,
    evaluation_strategy=params.evaluation_strategy,
    learning_rate=params.learning_rate,
    weight_decay=params.weight_decay,
    push_to_hub=params.push_to_hub,
    fp16=params.fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

print("Training ...")
trainer.train()
trainer.save_model(params.trained_model_dir)

print(f"Finished training, model stored in {params.trained_model_dir}")

# Evaluate finetuned model
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

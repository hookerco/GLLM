from pdfminer.high_level import extract_text
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from glob import glob
from pathlib import Path
import torch
import os
import math

#### Define parameters

block_len = 1024
test_split = 0.2

# directories
data_directory_pdf = 'pdfs'
data_directory_txt = 'txt'  # destination folder for text files taken from pdfs
trained_model_dir = "finetuned_model"

# model path from huggingface
model_path = 'bigcode/starcoder'

# lora parameters
lora_r = 16
lora_modules = ["q_proj", "v_proj"]  # https://stackoverflow.com/questions/76768226/target-modules-for-applying-peft-lora-on-different-models
lora_alpha = 32
lora_dropout = 0.05

# training parameters
evaluation_strategy = "epoch"
learning_rate = 2e-5
weight_decay = 0.01
push_to_hub = False
fp16 = True

# Check if GPU can be used
cuda_available = torch.cuda.is_available()
device = 'cuda' if cuda_available else 'cpu'
print(f"GPU is available: {cuda_available}")


def group_texts(examples, block_size=block_len):
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


#### Loading data

if not os.path.isdir(data_directory_txt):
    os.makedirs(data_directory_txt)

# Search recursively files in data directory
# To test finetuning, we can also only use subset of files
files = glob(data_directory_pdf + '/**/*.pdf', recursive=True)

# Convert pdf files to txt
for idx, file in enumerate(files):
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        # file name without .pdf extension and complete path
        filename_no_ext = os.path.basename(filename)[:-4]
        txt_filepath = os.path.join(data_directory_txt, f"{filename_no_ext}.txt")

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

#### Finetuning

dataset = load_dataset(data_directory_txt)
dataset = dataset['train'].train_test_split(test_size=test_split)

# Tokenize data and split into chunks
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text']), remove_columns=["text"])
lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# Use LoRA for parameter efficient fine tuning
lora_config = LoraConfig(
    r=lora_r,
    target_modules=lora_modules,
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout
)

# Define model
model = get_peft_model(AutoModelForCausalLM.from_pretrained(model_path), lora_config)
model.print_trainable_parameters()

# Build batches
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Finetune model
training_args = TrainingArguments(
    output_dir=trained_model_dir,
    evaluation_strategy=evaluation_strategy,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    push_to_hub=push_to_hub,
    fp16=fp16
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
trainer.save_model(trained_model_dir)

# Evaluate finetuned model
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

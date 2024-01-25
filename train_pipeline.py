from pdfminer.high_level import extract_text
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from glob import glob
from pathlib import Path
import os
import math

data_directory_pdf = 'pdfs'
data_directory_txt = 'txt'  # destination folder for text files taken from pdfs


def group_texts(examples, block_size=8192):
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
files = glob(data_directory_pdf + '/**/*.pdf', recursive=True) # [:2]   # todo remove [], only for testing

# Convert pdf files to txt
for file in files:  # todo use tqdm?
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        # Extract text from pdf file
        text = extract_text(Path(filename))
        text = text.encode('ascii', errors='ignore').decode()

        # file name without .pdf extension and complete path
        filename_no_ext = os.path.basename(filename)[:-4]

        # load to txt file
        with open(os.path.join(data_directory_txt, f"{filename_no_ext}.txt"), "w", encoding="utf-8") as text_file:
            text_file.write(text)

        # todo remove excess spaces in txt file

    else:
        continue


#### Finetuning

model_path = 'HuggingFaceH4/zephyr-7b-beta'
dataset = load_dataset(data_directory_txt)
dataset = dataset['train'].train_test_split(test_size=0.2)

# Tokenize data and split into chunks
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text']), remove_columns=["text"])
lm_dataset = tokenized_dataset.map(group_texts, batched=True)

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)

# Define model
model = get_peft_model(AutoModelForCausalLM.from_pretrained(model_path), lora_config)
model.print_trainable_parameters()

# Build batches
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Finetune model
training_args = TrainingArguments(
    output_dir="finetuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

# Evaluate
#eval_results = trainer.evaluate()
#print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

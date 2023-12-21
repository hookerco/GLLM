from pdfminer.high_level import extract_text
from datasets import load_dataset
from transformers import AutoTokenizer
import os

data_directory_pdf = 'pdfs'
data_directory_txt = 'txt'


def group_texts(examples, block_size=128):
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


if not os.path.isdir(data_directory_txt):
    os.makedirs(data_directory_txt)

# Convert pdf files to txt
# Iterate over files in data directory
for file in os.listdir(os.fsencode(data_directory_pdf)):  # todo use tqdm?
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        # Extract text from pdf file
        text = extract_text(os.path.join(data_directory_pdf, filename))
        text = text.encode('ascii', errors='ignore').decode()

        filename_no_ext = filename[:-4]
        # load to txt file
        with open(os.path.join(data_directory_txt, f"{filename_no_ext}.txt"), "w", encoding="utf-8") as text_file:
            text_file.write(text)

    else:
        continue

model = 'HuggingFaceH4/zephyr-7b-beta'
dataset = load_dataset(data_directory_txt)
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
tokenized_dataset = dataset.map(lambda example: tokenizer(example['text']), remove_columns=["text"])



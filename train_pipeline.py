from pdfminer.high_level import extract_text
import os

data_directory = 'pdfs'

# Iterate over files in data directory
for file in os.listdir(os.fsencode(data_directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"):
        # Extract text from pdf file
        text = extract_text(os.path.join(data_directory, filename))

        # load into hugging face dataset object?
    else:
        continue


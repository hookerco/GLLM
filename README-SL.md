# GLLM: G-Code generation using open-source LLM models

This repository consists of several python scripts that are at the moment independent of each other, containing implementations
of LLM-related tasks.

### Setup
Create a virtual environment with Python 3.11 and install required packages by running <code>pip install -r requirements.txt</code>

### Chatbot
```chatbot.py``` contains a simple web application to test different LLMs for human-machine interfaces, built using streamlit.

To run this program, follow these steps:
<ol>
  <li> Register or login at <a href="https://huggingface.co">Hugging Face</a> and create an API token in your profile settings </li>
  <li> Add a file called <code>secrets.toml</code> in the folder called <code>.streamlit</code> at the root of your repo, and provide your Hugging Face API token by typing <code>huggingface_token = "..."</code>
  <li> Run the application by running <code>streamlit run chatbot.py</code> in the terminal at the root of the repo. </li> 
</ol>

If you have an OpenAI API key, you can also insert it as in step 3 in the same file, by typing <code>openai_token = "..."</code>

### Question Generation
This file contains code that takes in text and generates question-answer pairs which could be used for LLM evaluation or instruction tuning.

Code was taken from [github](https://github.com/patil-suraj/question_generation).
Check repo for details to setup and run code.


### Finetuning an open-source LLM

```train_pipeline.py``` contains code to finetune open-source LLMs from Hugging Face. 

Run ```python train_pipeline.py``` to start the finetuning process. As default, the dataset used for finetuning are
PDF files stored in the directory ```pdfs```. To use "The Stack", specify this using: ```--dataset 'thestack'```

#### The Stack 
[The Stack](https://huggingface.co/datasets/bigcode/the-stack) contains code files collected from Github, including G-code.
Around 400 MB of G-code is available with a total of 16020 examples.

To use this dataset, you need to log in to Hugging Face in your terminal by:
1. Running ```huggingface-cli login```
2. Providing your Hugging Face access token.

To load this dataset, use ```ds = load_dataset("bigcode/the-stack", data_dir="data/g-code", split="train")```

#### Limitations to Model Size

So far, training is limited to models with <3B parameters due to memory limitations. 
Training code works for these models:
- WizardLM/WizardCoder-3B-V1.0
- bigcode/starcoderbase-3b

I tested [these methods](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#using--accelerate) when training larger models
such as setting smaller batch size, gradient accumulation and checkpointing, mixed precision training, setting device_map='auto'
when loading model, but nothing works so far

#### Pushing Finetuned Model to Hugging Face
To push model to hub after finetuning, make sure you are logged in via cli, just like when using "The Stack" dataset (provide token that has write permission)
#### Starcoder
To use the Starcoder model, you need to be granted access to the model. To do this,
- Log in to Hugging Face in a terminal like described above
- Log in to the Hugging Face website, go to [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
- Accept the conditions to access model files and content.

It is recommended to use the StarCoder tech assistant prompt, since the model is only trained on code completion.

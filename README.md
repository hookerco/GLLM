# GLLM: G-Code generation using open-source LLM models

This repository contains scripts for generating and validating G-codes automatically-generated using various LLM pipelines.

## Setup

### Clone with submodules

```shell
git clone https://github.com/mohamedyd/GLLM.git
```

### Install requirements

This project uses Python3.11. If not installed, you may install it via:  
```shell
sudo apt update
sudo apt install python3.11
```

Then, install poetry and guide it to use python 3.11
```shell
pipx install poetry
poetry env use /usr/bin/python3.11
```

Then, install the requirements
```shell
poetry install
```

To use `Huggingface` models, it is required to save the API access token as an environment variable.

<ol>
  <li> Register or login at <a href="https://huggingface.co">Hugging Face</a> and create an API token in your profile settings </li>
  <li> Add a file called <code>secrets.toml</code> in a folder called <code>.streamlit</code> at the root of the repo, and provide your HuggingFace API token by typing <code>huggingface_token = "..."</code>
  <li> For `OpenAI` models, add the access token <code>openai_token = "YourOpenAITokenHere" </code> to `.streamlit/secrets.toml`. </li>
</ol>

or you can open your shell's configuration file in a text editor: 
```shell
vim ~/.bashrc
```
Add the following line to the end of the file:
```shell
export HUGGINGFACEHUB_API_TOKEN="YourHFTokenHere"
```
Save and close the file. To apply the changes, source the file or restart your terminal:
```shell
source ~/.bashrc
```

## Usage

To run the GLLM application:
```shell
poetry run streamlit run gllm/code_generator_streamlit_reasoning_langchain_langgraph.py
```


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

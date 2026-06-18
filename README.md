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
  <li> To use models via <a href="https://openrouter.ai">OpenRouter</a>, add <code>openrouter_token = "YourOpenRouterTokenHere"</code> to the same file. Optionally specify <code>openrouter_model = "provider/model-name"</code> to choose a model. If no model is specified, the app uses <code>openrouter/free</code>.</li>
</ol>

or you can open your shell's configuration file in a text editor: 
```shell
vim ~/.bashrc
```
Add the following line to the end of the file:
```shell
export HUGGINGFACEHUB_API_TOKEN="YourHFTokenHere"
export OPENROUTER_API_KEY="YourOpenRouterTokenHere" # optional
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

On Windows, OpenRouter is the default model in the app and CLI. For Codex or
agent-based browser testing, prefer the detached launcher so the terminal does
not wait on the long-running Streamlit server:

```powershell
.\scripts\start_streamlit_detached.ps1
```

The script reuses an existing listener on port `8501` when one exists. If no
listener exists, it starts a hidden worker and returns immediately; the worker
starts Streamlit, waits for `http://localhost:8501`, and records logs and PID
details in `.codex-log/`.

To use the CLI with the same default model:

```powershell
poetry run python -m gllm.cli --prompt-type Unstructured
```

To build a prompt-to-verdict evidence packet from an existing generated program:

```powershell
poetry run python -m gllm.proof.cli --registry config/vericut_setups.example.json --setup-id vericut96_haas_minimill_sample --prompt "Mill a simple square pocket." --candidate-gcode-file path\to\generated.nc --output-root .proof-runs
```

The proof-run CLI writes `evidence_packet.json` and `evidence_packet.md` with
the prompt, setup ID, candidate G-code, static-check findings, staged Vericut
job, optional Vericut verdict, operator action, and repair context. See
`docs/prompt_to_verdict.md`.

To run the checked-in prompt corpus against the sample setups:

```powershell
poetry run python -m gllm.proof.corpus_cli --corpus config/proof_prompt_corpus.example.json --registry config/vericut_setups.example.json --output-root .proof-runs\corpus-smoke
```

The corpus includes a live-control Haas MiniMill fixture, smaller passing and
intentionally rejected MiniMill fixtures, and a second Haas VF3 sample setup. It
is the regression target for proving that prompt-to-verdict behavior is
scenario-based rather than a single hand-tuned prompt.

### Vericut Integration

This repo now includes a local Vericut staging CLI for generated G-code. It loads a setup registry, runs conservative static checks, copies referenced local Vericut assets into an ignored job folder, and prints the `vericut.bat BATCH ...` command.

```powershell
poetry run python -m gllm.vericut.cli --registry config/vericut_setups.example.json --setup-id vericut96_haas_minimill_sample --gcode-file path\to\generated.nc --output-root .vericut-runs
```

See `docs/vericut_integration.md` before adding proprietary machine setups.

When run with `--run-vericut`, the CLI writes `output/verdict.json` and
`output/verdict.md`. The verdict parser reads Vericut's log, so a Vericut
process return code of `0` is still rejected when the log reports toolpath
errors.


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

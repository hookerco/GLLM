Python version??

### Chatbot
```chatbot.py``` contains a simple web application to test different LLMs for human-machine interfaces, built using streamlit.

To run this program, follow these steps:
<ol>
  <li> Create a virtual environment and install required 
    packages by running <code>pip install -r requirements.txt</code>
  </li>
  <li> Register at <a href="https://huggingface.co">Hugging Face</a> and get an API token in your profile settings </li>
  <li> Add a file called <code>secrets.toml</code> in the folder called <code>.streamlit</code> at the root of your repo, and provide your Hugging Face API token by typing <code>huggingface_token = "..."</code>
  <li> Run the application by running <code>streamlit run chatbot.py</code> in the terminal at the root of the repo. </li> 
</ol>

If you have an OpenAI API key, you can also insert it as in step 3 in the same file, by typing <code>openai_token = "..."</code>

### Finetuning an open-source LLM

```train_pipeline.py``` contains code to finetune open-source LLMs from Hugging Face. 

run ```python train_pipeline.py``` to start the finetuning process.
```--dataset 'thestack'```

#### The Stack 
[The Stack](https://huggingface.co/datasets/bigcode/the-stack) contains code files collected from Github, including G-code.
Around 400 MB of G-code is available with a total of 16020 examples.

To use this dataset, you need to log in to Hugging Face in your terminal by:
1. Running ```huggingface-cli login```
2. Providing your Hugging Face access token.

To load this dataset, use ```ds = load_dataset("bigcode/the-stack", data_dir="data/g-code", split="train")```



#### Starcoder
To use the Starcoder model, you need to be granted access to the model. To do this,
- Log in to Hugging Face in a terminal like described above
- Log in to the Hugging Face website, go to [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
- Accept the conditions to access model files and content.
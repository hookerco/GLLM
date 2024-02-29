## Chatbot
A simple web application to test different LLMs for human-machine interfaces, built using streamlit.

### Getting Started

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

To use "the stack" dataset, you need to log in to huggingface by running "huggingface-cli login" in the terminal
and providing your huggingface access token.
Around 400 mb of g-code is available with a total of 16020 examples

```ds = load_dataset("bigcode/the-stack", data_dir="data/g-code", split="train")```

To use starcoder, you need to be granted access to the model. To do this,
- Log in to huggingface
- go to https://huggingface.co/bigcode/starcoder
- accept the conditions to access model files and content.
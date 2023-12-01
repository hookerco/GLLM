## Chatbot
A simple web application to test different LLMs, built using streamlit.

### Getting Started

To run this program, follow these steps:
<ol>
  <li> Create a virtual environment and install required 
    packages by running <code>pip install -r requirements.txt</code>
  </li>
  <li> Register at <a href="https://huggingface.co">Hugging Face</a> and get an API token in your profile settings </li>
  <li> add a file called <code>secrets.toml</code> in the folder called <code>.streamlit</code> at the root of your repo, and provide your Hugging Face API token by typing <code>huggingface_token = "..."</code>
  <li> Run the application by running <code>streamlit run chatbot.py</code> in the terminal at the root of the repo. </li> 
</ol>
import os
import toml
import streamlit as st
import openai

from utils.rag_utils import setup_langchain_with_rag
from utils.model_utils import setup_model, setup_langchain_without_rag
from utils.params_extraction_utils import extract_parameters_logic, display_extracted_parameters
from utils.gcode_utils import display_generated_gcode, generate_gcode_logic, plot_generated_gcode


# Define the path to the secrets.toml file
secrets_file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.streamlit', 'secrets.toml'))

# Load the secrets
secrets = toml.load(secrets_file_path)

# Set your OpenAI API key
openai.api_key = secrets["openai_token"]
HF_TOKEN = os.getenv(secrets["huggingface_token"])


def main():
    st.title("G-code Generator for CNC Machines")
    st.write("Please describe your CNC machining task in natural language:")
    task_description = st.text_area("Task Description", height=150)

    # Drop-down menu for model selection
    model_str = st.selectbox('Choose a Language Model:', ('StarCoder', 'GPT-3.5', 'Fine-tuned StarCoder'), index=1)
    model = setup_model(model=model_str)
    is_openai = True if model_str == "GPT-3.5" else False
    
    pdf_files = st.file_uploader("Upload PDF files with additional knowledge (RAG)", accept_multiple_files=True, type=['pdf'])

    if "langchain_chain" not in st.session_state:
        if pdf_files:
            st.session_state['langchain_chain'] = setup_langchain_with_rag(pdf_files, model)
        else:
            st.session_state['langchain_chain'] = setup_langchain_without_rag(is_openai=is_openai, model=model)
        
    if "extracted_parameters" not in st.session_state:
        st.session_state['extracted_parameters'] = None
        st.session_state['missing_parameters'] = None
        st.session_state['user_inputs'] = {}
        st.session_state['gcode'] = None

    if st.button("Extract Parameters") and "langchain_chain" in st.session_state:
        extract_parameters_logic(st.session_state['langchain_chain'], task_description)

    display_extracted_parameters()

    if st.button("Generate G-code") and "langchain_chain" in st.session_state:
        generate_gcode_logic(st.session_state['langchain_chain'])

    display_generated_gcode()

    plot_generated_gcode()

if __name__ == "__main__":
    main()
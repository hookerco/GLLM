import uuid
import streamlit as st
from utils.rag_utils import setup_langchain_with_rag
from utils.model_utils import setup_model, setup_langchain_without_rag
from utils.params_extraction_utils import extract_parameters_logic, display_extracted_parameters
from utils.gcode_utils import display_generated_gcode, generate_gcode_logic, plot_generated_gcode, validate_gcode, clean_gcode
from utils.graph_utils import construct_graph, _print_event


def main():

    _printed = set()
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,},
            "recursion_limit": 1000}

    st.title("G-code Generator for CNC Machines")
    st.write("Please describe your CNC machining task in natural language:")
    task_description = st.text_area("Task Description", height=150)

    # Drop-down menu for model selection
    model_str = st.selectbox('Choose a Language Model:', ('Zephyr-7b', 'GPT-3.5', 'Fine-tuned StarCoder'), index=1)
    model = setup_model(model=model_str)
    
    pdf_files = st.file_uploader("Upload PDF files with additional knowledge (RAG)", accept_multiple_files=True, type=['pdf'])

    if "langchain_chain" not in st.session_state:
        if pdf_files:
            st.session_state['langchain_chain'] = setup_langchain_with_rag(pdf_files, model)
        else:
            st.session_state['langchain_chain'] = setup_langchain_without_rag(model=model)
        
    if "extracted_parameters" not in st.session_state:
        st.session_state['extracted_parameters'] = None
        st.session_state['missing_parameters'] = None
        st.session_state['user_inputs'] = {}
        st.session_state['gcode'] = None

    if st.button("Extract Parameters") and "langchain_chain" in st.session_state:
        extract_parameters_logic(st.session_state['langchain_chain'], task_description)

    display_extracted_parameters()

    if st.button("Generate G-code") and "langchain_chain" in st.session_state:
        #generate_gcode_logic(st.session_state['langchain_chain'])

        graph = construct_graph(st.session_state['langchain_chain'], st.session_state['user_inputs'])
        events = graph.stream({"messages": [("user", task_description)], "iterations": 0}, config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)

        #cleaned_gcode = clean_gcode(event['generation'])
        st.session_state['gcode'] = event['generation']

        # while not validate_gcode(st.session_state['gcode']):
        #     generate_gcode_logic(st.session_state['langchain_chain'])
    
    display_generated_gcode()

    plot_generated_gcode()

if __name__ == "__main__":
    main()
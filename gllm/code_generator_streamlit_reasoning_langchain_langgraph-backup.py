import uuid
import streamlit as st
from gllm.utils.rag_utils import setup_langchain_with_rag
from gllm.utils.model_utils import setup_model, setup_langchain_without_rag
from gllm.utils.params_extraction_utils import extract_parameters_logic, display_extracted_parameters, parse_extracted_parameters, extract_numerical_values
from gllm.utils.gcode_utils import display_generated_gcode, generate_gcode_logic, plot_generated_gcode, validate_gcode, clean_gcode, generate_gcode_unstructured_prompt, generate_task_descriptions
from gllm.utils.graph_utils import construct_graph, _print_event
from gllm.utils.plot_utils import plot_user_specification
import plotly.express as px  # Import Plotly Express


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

    # Let the user choose whether to use structured or unstructured prompt
    prompt_type = st.selectbox('Prompt Type:', ('Structured', 'Unstructured'), index=0)
    
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
        
    if "parsed_parameters" not in st.session_state:
        st.session_state.parsed_parameters = {} 

    #################################################
    ############# Parameters Extraction #############
    #################################################

    disable_extract_button = False if prompt_type == 'Structured' else True    # Disable Parameter Extraction if user selects unstructured prompt 
    extract_button = st.button("Extract Parameters", disabled=disable_extract_button)   
    if extract_button and "langchain_chain" in st.session_state:
        extract_parameters_logic(st.session_state['langchain_chain'], task_description)

        # generate subtask descriptions if the input task invovles more than one shape
        if extract_numerical_values(st.session_state['user_inputs'], 'Number of Shapes')[0] > 1:
            task_descriptions = generate_task_descriptions(model, task_description)
            st.session_state['extracted_parameters'] += f"Subtasks: {task_descriptions}\n"
        
    if st.session_state['extracted_parameters']:
        display_extracted_parameters()

    if st.button("Simulate the tool path (2D)", disabled=disable_extract_button):
        if st.session_state['extracted_parameters']:
            st.session_state['parsed_parameters'] = parse_extracted_parameters(st.session_state['extracted_parameters'])
            st.text("If the plotted path is incorrect, please adjust the task description.")
            st.pyplot(plot_user_specification(parsed_parameters=st.session_state.parsed_parameters)) 

    ################################################
    ############ G-Code Generation #################
    ################################################

    if st.button("Generate G-code"):
        if disable_extract_button:
            st.session_state['gcode'] = generate_gcode_unstructured_prompt(st.session_state['langchain_chain'], task_description)
        else: 
            if "langchain_chain" in st.session_state and 'parsed_parameters' in st.session_state:
                # construct graph
                graph = construct_graph(st.session_state['langchain_chain'], st.session_state['user_inputs'], st.session_state['extracted_parameters'])
                events = graph.stream({"messages": [("user", task_description)], "iterations": 0}, config, stream_mode="values")
                for event in events:
                    _print_event(event, _printed)

                #cleaned_gcode = clean_gcode(event['generation'])
                st.session_state['gcode'] = event['generation']

    
    display_generated_gcode()

    plot_generated_gcode()

     # Debug information
    if st.checkbox("Show Debug Info"):
        st.write("Session State:", st.session_state)

if __name__ == "__main__":
    main()
"""
Description of this file:

This is a Streamlit application that uses LLM pipelines with Langchain and Langgraph to generate G-codes for CNC machines. 
The application takes a natural language instruction as input and generates a G-code based on the instruction. 
The G-code is then validated and can be downloaded or visualized as a 3D plot.

The application is written in Python and uses the Streamlit library for the user interface. 
It also uses the Langchain and Langgraph libraries for the LLM pipelines.

Authors: Mohamed Abdelaal, Samuel Lokadjaja

This work was done at Software AG, Darmstadt, Germany in 2023-2024 and is published under the Apache License 2.0.
"""


import uuid
import streamlit as st
from gllm.utils.rag_utils import setup_langchain_with_rag
from gllm.utils.model_utils import setup_model, setup_langchain_without_rag
from gllm.utils.params_extraction_utils import extract_parameters_logic, display_extracted_parameters, parse_extracted_parameters, extract_numerical_values
from gllm.utils.gcode_utils import display_generated_gcode, generate_gcode_logic, plot_generated_gcode, validate_gcode, clean_gcode, generate_gcode_unstructured_prompt, generate_task_descriptions
from gllm.utils.graph_utils import construct_graph, _print_event
from gllm.utils.plot_utils import plot_user_specification, refine_gcode
import plotly.express as px  # Import Plotly Express
from gllm.utils.params_extraction_utils import from_dict_to_text
    

def extract_parameters(description_text):
        
        extracted_parameters, missing_parameters = extract_parameters_logic(st.session_state['langchain_chain'], description_text)
        # update the relevant Streamlit states
        st.session_state['extracted_parameters'] = from_dict_to_text(extracted_parameters)
        st.session_state['missing_parameters'] = missing_parameters
        st.session_state['user_inputs'].update(extracted_parameters)


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
    input_description = st.text_area("Task Description", height=150)
    

    # Drop-down menu for model selection
    model_str = st.selectbox(
        'Choose a Language Model:',
        ('Zephyr-7b', 'GPT-3.5', 'Fine-tuned StarCoder', 'CodeLlama'),
        index=1,
    )

    # Store selected model in the session state and rebuild the chain if needed
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = model_str
    elif st.session_state["selected_model"] != model_str:
        st.session_state["selected_model"] = model_str
        if "langchain_chain" in st.session_state:
            del st.session_state["langchain_chain"]

    model = setup_model(model=st.session_state["selected_model"])

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
        st.session_state['task_descriptions'] = []
        st.session_state['decompose_task'] = None
        st.session_state['extracted_parameters_backup'] = None
        st.session_state['user_inputs_backup'] = {}
        
    if "parsed_parameters" not in st.session_state:
        st.session_state.parsed_parameters = {} 

    #################################################
    ############# Parameters Extraction #############
    #################################################

 

    disable_extract_button = False if prompt_type == 'Structured' else True    # Disable Parameter Extraction if user selects unstructured prompt 

    # user selects whether to use the task decomposor 
    st.session_state['decompose_task'] = st.selectbox("Decompose The task Description: ", ('Yes', 'No'), index=0, disabled=disable_extract_button)  

    extract_button = st.button("Extract Parameters", disabled=disable_extract_button)   
    if extract_button and "langchain_chain" in st.session_state:
        extract_parameters(description_text=input_description)
        st.session_state['extracted_parameters_backup'] = st.session_state['extracted_parameters']
        st.session_state['user_inputs_backup'] = st.session_state['user_inputs']

        # generate subtask descriptions if the input task invovles more than one shape
        values_in_number_shapes = extract_numerical_values(st.session_state['user_inputs'], 'Number of Shapes')
        number_shapes = values_in_number_shapes[0] if type(values_in_number_shapes) is list else values_in_number_shapes
        
        if number_shapes > 1 and st.session_state['decompose_task'] == 'Yes':
            st.session_state['task_descriptions'] = generate_task_descriptions(model, model_str, input_description)
            st.session_state['extracted_parameters'] += f"Subtasks: {st.session_state['task_descriptions']}\n"
        else:
            st.session_state['task_descriptions'] = [input_description]
        
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

        gcodes_combined = ""
        
        if not st.session_state['task_descriptions']:
            st.session_state['task_descriptions'] = [input_description]

        for subtask_description in st.session_state['task_descriptions']:
            print("++++++++++++++++++++++++++++++++++++++++++")
            print("SUBTASK DESCRIPTION:", subtask_description)
            print("++++++++++++++++++++++++++++++++++++++++++")
            if disable_extract_button:
                st.session_state['gcode'] = generate_gcode_unstructured_prompt(st.session_state['langchain_chain'], subtask_description)
            else:
                
                if len(st.session_state['task_descriptions']) > 1:
                    extract_parameters(description_text=subtask_description)

                if "langchain_chain" in st.session_state and 'parsed_parameters' in st.session_state:
                    # construct graph
                    graph = construct_graph(st.session_state['langchain_chain'], st.session_state['user_inputs'], st.session_state['extracted_parameters'])
                    events = graph.stream({"messages": [("user", subtask_description)], "iterations": 0}, config, stream_mode="values")
                    for event in events:
                       pass 
                        #_print_event(event, _printed)

                    gcodes_combined += f"\n{event['generation']}"
                    gcodes_combined = refine_gcode(gcodes_combined) 
        
                    st.session_state['gcode'] = gcodes_combined

        # restore the extracted parameters from the input task description
        st.session_state['user_inputs'] = st.session_state['user_inputs_backup']
        st.session_state['extracted_parameters'] = st.session_state['extracted_parameters_backup']

    display_generated_gcode()

    plot_generated_gcode()

     # Debug information
    if st.checkbox("Show Debug Info"):
        st.write("Session State:", st.session_state)

if __name__ == "__main__":
    main()
import streamlit as st
from utils.prompts_utils import REQUIRED_PARAMETERS


def extract_parameters_logic(chain, task_description):
    extracted_parameters_text = extract_parameters_with_langchain(chain, task_description)
    print(extracted_parameters_text) 
    #st.session_state['extracted_parameters'] = extracted_parameters_text['answer']
    st.session_state['extracted_parameters'] = extracted_parameters_text.content

    extracted_parameters = {}
    for line in extracted_parameters_text.content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            extracted_parameters[key.strip()] = value.strip()

    missing_parameters = [param for param in REQUIRED_PARAMETERS if param not in extracted_parameters]
    st.session_state['missing_parameters'] = missing_parameters
    st.session_state['user_inputs'].update(extracted_parameters)


def extract_parameters_with_langchain(chain, task_description):
    prompt = (
        "Extract the following details from the given CNC machining task description where the extracted details will later be converted into a dictionary:\n"
        "Type of CNC Machine\n"
        "Material\n"
        "Tool Type\n"
        "Tool Diameter\n"
        "Operation Type\n"
        "Dimensions\n"
        "Coordinates\n"
        "Depth of Cut\n"
        "Feed Rate\n"
        "Spindle Speed\n"
        "Coolant type\n"
        "Tool or work offsets\n"
        "Safety instructions\n"
        "\nTask description: {}\n\nExtracted parameters:".format(task_description)
    )
    response = chain.invoke({'input':prompt})
    return response


def display_extracted_parameters():
    if st.session_state['extracted_parameters']:
        st.subheader("Extracted Parameters")
        st.text_area("Extracted Parameters:", st.session_state['extracted_parameters'], height=300)

        if st.session_state['missing_parameters']:
            st.subheader("Missing Required Parameters")
            for param in st.session_state['missing_parameters']:
                st.session_state['user_inputs'][param] = st.text_input(f"Please provide the {param}")


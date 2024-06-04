import re
import streamlit as st
from utils.prompts_utils import REQUIRED_PARAMETERS


def extract_parameters_logic(chain, task_description):
    extracted_parameters_text = extract_parameters_with_langchain(chain, task_description)

    # convert the extracted parameters from string into a dictionary 
    extracted_parameters = {}
    print(extracted_parameters_text.content)
    for line in extracted_parameters_text.content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().lower()  # Normalize value to lowercase

            # Use a regular expression to check for any form of "not specified"
            if not re.match(r"not specified", value):
                extracted_parameters[key.strip()] = value.strip()
    
    st.session_state['extracted_parameters'] = from_dict_to_text(extracted_parameters)
    # find the required parameters which have not been assigned a value
    missing_parameters = [param for param in REQUIRED_PARAMETERS if param not in extracted_parameters]
    print(missing_parameters)
    # update the relevant Streamlit states
    st.session_state['missing_parameters'] = missing_parameters
    st.session_state['user_inputs'].update(extracted_parameters)


def extract_parameters_with_langchain(chain, task_description):
    prompt = (
        "Extract the following details from the given CNC machining task description. Each detail should be followed by its value, or 'Not specified' if the detail is missing from the description. Ensure the extracted details will later be converted into a dictionary:\n"
        "\n"
        "Type of CNC Machine: \n"
        "Material: \n"
        "Tool Type: \n"
        "Tool Diameter: \n"
        "Operation Type: \n"
        "Desired Shape: \n"
        "Shape Dimensions: \n"
        "Home Position: \n"
        "Workpiece Dimensions: \n"
        "Coordinates: \n"
        "Depth of Cut: \n"
        "Feed Rate: \n"
        "Spindle Speed: \n"
        "Coolant Type: \n"
        "Tool or Work Offsets: \n"
        "Safety Instructions: \n"
        "\n"
        "Task description: {}\n\nExtracted parameters:".format(task_description)
    )
    response = chain.invoke({'input':prompt})
    return response


def display_extracted_parameters():
    if st.session_state['extracted_parameters']:
        st.subheader("Extracted Parameters")
        st.text_area("Extracted Parameters:", st.session_state['extracted_parameters'], height=300)

        if st.session_state['missing_parameters']:
            st.subheader("Missing Parameters [Optional]")
            for param in st.session_state['missing_parameters']:
                st.session_state['user_inputs'][param] = st.text_input(f"Please provide the {param}")


def from_dict_to_text(input:dict):
    # Create a text string with each key-value pair on a new line
    output_text = ""
    for key, value in input.items():
        output_text += f"{key}: {value}\n"

    return output_text

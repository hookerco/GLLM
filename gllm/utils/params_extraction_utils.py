import re
import streamlit as st
from gllm.utils.prompts_utils import REQUIRED_PARAMETERS
from gllm.utils.plot_utils import plot_user_specification


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
        "Extract the following details from the given CNC machining task description. Each detail should be followed by its value, or 'Not specified' if the detail is missing from the description. Ensure the extracted details will later be converted into a dictionary. Make sure to extract/infer the cutting tool path (x, y, z) from the task description.\n"
        "\n"
        "Material: \n"
        "Operation Type: \n"
        "Desired Shape: \n"
        "Workpiece Dimensions: \n"
        "Starting Point: \n"
        "Home Position: \n"
        "Cutting Tool Path: \n"
        "Return Tool to Home After Execution: \n"
        "Workpiece Dimensions: \n"
        "Depth of Cut: \n"
        "Feed Rate: \n"
        "Spindle Speed: \n"
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
            st.subheader("Missing Parameters")
            st.text("Rerun 'Parameter Extraction' if below parameters already in Task Description")
            for param in st.session_state['missing_parameters']:
                st.session_state['user_inputs'][param] = st.text_input(f"Please provide the {param}")


def from_dict_to_text(input:dict):
    # Create a text string with each key-value pair on a new line
    output_text = ""
    for key, value in input.items():
        output_text += f"{key}: {value}\n"

    return output_text


def extract_numerical_values(parameters: dict, key: str):
    if key in parameters:
        # Regular expression to match numerical values
        numbers = re.findall(r'\d+', parameters[key])
        # Convert the extracted numbers to float
        numbers = list(map(float, numbers))
    else:
        numbers = 0 

    return numbers


def extract_path(parameters: dict, key: str):
    # Regular expression to find coordinates
    pattern = r'(?:x=)?(-?\d+\.?\d*)\s*,\s*(?:y=)?(-?\d+\.?\d*)\s*(?:,\s*(?:z=)?(-?\d+\.?\d*))?'

    # Find all matches
    matches = re.findall(pattern, parameters[key])

    # Convert matches to tuples of floats and set z to 0 if not given
    coordinates = [(float(x), float(y), float(z) if z else 0.0) for x, y, z in matches]

    return coordinates


def parse_extracted_parameters(parameter_string):
    parameters = {}
    current_key = None
    parsed_parameters = {}
    for line in parameter_string.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in REQUIRED_PARAMETERS:
                parameters[key.strip()] = value.strip()
                current_key = key.strip()
            else:
                # Handle cases where the key is missing
                if current_key is not None:
                    parameters[current_key] += " " + line  # Append to previous line
                else:
                    # Handle the case where there's no previous key (shouldn't happen ideally)
                    print(f"Warning: Line without a valid key: {line}")

    parsed_parameters['workpiece_diemensions'] = extract_numerical_values(parameters=parameters, key='Workpiece Dimensions') 
    parsed_parameters['starting_point'] = extract_numerical_values(parameters=parameters, key='Starting Point') 
    parsed_parameters['home_position'] = extract_numerical_values(parameters=parameters, key='Home Position') 
    parsed_parameters['tool_path'] = extract_path(parameters=parameters, key='Cutting Tool Path')
    parsed_parameters['cut_depth'] = extract_numerical_values(parameters=parameters, key='Depth of Cut') 

    return parsed_parameters


def validate_parameters_extraction():

    if st.button("Simulate the tool path (2D)"):
        parsed_parameters = parse_extracted_parameters(st.session_state['extracted_parameters'])
        st.pyplot(plot_user_specification(parsed_parameters=parsed_parameters))

        # Ask the user for confirmation
    if st.session_state['user_confirmation'] is None:
        st.session_state['user_confirmation'] = st.radio(
            "Does the plotted path accurately represent the desired shape?",
            ["Yes", "No"], index=None)

    else:
        if st.session_state['user_confirmation'] == "No":
            st.warning("Please adjust the task description to correct the path.")
        elif st.session_state['user_confirmation'] == "Yes":
            st.success("Great! Let's proceed.")
import re
import math
import streamlit as st
from gllm.utils.prompts_utils import REQUIRED_PARAMETERS



def extract_parameters_logic(chain, task_description):
    extracted_parameters_text = extract_parameters_with_langchain(chain, task_description)

    # convert the extracted parameters from string into a dictionary 
    extracted_parameters = {}
    cutting_tool_path = []
    print(extracted_parameters_text.content)
    for line in extracted_parameters_text.content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().lower()  # Normalize value to lowercase

            # Use a regular expression to check for any form of "not specified"
            if not re.match(r"not specified", value):
                extracted_parameters[key.strip()] = value.strip()
        else:
            # capture x and y values of the cutting tool path
            match = re.search(r'(?:[xX]\s*=\s*([\d.]+)\s*[,;\s]\s*[yY]\s*=\s*([\d.]+))|(?:\(([\d.]+),\s*([\d.]+)\))', line)
            print("MATCH: ", match)
            if match:
                x, y = match.groups()[0:2] if match.groups()[0] else match.groups()[2:4]
                print("X, Y: ", x, y)
                cutting_tool_path.append((float(x), float(y)))

    # Add the cutting tool path to the extracted parameters
    if cutting_tool_path:
        extracted_parameters['Cutting Tool Path'] = cutting_tool_path
    
    
     
    # find the required parameters which have not been assigned a value
    missing_parameters = [param for param in REQUIRED_PARAMETERS if param not in extracted_parameters]
    


    return extracted_parameters, missing_parameters


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
        "Depth of Cut: \n"
        "Feed Rate: \n"
        "Spindle Speed: \n"
        "Radius: \n"
        "Number of Shapes: \n"
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
                st.session_state['user_inputs'][param] = st.text_input(f"Please provide the {param}", key=f"text_input_{param}")  # Unique key for each text input
                if st.session_state['user_inputs'][param]:
                    print("PARAM: ", param)
                    st.session_state['extracted_parameters'] += f"{param}: {st.session_state['user_inputs'][param]}\n" 
                    st.session_state['missing_parameters'].remove(param)    # remove the entry from the list, if the user added it.


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

def find_circular_path(parameters: dict):
    num_points = 100
    radius = extract_numerical_values(parameters=parameters, key='Radius')[0]
    starting_point = extract_numerical_values(parameters=parameters, key='Starting Point')
    center_x = starting_point[0] + radius
    center_y = starting_point[1]
    path = []
    print("CENTER_X: ", center_x)
    print("CENTER_Y: ", center_y)
    print("RADIUS: ", radius)
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle) 
        path.append((x,y, 0))
    
    return path

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
    parsed_parameters['tool_path'] = find_circular_path(parameters=parameters) if parameters["Desired Shape"] in ['circle', 'circular pocket', 'circular', 'Circular', 'Circular Pocket', 'Circle'] else extract_path(parameters=parameters, key='Cutting Tool Path')
    parsed_parameters['cut_depth'] = extract_numerical_values(parameters=parameters, key='Depth of Cut') 

    return parsed_parameters

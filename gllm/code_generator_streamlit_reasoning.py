import os
import re
import toml
import streamlit as st
import openai

# Define the path to the secrets.toml file
secrets_file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.streamlit', 'secrets.toml'))

# Load the secrets
secrets = toml.load(secrets_file_path)

# Set your OpenAI API key
openai.api_key = secrets["openai_token"]

REQUIRED_PARAMETERS = [
    "Type of CNC Machine",
    "Material",
    "Tool Type",
    "Tool Diameter",
    "Operation Type",
    "Dimensions",
    "Coordinates",
    "Depth of Cut",
    "Feed Rate",
    "Spindle Speed"
]

def clean_gcode(gcode):
    # Regular expression for matching valid G-code commands
    gcode_pattern = re.compile(r"^(?:G|M|T|F|S|X|Y|Z|I|J|K|R|P|Q)\d+.*")

    cleaned_lines = []
    for line in gcode.split('\n'):
        if gcode_pattern.match(line):
            cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)

def get_response(prompt, max_tokens=150):

    # Create a client instance
    client = openai.OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

def extract_parameters(task_description):
    prompt = (
        "Extract the following details from the given CNC machining task description where the extraced details will later be converted into a dictionary:\n"
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
    response = get_response(prompt, max_tokens=300)
    return response

def generate_gcode(user_inputs):
    final_prompt = (
        "Based on the details provided, generate a robust G-code for the CNC machining operation:\n\n"
        f"Type of CNC Machine: {user_inputs['Type of CNC Machine']}\n"
        f"Material: {user_inputs['Material']}\n"
        f"Tool Information:\n"
        f"Tool Type: {user_inputs['Tool Type']}\n"
        f"Tool Diameter: {user_inputs['Tool Diameter']}\n"
        f"Operation Details:\n"
        f"Operation Type: {user_inputs['Operation Type']}\n"
        f"Dimensions: {user_inputs['Dimensions']}\n"
        f"Coordinates: {user_inputs['Coordinates']}\n"
        f"Depth of Cut: {user_inputs['Depth of Cut']}\n"
        f"Feed Rate: {user_inputs['Feed Rate']}\n"
        f"Spindle Speed: {user_inputs['Spindle Speed']}\n"
        f"Additional Parameters:\n"
        f"Coolant type: {user_inputs['Coolant type']}\n"
        f"Tool or work offsets: {user_inputs['Tool or work offsets']}\n"
        f"Safety instructions: {user_inputs['Safety instructions']}\n"
    )
    gcode_response = get_response(final_prompt)
    return gcode_response

    
def extract_parameters_logic(task_description):
    extracted_parameters_text = extract_parameters(task_description)
    st.session_state['extracted_parameters'] = extracted_parameters_text

    # Parse the extracted parameters into a dictionary
    extracted_parameters = {}
    for line in extracted_parameters_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            extracted_parameters[key.strip()] = value.strip()

    # Identify missing required parameters
    missing_parameters = [param for param in REQUIRED_PARAMETERS if param not in extracted_parameters]
    st.session_state['missing_parameters'] = missing_parameters
    st.session_state['user_inputs'].update(extracted_parameters)

def display_extracted_parameters():
    if st.session_state['extracted_parameters']:
        st.subheader("Extracted Parameters")
        st.text_area("Extracted Parameters:", st.session_state['extracted_parameters'], height=300)

        if st.session_state['missing_parameters']:
            st.subheader("Missing Required Parameters")
            for param in st.session_state['missing_parameters']:
                st.session_state['user_inputs'][param] = st.text_input(f"Please provide the {param}")

def generate_gcode_logic():
    if any(param not in st.session_state['user_inputs'] for param in REQUIRED_PARAMETERS):
        st.error("Please provide all the required parameters.")
    else:
        gcode = generate_gcode(st.session_state['user_inputs'])

        cleaned_gcode = clean_gcode(gcode)

        st.session_state['gcode'] = cleaned_gcode

def display_generated_gcode():
    if st.session_state['gcode']:
        st.subheader("Generated G-code")
        st.text_area("G-code:", st.session_state['gcode'], height=300)
        st.download_button(
            label="Download G-code",
            data=st.session_state['gcode'],
            file_name="generated.gcode",
            mime="text/plain")

def main():
    st.title("G-code Generator for CNC Machines")
    st.write("Please describe your CNC machining task in natural language:")
    task_description = st.text_area("Task Description", height=150)

    # Initialize session state variables if not already done
    if "extracted_parameters" not in st.session_state:
        st.session_state['extracted_parameters'] = None
        st.session_state['missing_parameters'] = None
        st.session_state['user_inputs'] = {}
        st.session_state['gcode'] = None

    # Button to extract parameters from task description
    if st.button("Extract Parameters"):
        extract_parameters_logic(task_description)

    # Display results from parameter extraction
    display_extracted_parameters()

    # Button to generate G-code
    if st.button("Generate G-code"):
        generate_gcode_logic()

    # Optionally display generated G-code
    display_generated_gcode()


 
if __name__ == "__main__":
    main()
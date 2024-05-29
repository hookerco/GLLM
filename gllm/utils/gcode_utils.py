import re
import streamlit as st
from utils.plot_utils import plot_gcode
from utils.prompts_utils import REQUIRED_PARAMETERS



def generate_gcode_logic(chain):
    if any(param not in st.session_state['user_inputs'] for param in REQUIRED_PARAMETERS):
        print(st.session_state['user_inputs'])
        st.error("Please provide all the required parameters.")
    else:
        gcode = generate_gcode_with_langchain(chain, st.session_state['user_inputs'])
        cleaned_gcode = clean_gcode(gcode)
        st.session_state['gcode'] = cleaned_gcode


def generate_gcode_with_langchain(chain, user_inputs):
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
    gcode_response = chain.invoke({'input':final_prompt})
    return gcode_response


def clean_gcode(gcode):
    gcode_pattern = re.compile(r"^(?:G|M|T|F|S|X|Y|Z|I|J|K|R|P|Q)\d+.*")
    cleaned_lines = [line.strip() for line in gcode.content.split('\n') if gcode_pattern.match(line)]
    return '\n'.join(cleaned_lines)


def display_generated_gcode():
    if st.session_state['gcode']:
        st.subheader("Generated G-code")
        st.text_area("G-code:", st.session_state['gcode'], height=300)
        st.download_button(
            label="Download G-code",
            data=st.session_state['gcode'],
            file_name="generated.gcode",
            mime="text/plain")


def plot_generated_gcode():
    if st.button("Plot G-code"):
        plt = plot_gcode(st.session_state['gcode'])
        st.pyplot(plt) 
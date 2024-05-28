import re
import os
import streamlit as st
import openai
import toml

# Define the path to the secrets.toml file
secrets_file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.streamlit', 'secrets.toml'))

# Load the secrets
secrets = toml.load(secrets_file_path)

# Set your OpenAI API key
openai.api_key = secrets["openai_token"]

def get_response(prompt):

    # Create a client instance
    client = openai.OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def clean_gcode(gcode):
    # Regular expression for matching valid G-code commands
    gcode_pattern = re.compile(r"^(?:G|M|T|F|S|X|Y|Z|I|J|K|R|P|Q)\d+.*")

    cleaned_lines = []
    for line in gcode.split('\n'):
        if gcode_pattern.match(line):
            cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)


def generate_gcode(user_inputs):
    final_prompt = (
        "Based on the details provided, generate the G-code for the CNC machining operation:\n\n"
        f"1. Type of CNC Machine: {user_inputs['Type of CNC Machine']}\n"
        f"2. Material: {user_inputs['Material']}\n"
        f"3. Tool Information:\n"
        f"   - Tool Type: {user_inputs['Tool Type']}\n"
        f"   - Tool Diameter: {user_inputs['Tool Diameter']}\n"
        f"4. Operation Details:\n"
        f"   - Operation Type: {user_inputs['Operation Type']}\n"
        f"   - Dimensions: {user_inputs['Dimensions']}\n"
        f"   - Coordinates: {user_inputs['Coordinates']}\n"
        f"   - Depth of Cut: {user_inputs['Depth of Cut']}\n"
        f"   - Feed Rate: {user_inputs['Feed Rate']}\n"
        f"   - Spindle Speed: {user_inputs['Spindle Speed']}\n"
        f"5. Additional Parameters:\n"
        f"   - Coolant type: {user_inputs['Coolant type']}\n"
        f"   - Tool or work offsets: {user_inputs['Tool or work offsets']}\n"
        f"   - Safety instructions: {user_inputs['Safety instructions']}\n"
    )

    gcode_response = get_response(final_prompt)
    return gcode_response


def main():
    st.title("G-code Generator for CNC Machines")

    st.write("Please provide the following details to generate the G-code for your CNC machining operation. If you are unsure about any value, you can leave it blank and the default value will be used.")

    # Dictionary to hold user inputs
    default_values = {
        "Type of CNC Machine": "milling machine",
        "Material": "aluminum",
        "Tool Type": "end mill",
        "Tool Diameter": "8mm",
        "Operation Type": "milling",
        "Dimensions": "100mm x 100mm",
        "Coordinates": "start: (0,0), end: (50,50)",
        "Depth of Cut": "2mm",
        "Feed Rate": "600 mm/min",
        "Spindle Speed": "1200 RPM",
        "Coolant type": "water",
        "Tool or work offsets": "none",
        "Safety instructions": "standard safety protocols"
    }

    user_inputs = {key: st.text_input(key, default_value) for key, default_value in default_values.items()}

    if st.button("Generate G-code"):
        gcode = generate_gcode(user_inputs)
        st.subheader("Generated G-code")
        st.text_area("G-code:", gcode, height=300)

        cleaned_gcode = clean_gcode(gcode)

        st.subheader("Cleaned G-code")
        st.text_area("Cleaned G-code:", cleaned_gcode, height=300)

        st.download_button(
            label="Download G-code",
            data=cleaned_gcode,
            file_name="generated.gcode",
            mime="text/plain"
        )

        satisfaction = st.radio("Are you satisfied with the generated G-code?", ("Yes", "No"))
        if satisfaction == "Yes":
            st.write("Great! If you need any more assistance, feel free to ask.")
        else:
            st.write("Please adjust the details and generate the G-code again.")


if __name__ == "__main__":
    main()
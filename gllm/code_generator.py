import os
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

    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=150,
    #     n=1,
    #     stop=None,
    #     temperature=0.7
    # )
    # Generate a chat completion
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

def collect_user_inputs():
    # Initial prompt to start the conversation
    initial_prompt = (
        "Sure, I can help you generate the G-code for your CNC machining operation. "
        "I'll need some details to get started. Please provide the following information:\n"
        "1. Type of CNC Machine (e.g., milling machine, lathe, router)\n"
        "2. Material (e.g., aluminum, steel, wood, plastic)\n"
        "3. Tool Information:\n"
        "   - Tool Type (e.g., end mill, drill bit, lathe tool)\n"
        "   - Tool Diameter (mm or inches)\n"
        "4. Operation Details:\n"
        "   - Operation Type (e.g., drilling, milling, turning)\n"
        "   - Dimensions of the workpiece and final product\n"
        "   - Coordinates for the operation (start and end points)\n"
        "   - Depth of Cut (mm or inches)\n"
        "   - Feed Rate (mm/min or inches/min)\n"
        "   - Spindle Speed (RPM)\n"
        "5. Additional Parameters (if applicable):\n"
        "   - Coolant type\n"
        "   - Tool or work offsets\n"
        "   - Any specific safety instructions or constraints\n\n"
        "If you are unsure about any value, you can leave it blank and I will use default values."
    )

    # Dictionary to hold user inputs
    user_inputs = {
        "Type of CNC Machine": None,
        "Material": None,
        "Tool Type": None,
        "Tool Diameter": None,
        "Operation Type": None,
        "Dimensions": None,
        "Coordinates": None,
        "Depth of Cut": None,
        "Feed Rate": None,
        "Spindle Speed": None,
        "Coolant type": None,
        "Tool or work offsets": None,
        "Safety instructions": None
    }

    # Default values for required information
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

    # Function to check if all required inputs are provided
    def all_inputs_provided(inputs):
        required_keys = [
            "Type of CNC Machine", "Material", "Tool Type", "Tool Diameter",
            "Operation Type", "Dimensions", "Coordinates", "Depth of Cut",
            "Feed Rate", "Spindle Speed"
        ]
        return all(inputs[key] is not None for key in required_keys)

    # Start the conversation
    print(initial_prompt)

    while not all_inputs_provided(user_inputs):
        user_input = input("Your response: ")

        # Find the first missing input and apply default if user input is empty
        for key in user_inputs:
            if user_inputs[key] is None:
                if user_input.strip() == "":
                    user_inputs[key] = default_values[key]
                else:
                    user_inputs[key] = user_input.strip()
                break

        # Check if all required inputs are provided
        if not all_inputs_provided(user_inputs):
            missing_info_prompt = "Thank you. Here’s the information I have so far:\n"
            for key, value in user_inputs.items():
                if value:
                    missing_info_prompt += f"- {key}: {value}\n"
            missing_info_prompt += "\nI still need the following details:\n"
            for key, value in user_inputs.items():
                if value is None:
                    missing_info_prompt += f"- {key}\n"
            missing_info_prompt += "Could you please provide the missing details?"

            print(missing_info_prompt)
    return user_inputs


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
    while True:
        # Collect user inputs
        user_inputs = collect_user_inputs()

        # Generate G-code
        gcode = generate_gcode(user_inputs)
        print("\nGenerated G-code:")
        print(gcode)

        # Ask if the user is satisfied with the generated G-code
        satisfaction = input("\nAre you satisfied with the generated G-code? (yes/no): ").strip().lower()
        if satisfaction == 'yes':
            print("Great! If you need any more assistance, feel free to ask.")
            break
        else:
            print("Let's try again.")


# def main():
#     # Initial prompt to start the conversation
#     initial_prompt = (
#         "Sure, I can help you generate the G-code for your CNC machining operation. "
#         "I'll need some details to get started. Please provide the following information:\n"
#         "1. Type of CNC Machine (e.g., milling machine, lathe, router)\n"
#         "2. Material (e.g., aluminum, steel, wood, plastic)\n"
#         "3. Tool Information:\n"
#         "   - Tool Type (e.g., end mill, drill bit, lathe tool)\n"
#         "   - Tool Diameter (mm or inches)\n"
#         "4. Operation Details:\n"
#         "   - Operation Type (e.g., drilling, milling, turning)\n"
#         "   - Dimensions of the workpiece and final product\n"
#         "   - Coordinates for the operation (start and end points)\n"
#         "   - Depth of Cut (mm or inches)\n"
#         "   - Feed Rate (mm/min or inches/min)\n"
#         "   - Spindle Speed (RPM)\n"
#         "5. Additional Parameters (if applicable):\n"
#         "   - Coolant type\n"
#         "   - Tool or work offsets\n"
#         "   - Any specific safety instructions or constraints\n\n"
#         "If you are unsure about any value, you can leave it blank and I will use default values."
#     )

#     # Dictionary to hold user inputs
#     user_inputs = {
#         "Type of CNC Machine": None,
#         "Material": None,
#         "Tool Type": None,
#         "Tool Diameter": None,
#         "Operation Type": None,
#         "Dimensions": None,
#         "Coordinates": None,
#         "Depth of Cut": None,
#         "Feed Rate": None,
#         "Spindle Speed": None,
#         "Coolant type": None,
#         "Tool or work offsets": None,
#         "Safety instructions": None
#     }

#     # Default values for required information
#     default_values = {
#         "Type of CNC Machine": "milling machine",
#         "Material": "aluminum",
#         "Tool Type": "end mill",
#         "Tool Diameter": "8mm",
#         "Operation Type": "milling",
#         "Dimensions": "100mm x 100mm",
#         "Coordinates": "start: (0,0), end: (50,50)",
#         "Depth of Cut": "2mm",
#         "Feed Rate": "600 mm/min",
#         "Spindle Speed": "1200 RPM",
#         "Coolant type": "water",
#         "Tool or work offsets": "none",
#         "Safety instructions": "standard safety protocols"
#     }

#     # Function to check if all required inputs are provided
#     def all_inputs_provided(inputs):
#         required_keys = [
#             "Type of CNC Machine", "Material", "Tool Type", "Tool Diameter",
#             "Operation Type", "Dimensions", "Coordinates", "Depth of Cut",
#             "Feed Rate", "Spindle Speed"
#         ]
#         return all(inputs[key] is not None for key in required_keys)

#     # Start the conversation
#     print(initial_prompt)

#     while not all_inputs_provided(user_inputs):
#         user_input = input("Your response: ")

#         # Find the first missing input
#         for key in user_inputs:
#             if user_inputs[key] is None:
#                 if user_input.strip() == "":
#                     user_inputs[key] = default_values[key]
#                 else:
#                     user_inputs[key] = user_input.strip()
#                 break

#         # Check if all required inputs are provided
#         if not all_inputs_provided(user_inputs):
#             missing_info_prompt = "Thank you. Here’s the information I have so far:\n"
#             for key, value in user_inputs.items():
#                 if value:
#                     missing_info_prompt += f"- {key}: {value}\n"
#             missing_info_prompt += "\nI still need the following details:\n"
#             for key, value in user_inputs.items():
#                 if value is None:
#                     missing_info_prompt += f"- {key}\n"
#             missing_info_prompt += "Could you please provide the missing details?"

#             print(missing_info_prompt)

#     # Now all required inputs are provided, generate the G-code
#     final_prompt = (
#         "Based on the details provided, generate the G-code for the CNC machining operation:\n\n"
#         f"1. Type of CNC Machine: {user_inputs['Type of CNC Machine']}\n"
#         f"2. Material: {user_inputs['Material']}\n"
#         f"3. Tool Information:\n"
#         f"   - Tool Type: {user_inputs['Tool Type']}\n"
#         f"   - Tool Diameter: {user_inputs['Tool Diameter']}\n"
#         f"4. Operation Details:\n"
#         f"   - Operation Type: {user_inputs['Operation Type']}\n"
#         f"   - Dimensions: {user_inputs['Dimensions']}\n"
#         f"   - Coordinates: {user_inputs['Coordinates']}\n"
#         f"   - Depth of Cut: {user_inputs['Depth of Cut']}\n"
#         f"   - Feed Rate: {user_inputs['Feed Rate']}\n"
#         f"   - Spindle Speed: {user_inputs['Spindle Speed']}\n"
#         f"5. Additional Parameters:\n"
#         f"   - Coolant type: {user_inputs['Coolant type']}\n"
#         f"   - Tool or work offsets: {user_inputs['Tool or work offsets']}\n"
#         f"   - Safety instructions: {user_inputs['Safety instructions']}\n"
#     )

#     gcode_response = get_response(final_prompt)
#     print("\nGenerated G-code:")
#     print(gcode_response)

if __name__ == '__main__':
    main()
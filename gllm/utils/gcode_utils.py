import re
import pygcode
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
        f"Desired Shape: {user_inputs['Desired Shape']}\n"
        f"Home Position: {user_inputs['Home Position']}\n"
        f"Shape Dimensions: {user_inputs['Shape Dimensions']}\n"
        f"Workpiece Dimensions: {user_inputs['Workpiece Dimensions']}\n"
        f"Coordinates: {user_inputs['Coordinates']}\n"
        f"Depth of Cut: {user_inputs['Depth of Cut']}\n"
        f"Feed Rate: {user_inputs['Feed Rate']}\n"
        f"Spindle Speed: {user_inputs['Spindle Speed']}\n"
        f"Additional Parameters:\n"
        f"Coolant type: {user_inputs['Coolant type'] if 'coolant type' in user_inputs else None}\n"
        f"Tool or work offsets: {user_inputs['Tool or work offsets'] if 'Tool or work offsets' in user_inputs else None}\n"
        f"Safety instructions: {user_inputs['Safety instructions'] if 'Safety instructions' in user_inputs else None}\n"
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


def validate_syntax(gcode_string):
    """Parsing G-code and checking for syntax errors."""
    try:
        for line in gcode_string.splitlines():
            gcode_line = pygcode.Line(line)
            for word in gcode_line.block.gcodes:
                # Validate that the word is a valid G-code command
                if not isinstance(word, pygcode.gcodes.GCode):
                    raise ValueError(f"Invalid G-code command: {word}")
        return True
    except Exception as e:
        print(f"Syntax error in G-code: {e}")
        return False

def validate_unreachable_code(gcode_string):
    """Detecting unreachable code in the program"""
    reached_end = False
    for line_text in gcode_string.strip().split('\n'):
        if 'M30' in line_text:
            reached_end = True
        elif reached_end:
            print(f"Unreachable code detected: {line_text}")
            return False
        else:
            print(f"Executed: {line_text}")

    return True

def validate_safety(gcode_string):
    """Ensuring that rapid movements do not pass through the material"""
    is_cutting = False
    for line_text in gcode_string.strip().split('\n'):
        if 'G1' in line_text or 'G01' in line_text:
            is_cutting = True
        if ('G0 ' in line_text or 'G00' in line_text) and is_cutting:
            print(f"Warning: Rapid movement through potential material at {line_text}")
            return False
        if is_cutting:
            is_cutting = False
        print(f"Processed: {line_text}")
    
    return True

def validate_continuity(gcode_string):
    """Checking for continuity in tool paths"""
    lines = gcode_string.strip().split('\n')
    last_position = None

    for line_text in lines:
        line = pygcode.Line(line_text)
        # Ensure the line has G-code commands and check for X, Y parameters
        if line.block.gcodes and 'X' in line.block.gcodes[0].params and 'Y' in line.block.gcodes[0].params:
            current_position = (line.block.gcodes[0].params['X'], line.block.gcodes[0].params['Y'])
            if last_position and (current_position != last_position):
                print(f"Discontinuity detected at {line_text}")
                return False
            last_position = current_position
        print(f"Processed: {line_text}")
    return True


def validate_feed_rate(gcode_string, min_feed, max_feed):
    """Ensure that the feed rate specified in G-code commands is within the acceptable limits for the material and tool being used. 
       This can prevent tool breakage and suboptimal machining conditions."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'F' in line.block.modal_params:
            feed_rate = line.block.modal_params['F']
            if not (min_feed <= feed_rate <= max_feed):
                print(f"Feed rate out of bounds at {line_text}")
                return False
    return True


def validate_tool_changes(gcode_string):
    expecting_initialization = False
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'T' in line.block.modal_params:
            expecting_initialization = True
        elif 'S' in line.block.modal_params and expecting_initialization:
            expecting_initialization = False
        elif expecting_initialization:
            print(f"Missing spindle speed after tool change before {line_text}")
            return False
    return True


def validate_spindle_speed(gcode_string, max_spindle_speed):
    """"Ensure that spindle speeds are within the machine's operational limits."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'S' in line.block.modal_params:
            spindle_speed = line.block.modal_params['S']
            if spindle_speed > max_spindle_speed:
                print(f"Spindle speed exceeds maximum limit at {line_text}")
                return False
    return True

def validate_z_levels(gcode_string, max_depth):
    """Verify that Z-level movements do not exceed certain depth limits to prevent the tool from crashing into the workpiece or machine bed."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'Z' in line.block.gcodes[0].params:
            z_level = line.block.gcodes[0].params['Z']
            if z_level > max_depth:
                print(f"Z-level exceeds maximum depth at {line_text}")
                return False
    return True


# def check_return_to_home(gcode_string, home_position=(0, 0, 0)):
#     """Ensure that the G-code program returns the tool to a safe position at the end (like returning to home position)."""
#     lines = gcode_string.strip().split('\n')
#     last_position = None
#     for line_text in lines:
#         line = pygcode.Line(line_text)
#         if any(param in line.block.gcodes[0].params for param in ['X', 'Y', 'Z']):
#             x = line.block.gcodes[0].params.get('X', last_position[0] if last_position else None)
#             y = line.block.gcodes[0].params.get('Y', last_position[1] if last_position else None)
#             z = line.block.gcodes[0].params.get('Z', last_position[2] if last_position else None)
#             last_position = (x, y, z)
#     if last_position != home_position:
#         print(f"Program does not return to home position. Last position was {last_position}")
#         return False
#     return True

def check_return_to_home(gcode_string, home_position=(0, 0, 0)):
    """
    Ensure that the G-code program returns the tool to a safe position at the end (like returning to home position).

    Parameters:
    gcode_string (str): The G-code program as a string.
    home_position (tuple): The expected home position (X, Y, Z).

    Returns:
    bool: True if the program returns to the home position, False otherwise.
    """
    lines = gcode_string.strip().split('\n')
    last_position = [None, None, None]  # X, Y, Z

    for line_text in lines:
        line = pygcode.Line(line_text)
        for word in line.block.gcodes:
            if isinstance(word, pygcode.gcodes.GCode):
                params = word.params
                if 'X' in params:
                    last_position[0] = params['X'].value
                if 'Y' in params:
                    last_position[1] = params['Y'].value
                if 'Z' in params:
                    last_position[2] = params['Z'].value

    # Replace None with the home position values if they were not specified in the G-code
    last_position = tuple(
        last_position[i] if last_position[i] is not None else home_position[i] 
        for i in range(3)
    )

    if last_position != home_position:
        print(f"Program does not return to home position. Last position was {last_position}")
        return False
    return True

def check_tool_offsets(gcode_string):
    """Validate that tool offsets are being used correctly and reset appropriately to avoid unintended tool paths."""
    tool_offset_active = False
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'G43' in line_text:  # Tool length offset compensation activate
            tool_offset_active = True
        elif 'G49' in line_text:  # Tool length offset compensation cancel
            tool_offset_active = False
        elif tool_offset_active and 'Z' in line.block.gcodes[0].params:
            print(f"Z movement with active tool offset in line: {line_text}")
            return False
    return True

def validate_gcode(gcode_string):

    is_syntax = validate_syntax(gcode_string)

    is_unreachable_code= validate_unreachable_code(gcode_string)

    is_safe = validate_safety(gcode_string)

    is_continuous = validate_continuity(gcode_string)

    is_valid_feed_rate= validate_feed_rate(gcode_string, min_feed=1, max_feed=100)

    is_valid_tool_change = validate_tool_changes(gcode_string)

    is_valid_spindle_speed = validate_spindle_speed(gcode_string, max_spindle_speed=900)

    is_return_home = check_return_to_home(gcode_string)

    is_tool_offset = check_tool_offsets(gcode_string)

    return True if  is_syntax and \
                    is_unreachable_code and \
                    is_safe and is_continuous and \
                    is_valid_feed_rate and \
                    is_valid_tool_change and \
                    is_return_home and \
                    is_tool_offset and \
                    is_valid_spindle_speed else False




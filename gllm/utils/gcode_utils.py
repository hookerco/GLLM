import re
import pygcode
import streamlit as st
from utils.plot_utils import plot_gcode
from utils.prompts_utils import REQUIRED_PARAMETERS
from langchain_core.messages.ai import AIMessage


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
        f"Shape Location: {user_inputs['Shape Location']}\n"
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
    gcode_response = gcode.content if isinstance(gcode, AIMessage) else str(gcode)
    gcode_pattern = re.compile(r"^(?:G|M|T|F|S|X|Y|Z|I|J|K|R|P|Q)\d+.*")
    cleaned_lines = [line.strip() for line in gcode_response.split('\n') if gcode_pattern.match(line)]
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
                    error_msg = f"Invalid G-code command: {word}"
                    raise ValueError(error_msg)
        return True, None
    except Exception as e:
        print(f"Syntax error in G-code: {e}")
        return False, e

def validate_unreachable_code(gcode_string):
    """Detecting unreachable code in the program"""
    reached_end = False
    for line_text in gcode_string.strip().split('\n'):
        if 'M30' in line_text:
            reached_end = True
        elif reached_end:
            error_msg = f"Unreachable code detected: {line_text}"
            print(error_msg)
            return False, error_msg
        else:
            print(f"Executed: {line_text}")

    return True, None 

def validate_safety(gcode_string):
    """Ensuring that rapid movements do not pass through the material"""
    is_cutting = False
    for line_text in gcode_string.strip().split('\n'):
        if 'G1' in line_text or 'G01' in line_text:
            is_cutting = True
        if ('G0 ' in line_text or 'G00' in line_text) and is_cutting:
            error_msg = f"Warning: Rapid movement through potential material at {line_text}"
            print(error_msg)
            return False, error_msg
        if is_cutting:
            is_cutting = False
        print(f"Processed: {line_text}")
    
    return True, None

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
                error_msg = f"Discontinuity detected at {line_text}"
                print(error_msg)
                return False, error_msg
            last_position = current_position
        print(f"Processed: {line_text}")
    return True, None


def validate_feed_rate(gcode_string, min_feed, max_feed):
    """Ensure that the feed rate specified in G-code commands is within the acceptable limits for the material and tool being used. 
       This can prevent tool breakage and suboptimal machining conditions."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'F' in line.block.modal_params:
            feed_rate = line.block.modal_params['F']
            if not (min_feed <= feed_rate <= max_feed):
                error_msg = f"Feed rate out of bounds at {line_text}"
                print(error_msg)
                return False, error_msg
    return True, None


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
            error_msg = f"Missing spindle speed after tool change before {line_text}"
            print(error_msg)
            return False, error_msg
    return True, None


def validate_spindle_speed(gcode_string, max_spindle_speed):
    """"Ensure that spindle speeds are within the machine's operational limits."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'S' in line.block.modal_params:
            spindle_speed = line.block.modal_params['S']
            if spindle_speed > max_spindle_speed:
                error_msg = f"Spindle speed exceeds maximum limit at {line_text}"
                print(error_msg)
                return False, error_msg
    return True, None

def validate_z_levels(gcode_string, max_depth):
    """Verify that Z-level movements do not exceed certain depth limits to prevent the tool from crashing into the workpiece or machine bed."""
    lines = gcode_string.strip().split('\n')
    for line_text in lines:
        line = pygcode.Line(line_text)
        if 'Z' in line.block.gcodes[0].params:
            z_level = line.block.gcodes[0].params['Z']
            if z_level > max_depth:
                error_msg = f"Z-level exceeds maximum depth at {line_text}"
                print(error_msg)
                return False, error_msg
    return True, None


def check_return_to_home(gcode_string, home_position=(0, 0)):
    """
    Ensure that the G-code program returns the tool to a safe position at the end (like returning to home position).

    Parameters:
    gcode_string (str): The G-code program as a string.
    home_position (tuple): The expected home position (X, Y, Z).

    Returns:
    bool: True if the program returns to the home position, False otherwise.
    """
    lines = gcode_string.strip().split('\n')
    current_position = [0, 0]  # Start at the origin
    absolute_positioning = True  # Most CNC machines start with absolute positioning

    for line_text in lines:
        line = pygcode.Line(line_text)
        for word in line.block.gcodes:
            if isinstance(word, pygcode.gcodes.GCode):
                # Check for positioning mode
                if word.modal_group == 'MG_GROUP_03':
                    if word.gcode == 90:  # G90 - Absolute positioning
                        absolute_positioning = True
                    elif word.gcode == 91:  # G91 - Relative positioning
                        absolute_positioning = False

                # Update positions
                params = word.params
                if absolute_positioning:
                    if 'X' in params:
                        current_position[0] = params['X'].value
                    if 'Y' in params:
                        current_position[1] = params['Y'].value
                else:
                    if 'X' in params:
                        current_position[0] += params['X'].value
                    if 'Y' in params:
                        current_position[1] += params['Y'].value


    # Compare the final position to the home position
    last_position = tuple(current_position)
    if last_position != home_position:
        error_msg = f"Program does not return to home position ({home_position}). Last position was {last_position}"
        print(error_msg)
        return False, error_msg
    return True, None


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
            error_msg = f"Z movement with active tool offset in line: {line_text}"
            print(error_msg)
            return False, error_msg
    return True, None

def validate_gcode(gcode_string):

    is_syntax,_ = validate_syntax(gcode_string)

    is_unreachable_code,_ = validate_unreachable_code(gcode_string)

    is_safe,_ = validate_safety(gcode_string)

    is_continuous,_ = validate_continuity(gcode_string)

    is_valid_feed_rate,_ = validate_feed_rate(gcode_string, min_feed=1, max_feed=100)

    is_valid_tool_change,_ = validate_tool_changes(gcode_string)

    is_valid_spindle_speed,_ = validate_spindle_speed(gcode_string, max_spindle_speed=900)

    #is_return_home = check_return_to_home(gcode_string)

    is_tool_offset,_ = check_tool_offsets(gcode_string)

    return True if  is_syntax and \
                    is_unreachable_code and \
                    is_safe and is_continuous and \
                    is_valid_feed_rate and \
                    is_valid_tool_change and \
                    is_tool_offset and \
                    is_valid_spindle_speed else False




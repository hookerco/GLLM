import numpy as np
import matplotlib.pyplot as plt
import re

def parse_coordinates(command):
    # Regular expression to find coordinates
    coord_pattern = re.compile(r'[XYZIJR]-?\d+\.?\d*')
    coords = coord_pattern.findall(command)
    coord_dict = {}
    for coord in coords:
        if coord.startswith('X'):
            coord_dict['X'] = float(coord[1:])
        elif coord.startswith('Y'):
            coord_dict['Y'] = float(coord[1:])
        elif coord.startswith('Z'):
            coord_dict['Z'] = float(coord[1:])
        elif coord.startswith('I'):
            coord_dict['I'] = float(coord[1:])
        elif coord.startswith('J'):
            coord_dict['J'] = float(coord[1:])
        elif coord.startswith('R'):
            coord_dict['R'] = float(coord[1:])
    return coord_dict


def parse_gcode(gcode):
    x_points, y_points = [], []
    x, y = 0, 0  # Initialize starting point
    # x_points.append(x)
    # y_points.append(y)
    
    for command in gcode.splitlines():
        
        command = command.split(';')[0].strip()  # Remove comments and trim
        
        if not command:
            continue
        elif 'M30' in command:
            # End of program
            break
        elif 'G00' in command or 'G0 ' in command:
            print("rapid positioning", command)
            # Rapid positioning
            coords = parse_coordinates(command)
            x = coords.get('X', x)
            y = coords.get('Y', y)
            print(command, x, y)
            x_points.append(x)
            y_points.append(y)
        elif 'G01' in command or 'G1 ' in command:
            # Linear interpolation
            print("linear interpolation", command)
            coords = parse_coordinates(command)
            print(command, x, y)
            x = coords.get('X', x)
            y = coords.get('Y', y)
            x_points.append(x)
            y_points.append(y)
        # Handle circular interpolation if present
        elif 'G02' in command or 'G2 ' in command or 'G03' in command or 'G3 ' in command:
            print("plotting Circualar shape!", command)
            # Circular interpolation
            coords = parse_coordinates(command)
            i_center = coords.get('I', 0)
            j_center = coords.get('J', 0)
            radius = coords.get('R', None)

            if radius is not None:
                # Calculate center of the arc from radius
                dx = x - (x + radius)
                dy = y - (y + radius)
                center_x = x + np.cos(np.arctan2(dy, dx)) * radius
                center_y = y + np.sin(np.arctan2(dy, dx)) * radius
            else:
                center_x = x + i_center
                center_y = y + j_center
            
            # Determine the end position
            if 'X' in coords and 'Y' in coords:
                x_end = coords.get('X', x)
                y_end = coords.get('Y', y)
            else:
                # If no end coordinates are provided, compute them assuming a complete circle
                x_end = x
                y_end = y

            # Calculate angles for the arc
            start_angle = np.arctan2(y - center_y, x - center_x)
            end_angle = np.arctan2(y_end - center_y, x_end - center_x)

            # Generate points along the arc
            if 'G02' in command or 'G2' in command:  # Clockwise
                if start_angle <= end_angle:
                    start_angle += 2 * np.pi
            elif 'G03' in command or 'G3' in command:  # Counterclockwise
                if start_angle >= end_angle:
                    end_angle += 2 * np.pi

            angles = np.linspace(start_angle, end_angle, 100)
            arc_x = center_x + radius * np.cos(angles) if radius is not None else center_x + np.sqrt(i_center**2 + j_center**2) * np.cos(angles)
            arc_y = center_y + radius * np.sin(angles) if radius is not None else center_y + np.sqrt(i_center**2 + j_center**2) * np.sin(angles)
            x_points.extend(arc_x)
            y_points.extend(arc_y)
            # Update current position to end of arc
            x, y = x_end, y_end
        
        else:
            # skipp all other irrelevant commands
            continue
    
    return x_points, y_points


def plot_gcode(gcode):

    x_points, y_points = parse_gcode(gcode)

    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker='o')
    plt.title('CNC Path Plot')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True)
    plt.axis('equal')
    return plt


def plot_user_specification(parsed_parameters):
    """Plots the CNC task in 2D."""

    wp_dims = parsed_parameters['workpiece_diemensions']
    start_point = parsed_parameters['starting_point']
    tool_path = parsed_parameters['tool_path']
    cut_depth = parsed_parameters['cut_depth'][0]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot workpiece as a rectangle
    rect = plt.Rectangle((0, 0), wp_dims[0], wp_dims[1], 
                         linewidth=2, edgecolor='k', facecolor='lightgray')
    ax.add_patch(rect)

    # Plot tool path
    x_path, y_path, _ = zip(*tool_path)  # Ignore z-coordinates for 2D plot

    # Move to starting point if not already at the beginning
    if tool_path and (start_point[0], start_point[1]) != tool_path[0]:
        x_path = (start_point[0],) + x_path
        y_path = (start_point[1],) + y_path

    ax.plot(x_path, y_path, 'r-', linewidth=2, label='Tool Path')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('CNC Task Visualization (2D)')
    ax.legend()

    # Set axis limits to match workpiece dimensions
    ax.set_xlim([0, wp_dims[0]])
    ax.set_ylim([0, wp_dims[1]])

    # Add cut depth as text annotation
    ax.text(0.05, 0.95, f'Cut Depth: {cut_depth}mm', 
            transform=ax.transAxes, verticalalignment='top')

    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    return plt
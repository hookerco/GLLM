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


def plot_gcode(gcode):
    x_points, y_points = [], []
    x, y = 0, 0  # Initialize starting point
    x_points.append(x)
    y_points.append(y)
    
    for command in gcode.splitlines():
        
        command = command.split(';')[0].strip()  # Remove comments and trim
        if not command:
            continue
        
        if 'G21' in command:
            # Set units to millimeters (default assumption)
            pass
        elif 'G17' in command:
            # Select XY plane (default assumption)
            pass
        elif 'G90' in command:
            # Set to absolute positioning (default assumption)
            pass
        elif 'M06' in command:
            # Tool change (assume tool is ready)
            pass
        elif 'M30' in command:
            # End of program
            break
        elif 'G00' in command or 'G0' in command:
            # Rapid positioning
            coords = parse_coordinates(command)
            x = coords.get('X', x)
            y = coords.get('Y', y)
            x_points.append(x)
            y_points.append(y)
        elif 'G01' in command or 'G1' in command:
            # Linear interpolation
            coords = parse_coordinates(command)
            x = coords.get('X', x)
            y = coords.get('Y', y)
            x_points.append(x)
            y_points.append(y)

        # Handle circular interpolation if present
        elif 'G02' in command or 'G2' in command or 'G03' in command or 'G3' in command:
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

            if 'X' in coords and 'Y' in coords:
                x_end = coords.get('X', x)
                y_end = coords.get('Y', y)
            else:
                # If no end coordinates are provided, compute them assuming a complete circle
                x_end = x
                y_end = y
                print(x_end, y_end)

            start_angle = np.arctan2(y - center_y, x - center_x)
            end_angle = np.arctan2(y_end - center_y, x_end - center_x)

            if 'G02' in command or 'G2' in command:  # Clockwise
                if start_angle <= end_angle:
                    start_angle += 2 * np.pi
            elif 'G03' in command or 'G3' in command:  # Counterclockwise
                if start_angle >= end_angle:
                    end_angle += 2 * np.pi

            angles = np.linspace(start_angle, end_angle, 100)
            print(angles)
            print(start_angle, end_angle)
            arc_x = center_x + radius * np.cos(angles) if radius is not None else center_x + np.sqrt(i_center**2 + j_center**2) * np.cos(angles)
            arc_y = center_y + radius * np.sin(angles) if radius is not None else center_y + np.sqrt(i_center**2 + j_center**2) * np.sin(angles)
            x_points.extend(arc_x)
            y_points.extend(arc_y)
            x = x_end
            y = y_end
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, y_points, marker='o')
    plt.title('CNC Path Plot')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True)
    plt.axis('equal')
    return plt
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

from typing import Union, List


def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def wrap_angle(angle):
    # Only Works in Radians
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle_list(angles):
    angles = np.array(angles)
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    return angles


def wrap_value_zero_max(v, max_v):
    """
    Wraps value (v) between zero and the max value (max_v) given.
    """
    if max_v <= 0:
        raise ValueError("max_value must be greater than 0")
    return v % max_v


def wrap_list_zero_max(values, max_values):
    if isinstance(values, list) and isinstance(max_values, list):
        if len(values) != len(max_values):
            raise ValueError("values and max_values lists must have the same length")
        return [wrap_value_zero_max(v, max_v) for v, max_v in zip(values, max_values)]
    elif isinstance(values, list):
        return [wrap_value_zero_max(v, max_values) for v in values]
    else:
        return wrap_value_zero_max(values, max_values)


def wrap_value_half_max(v, max_v):
    """
    Wraps value (v) between -max_v/2 and +max_v/2.
    """
    if max_v <= 0:
        raise ValueError("max_value must be greater than 0")

    half_max_v = max_v / 2

    # Wrap the value between 0 and max_v first
    wrapped_value = (v + half_max_v) % max_v - half_max_v

    return wrapped_value


def wrap_list_half_max(values, max_values):
    if isinstance(values, list) and isinstance(max_values, list):
        if len(values) != len(max_values):
            raise ValueError("values and max_values lists must have the same length")
        return [wrap_value_half_max(v, max_v) for v, max_v in zip(values, max_values)]
    elif isinstance(values, list):
        return [wrap_value_half_max(v, max_values) for v in values]
    else:
        return wrap_value_half_max(values, max_values)


def check_linear_in_range(value: float, valid_range: tuple, tolerance: float = 1e-3):
    
    lowBound, highBound = valid_range
    return (lowBound - tolerance) <= value <= (highBound + tolerance)


def linear_dist_in_range(value1, value2, valid_range):
    # Check if initial angles are within the valid range
    if not check_linear_in_range(value1, valid_range):
        raise ValueError(f"value1 ({value1}) is not within the valid range.")
    if not check_linear_in_range(value1, valid_range):
        raise ValueError(f"value2 ({value1}) is not within the valid range.")
    return value2 - value1


def angle_dist(angle1, angle2):
    # Wrap angles to be within -pi to pi
    angle1 = wrap_angle(angle1)
    angle2 = wrap_angle(angle2)
    diff = angle2 - angle1
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff


def check_angle_in_range(angle: float, valid_range: Union[list, tuple], tolerance = 1e-3):
    # Wrap angle to be within -pi to pi
    angle = wrap_angle(angle)
    lowBound, highBound = valid_range

    if lowBound < highBound:
        return (lowBound - tolerance) <= angle <= (highBound + tolerance)
    elif lowBound > highBound:
        return angle >= (lowBound - tolerance) or angle <= (highBound + tolerance)
    else:
        raise ValueError("Invalid range. Bounds must be different.")


def angle_dist_in_range(angle1, angle2, valid_range):
    # Check if initial angles are within the valid range
    if not check_angle_in_range(angle1, valid_range):
        raise ValueError(
            f"angle1 ({angle1}) is not within the valid range {valid_range}."
        )
    if not check_angle_in_range(angle2, valid_range):
        raise ValueError(f"angle2 ({angle2}) is not within the valid range {valid_range}.")

    short_dist = angle_dist(angle1, angle2)
    direction = 1 if short_dist >= 0 else -1
    step = direction * math.pi / 180  # Increment in degrees, converted to radians

    # Check short path
    current_angle = angle1
    for _ in range(int(abs(short_dist) // abs(step))):
        current_angle = (current_angle + step) % (2 * math.pi)
        if not check_angle_in_range(current_angle, valid_range):
            # Calculate long path if short path is not valid
            long_dist = 2 * math.pi - abs(short_dist)
            return long_dist * direction
    return short_dist


def calc_dist_in_range(value1, value2, is_linear, valid_range):
    """
    Calculate distances between values either linearly or rotationally based on the is_linear flag.
    Supports both single values and lists of values.

    Parameters:
    value1 (float/list): First value or list of values.
    value2 (float/list): Second value or list of values.
    is_linear (bool/list): Boolean or list of booleans indicating calculation mode.
    valid_range (tuple/list of tuples): Valid range(s) for rotational calculations.

    Returns:
    float/list: The calculated distances.
    """

    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        if is_linear:
            return linear_dist_in_range(value1, value2)
        else:
            return angle_dist_in_range(value1, value2, valid_range)

    distances = []
    for i,(v1, v2, lin, range_val) in enumerate(zip(value1, value2, is_linear, valid_range)):
        if lin:
            distances.append(linear_dist_in_range(v1, v2, range_val))
        else:
            distances.append(angle_dist_in_range(v1, v2, range_val))

    return distances


def interpolate_q_in_range(q1, q2, joint_type, joint_ranges, n_interp):
    import numpy as np

    print("JOINT INTERPOLATION IN RANGE")
    print(f"q1: {q1}")
    print(f"q2: {q2}")
    print(f"joint_type: {joint_type}")
    print(f"joint_ranges: {joint_ranges}")
    print(f"n_interp: {n_interp}")

    if len(q1) != len(q2):
        raise ValueError("q arrays must be of the same length.")

    # Assuming calc_dist_in_range is a function that calculates the distance considering joint limits
    valid_distances = calc_dist_in_range(q1, q2, joint_type, joint_ranges)
    print(f"Valid distances in range: {valid_distances}")

    q_interp = np.zeros((n_interp, len(q1)))

    for i in range(len(q1)):
        print(f"Interpolating joint {i} from {q1[i]} to {q2[i]}")
        # Determine the number of interpolation points for each joint, excluding the last point
        points = np.linspace(0, valid_distances[i], n_interp + 1)
        print(f"Points for joint {i}: {points}")

        # Interpolating for each joint
        for j, n in enumerate(
            points[:-1]
        ):  # Exclude the last point to avoid q2 duplication
            interpolated_value = q1[i] + n
            print(f"Interpolated value for joint {i} at step {j}: {interpolated_value}")
            q_interp[j, i] = interpolated_value

    print(f"q_interp before wrapping angles: {q_interp}")

    # Wrap angles if necessary, assuming wrap_angle_list is a function that handles angle wrapping
    q_interp = wrap_angle_list(q_interp)
    print(f"q_interp after wrapping angles: {q_interp}")

    return q_interp

def get_one_every_n(array, n):
    """
    Given a listreturns a new array containing only
    one out of n intermediate elements.
    """
    array_lite = []
    for i in range(0, len(array), n):
        array_lite.append(array[i])

    return np.array(array_lite)


def find_value_index(container, value):
    if isinstance(container, list):
        try:
            return container.index(value)
        except ValueError:
            return -1  # Return -1 if the value is not found
    elif isinstance(container, np.ndarray):
        indices = np.where(container == value)
        return (
            indices[0][0] if indices[0].size > 0 else -1
        )  # Return -1 if the value is not found
    else:
        raise TypeError("The container must be either a list or a numpy array")


def round_list(input_list, decimals=2):
    rounded_list = []
    for elem in input_list:
        if isinstance(elem, list):
            rounded_list.append(
                round_list(elem, decimals)
            )  # Recursively round elements in nested lists
        elif isinstance(elem, tuple):
            rounded_list.append(
                tuple(round_list(list(elem), decimals))
            )  # Convert tuple to list, round, then convert back to tuple
        elif isinstance(elem, (int, float)):
            rounded_list.append(round(elem, decimals))
        else:
            rounded_list.append(
                elem
            )  # If the element is not a list, tuple, int, or float, append it as is
    return rounded_list


def floats_equal(a, b, tol=1e-9):
    """
    Check if two floats or lists of floats are equal within a given tolerance.

    Parameters:
    a (float or list of floats): First number or list of numbers to compare.
    b (float or list of floats): Second number or list of numbers to compare.
    tol (float): Tolerance within which the numbers are considered equal. Default is 1e-9.

    Returns:
    bool: True if the numbers or lists are equal within the given tolerance, False otherwise.
    """
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(abs(x - y) <= tol for x, y in zip(a, b))
    elif isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    else:
        raise ValueError(
            "Both inputs must be either floats or lists of floats of the same length."
        )


def nearest_neighbor(points, target_value, return_index=False):
    """
    Finds the nearest neighbor of a target value from an array of points using a KD-tree.

    Args:
    points (np.ndarray): An array of points (each a row in the array) from which the nearest neighbor is found.
    target_value (array-like): The target value as an array-like object for which the nearest neighbor is searched.

    Returns:
    np.ndarray or int: The nearest point in the array to the target value, or the index of that point.
    """
    # Ensure that points is a numpy array
    points = np.asarray(points)
    target_value = np.asarray(target_value)

    # Ensure points is a 2D array
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    target_value = np.asarray(target_value)
    if target_value.ndim == 1:
        target_value = target_value.reshape(1, -1)

    # Create a KDTree object using the array of points
    kdtree = KDTree(points)

    # Query the KDTree for the nearest neighbor to the target value
    _, index = kdtree.query(target_value)

    if return_index:
        return index
    else:
        # Return the nearest neighbor
        return points[index]


def disable_array(array, el_disable):

    return np.multiply(array, 1 - el_disable)


def map_values(values, old_max_value, new_max_value, to_int=False):

    def map_single_value(value):
        new_value = (value / old_max_value) * new_max_value
        if to_int:
            new_value = int(
                round(new_value)
            )  # Using round before int conversion for better accuracy
        return new_value

    if isinstance(values, list):
        return [map_single_value(value) for value in values]
    else:
        return map_single_value(values)


def flatten_list(nested_list):
    return [element for sublist in nested_list for element in sublist]


def unflatten_list(flat_list, final_list_structure):

    result = []

    start = 0

    for length in final_list_structure:

        end = start + length
        result.append(flat_list[start:end])
        start = end

    return result


def interpolate_arrays(start, finish, i):
    """
    Interpolates i intermediate steps between two numpy arrays 'start' and 'finish'.

    Parameters:
        start (np.ndarray): The starting array.
        finish (np.ndarray): The ending array.
        i (int): The number of intermediate steps to generate.

    Returns:
        np.ndarray: A new array of shape (i, len(start)) with interpolated values.
    """
    # Ensure 'start' and 'finish' are numpy arrays and have the same shape
    start = np.asarray(start)
    finish = np.asarray(finish)

    if start.shape != finish.shape:
        raise ValueError("Start and finish arrays must have the same shape.")

    interpolated = np.linspace(start, finish, i, axis=0)

    return interpolated


if __name__ == "__main__":

    # Define the valid range, excluding a sector
    valid_range = (math.radians(-10), math.radians(10))
    print(f"Valid range (in radians): {round_list(valid_range)}")

    # Convert angles from degrees to radians
    angle1 = math.radians(30)  # Start angle in radians
    angle2 = math.radians(300)  # Target angle in radians

    # Calculate the shortest angular distance
    shortest_distance = angle_dist(angle1, angle2)
    print(f"Shortest Angular Distance (in radians): {shortest_distance:.2f}")
    print(
        f"Shortest Angular Distance (in degrees): {shortest_distance * 180 / math.pi:.2f} degrees"
    )

    # Calculate the distance using the valid range
    try:
        valid_distance = angle_dist_in_range(angle1, angle2, valid_range)
        print(f"Valid Angular Distance (in radians): {valid_distance:.2f}")
        print(
            f"Valid Angular Distance (in degrees): {valid_distance * 180 / math.pi:.2f} degrees"
        )
    except ValueError as e:
        print(str(e))

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

from typing import Union, List


def wrap_angle(angle):
    """
    Wraps an angle to the range [-π, π].

    This function uses the `atan2` function to compute the wrapped angle
    by considering the sine and cosine of the input angle. The result is
    an equivalent angle in the range [-π, π], effectively keeping the angle
    within one full rotation.

    Args:
        angle (float): The input angle in radians.

    Returns:
        float: The wrapped angle in the range [-π, π].
    """

    # Use atan2 to wrap the angle to the range [-π, π] by computing
    # the arctangent of the sine and cosine of the angle.
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle_list(angles):
    """
    Wraps a list of angles to the range [-pi, pi].

    This function takes a list of angles (in radians), wraps each angle so that
    it lies within the range [-π, π]. It does so using the arctangent function
    on the sine and cosine of the angles, ensuring all angles fall within the
    desired range.

    Parameters:
    angles (list or array-like): A list or array of angles in radians.

    Returns:
    numpy.ndarray: An array of angles wrapped to the range [-pi, pi].
    """
    # Convert the input angles to a NumPy array for efficient numerical operations.
    angles = np.array(angles)

    # Use the arctan2 function to wrap angles. This computes the angle from the
    # sine and cosine of the input, returning values between -pi and pi.
    angles = np.arctan2(np.sin(angles), np.cos(angles))

    # Return the wrapped angles as a NumPy array.
    return angles


def wrap_value_zero_max(v, max_v):
    """
    Wraps a value `v` into the range [0, max_v).

    This function ensures that the value `v` is wrapped within the range from
    0 (inclusive) to `max_v` (exclusive). If the value exceeds `max_v`, it
    wraps around. If `v` is negative, it will still wrap into the range.

    Parameters:
    v (float or int): The value to wrap.
    max_v (float or int): The upper limit for wrapping. Must be greater than 0.

    Returns:
    float or int: The wrapped value, which will be in the range [0, max_v).

    Raises:
    ValueError: If `max_v` is less than or equal to 0.
    """
    # Ensure that the maximum value is greater than 0, as wrapping only makes sense with positive max_v.
    if max_v <= 0:
        raise ValueError("max_v must be greater than 0")

    # Use modulus to wrap the value into the range [0, max_v).
    return v % max_v


def wrap_list_zero_max(values, max_values):
    """
    Wraps values or a list of values into the range [0, max_values).

    This function wraps one or more values into the range defined by `max_values`.
    It supports three cases:
    1. If both `values` and `max_values` are lists, each value in `values` is
       wrapped by the corresponding value in `max_values`. The lists must have
       the same length.
    2. If `values` is a list and `max_values` is a single number, each value
       in `values` is wrapped by `max_values`.
    3. If `values` is a single number, it is wrapped by `max_values`.

    Parameters:
    values (list or float/int): A list of values or a single value to be wrapped.
    max_values (list or float/int): A list of max values or a single max value
                                    to wrap the values by.

    Returns:
    list or float/int: The wrapped values. If input was a list, a list of wrapped
                       values is returned. If input was a single value, a single
                       wrapped value is returned.

    Raises:
    ValueError: If both `values` and `max_values` are lists of different lengths.
    """
    # If both values and max_values are lists, ensure they have the same length
    if isinstance(values, list) and isinstance(max_values, list):
        if len(values) != len(max_values):
            # Raise an error if the lists are not of equal length
            raise ValueError("values and max_values lists must have the same length")

        # Use list comprehension to wrap each value with the corresponding max value
        return [wrap_value_zero_max(v, max_v) for v, max_v in zip(values, max_values)]

    # If only values is a list, wrap each value by the single max_values value
    elif isinstance(values, list):
        return [wrap_value_zero_max(v, max_values) for v in values]

    # If neither are lists, treat them as single values and wrap accordingly
    else:
        return wrap_value_zero_max(values, max_values)


def wrap_value_half_max(v, max_v):
    """
    Wraps a value `v` into the range [-max_v/2, +max_v/2).

    This function wraps a value into the range centered around zero,
    specifically between `-max_v/2` (inclusive) and `+max_v/2` (exclusive).
    It first shifts the value by `max_v/2` to ensure it falls within a range of
    [0, max_v), then adjusts back to center the range.

    Parameters:
    v (float or int): The value to be wrapped.
    max_v (float or int): The upper limit of the wrapping range. Must be greater than 0.

    Returns:
    float or int: The wrapped value, which will be in the range [-max_v/2, max_v/2).

    Raises:
    ValueError: If `max_v` is less than or equal to 0.
    """
    # Ensure that max_v is positive, as negative or zero max_v is not valid for wrapping
    if max_v <= 0:
        raise ValueError("max_v must be greater than 0")

    # Compute half of the max_v value
    half_max_v = max_v / 2

    # Shift the value by half_max_v to wrap it within the range [0, max_v),
    # then shift it back to the range [-max_v/2, +max_v/2).
    wrapped_value = (v + half_max_v) % max_v - half_max_v

    return wrapped_value


def wrap_list_half_max(values, max_values):
    """
    Wraps values or a list of values into the range [-max_v/2, +max_v/2).

    This function wraps one or more values into the range centered around zero,
    specifically between `-max_v/2` and `+max_v/2`. It supports three cases:
    1. If both `values` and `max_values` are lists, each value in `values` is
       wrapped by the corresponding value in `max_values`. The lists must have
       the same length.
    2. If `values` is a list and `max_values` is a single number, each value
       in `values` is wrapped by `max_values`.
    3. If `values` is a single number, it is wrapped by `max_values`.

    Parameters:
    values (list or float/int): A list of values or a single value to be wrapped.
    max_values (list or float/int): A list of max values or a single max value
                                    to wrap the values by.

    Returns:
    list or float/int: The wrapped values. If input was a list, a list of wrapped
                       values is returned. If input was a single value, a single
                       wrapped value is returned.

    Raises:
    ValueError: If both `values` and `max_values` are lists of different lengths.
    """
    # If both values and max_values are lists, ensure they have the same length
    if isinstance(values, list) and isinstance(max_values, list):
        if len(values) != len(max_values):
            # Raise an error if the lists are not of equal length
            raise ValueError("values and max_values lists must have the same length")

        # Use list comprehension to wrap each value with the corresponding max value
        return [wrap_value_half_max(v, max_v) for v, max_v in zip(values, max_values)]

    # If only values is a list, wrap each value by the single max_values value
    elif isinstance(values, list):
        return [wrap_value_half_max(v, max_values) for v in values]

    # If neither are lists, treat them as single values and wrap accordingly
    else:
        return wrap_value_half_max(values, max_values)


def check_linear_in_range(
    value: float, valid_range: tuple, tolerance: float = 1e-3
) -> bool:
    """
    Checks if a given value is within a specified range, with optional tolerance.

    This function checks whether a value falls within a given range `[lowBound, highBound]`,
    allowing for a tolerance. The tolerance extends the valid range slightly beyond
    the provided bounds, ensuring that values close to the boundaries (within tolerance)
    are still considered valid.

    Parameters:
    value (float): The value to check.
    valid_range (tuple): A tuple of two floats representing the valid range (lowBound, highBound).
    tolerance (float, optional): The allowable tolerance when checking the boundaries.
                                 Default is `1e-3`.

    Returns:
    bool: True if the value is within the valid range, accounting for tolerance; False otherwise.
    """
    # Unpack the valid range into lower and upper bounds
    lowBound, highBound = valid_range

    # Check if the value is within the extended range, considering tolerance on both sides
    return (lowBound - tolerance) <= value <= (highBound + tolerance)


def linear_dist_in_range(value1: float, value2: float, valid_range: tuple) -> float:
    """
    Calculates the linear distance between two values if both are within the specified range.

    This function checks whether two values (`value1` and `value2`) fall within the given
    `valid_range`. If either value is outside the range, a `ValueError` is raised. If both
    values are valid, the function returns the difference `value2 - value1`.

    Parameters:
    value1 (float): The first value to check.
    value2 (float): The second value to check.
    valid_range (tuple): A tuple representing the valid range (lowBound, highBound)
                         for both values.

    Returns:
    float: The difference `value2 - value1`, representing the linear distance between
           the two values.

    Raises:
    ValueError: If either `value1` or `value2` is not within the valid range.
    """
    # Check if value1 is within the valid range
    if not check_linear_in_range(value1, valid_range):
        raise ValueError(f"value1 ({value1}) is not within the valid range.")

    # Check if value2 is within the valid range
    if not check_linear_in_range(value2, valid_range):
        raise ValueError(f"value2 ({value2}) is not within the valid range.")

    # Return the difference between value2 and value1 (the linear distance)
    return value2 - value1


def angle_dist(angle1: float, angle2: float) -> float:
    """
    Computes the shortest angular distance between two angles.

    This function calculates the smallest difference between two angles (`angle1` and `angle2`),
    where the angles are wrapped to the range [-π, π] before the computation. The result will
    be in the range [-π, π], representing the shortest angular distance between the two angles.

    Parameters:
    angle1 (float): The first angle in radians.
    angle2 (float): The second angle in radians.

    Returns:
    float: The shortest angular distance between `angle1` and `angle2`, in radians, within
           the range [-π, π].
    """
    # Wrap both angles to the range [-π, π] using the wrap_angle function
    angle1 = wrap_angle(angle1)
    angle2 = wrap_angle(angle2)

    # Compute the difference between the two angles
    diff = angle2 - angle1

    # Normalize the difference to be within the range [-π, π]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi

    return diff


def check_angle_in_range(
    angle: float, valid_range: Union[list, tuple], tolerance: float = 1e-3
) -> bool:
    """
    Checks if an angle is within a specified angular range, with optional tolerance.

    This function wraps the given `angle` to the range [-π, π] and checks if it falls
    within the specified `valid_range`, which can span [-π, π]. The range can be either
    continuous (lowBound < highBound) or can wrap around (-π to +π) if (lowBound > highBound).
    A small tolerance is allowed to account for precision errors.

    Parameters:
    angle (float): The angle to check, in radians.
    valid_range (list or tuple): A tuple or list of two floats (lowBound, highBound)
                                 representing the valid angular range.
    tolerance (float, optional): The allowable tolerance when checking the boundaries.
                                 Default is `1e-3`.

    Returns:
    bool: True if the angle is within the valid range (accounting for tolerance);
          False otherwise.

    Raises:
    ValueError: If the `valid_range` is invalid (i.e., lowBound equals highBound).
    """
    # Wrap the angle to ensure it's within the range [-π, π]
    angle = wrap_angle(angle)

    # Unpack the valid range into lower and upper bounds
    lowBound, highBound = valid_range

    # If the range is continuous (lowBound < highBound), check if angle is within the range
    if lowBound < highBound:
        return (lowBound - tolerance) <= angle <= (highBound + tolerance)

    # If the range wraps around (lowBound > highBound), check if angle is outside or crossing the wrap
    elif lowBound > highBound:
        return angle >= (lowBound - tolerance) or angle <= (highBound + tolerance)

    # If lowBound and highBound are equal, the range is invalid
    else:
        raise ValueError("Invalid range. Bounds must be different.")


def angle_dist_in_range(angle1: float, angle2: float, valid_range: tuple) -> float:
    """
    Calculates the angular distance between two angles within a specified valid range.

    This function computes the shortest angular distance between `angle1` and `angle2`,
    ensuring both angles are within the `valid_range`. If the direct (short) path
    between the angles goes outside the valid range, the function calculates and returns
    the long path around the circle. The valid range can handle wrapped intervals.

    Parameters:
    angle1 (float): The first angle, in radians.
    angle2 (float): The second angle, in radians.
    valid_range (tuple): A tuple (lowBound, highBound) specifying the valid angular range.
                         It can span across [-π, π].

    Returns:
    float: The angular distance between `angle1` and `angle2`, either the short or long
           path, depending on whether the short path stays within the valid range.

    Raises:
    ValueError: If either `angle1` or `angle2` is not within the valid range.
    """
    # Check if both angles are within the valid range
    if not check_angle_in_range(angle1, valid_range):
        raise ValueError(
            f"angle1 ({angle1}) is not within the valid range {valid_range}."
        )
    if not check_angle_in_range(angle2, valid_range):
        raise ValueError(
            f"angle2 ({angle2}) is not within the valid range {valid_range}."
        )

    # Calculate the shortest angular distance between the two angles
    short_dist = angle_dist(angle1, angle2)
    direction = 1 if short_dist >= 0 else -1  # Determine the direction of movement
    step = direction * math.pi / 180  # Increment in degrees, converted to radians

    # Check if the short path remains within the valid range
    current_angle = angle1
    for _ in range(int(abs(short_dist) // abs(step))):
        # Incrementally move along the short path
        current_angle = (current_angle + step) % (2 * math.pi)
        if not check_angle_in_range(current_angle, valid_range):
            # If at any step the angle is outside the valid range, calculate the long path
            long_dist = 2 * math.pi - abs(short_dist)  # Long path distance
            return (
                long_dist * direction
            )  # Return the long path in the correct direction

    # If the short path is valid, return the shortest distance
    return short_dist


def calc_dist_in_range(value1, value2, is_linear, valid_range):
    """
    Calculate the distances between values either linearly or rotationally based on the `is_linear` flag.
    Supports both single values and lists of values.

    This function computes the distance between `value1` and `value2` either linearly
    (if `is_linear` is True) or rotationally (if `is_linear` is False). For rotational
    calculations, angles are wrapped within the provided `valid_range`. It supports both
    individual values and lists of values for batch calculations.

    Parameters:
    value1 (float or list of floats): The first value or list of values.
    value2 (float or list of floats): The second value or list of values.
    is_linear (bool or list of bools): A boolean or list of booleans indicating whether
                                       to calculate linearly or rotationally.
    valid_range (tuple or list of tuples): A valid range or list of valid ranges
                                           for rotational calculations. The range should
                                           be a tuple for each value.

    Returns:
    float or list of floats: The calculated distance(s). If single values are provided,
                             a single distance is returned. If lists are provided,
                             a list of distances is returned.

    Raises:
    ValueError: If the length of `value1`, `value2`, `is_linear`, and `valid_range` do not match
                when they are lists.
    """
    # Check if the inputs are single values (int/float) or lists
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        # For single values, compute linear or angular distance based on the is_linear flag
        if is_linear:
            return linear_dist_in_range(value1, value2, valid_range)
        else:
            return angle_dist_in_range(value1, value2, valid_range)

    # If lists are provided, ensure all lists are of the same length
    if not (len(value1) == len(value2) == len(is_linear) == len(valid_range)):
        raise ValueError(
            "value1, value2, is_linear, and valid_range must have the same length when they are lists."
        )

    # Compute distances for each pair of values, either linearly or rotationally
    distances = []
    for v1, v2, lin, range_val in zip(value1, value2, is_linear, valid_range):
        if lin:
            distances.append(linear_dist_in_range(v1, v2, range_val))
        else:
            distances.append(angle_dist_in_range(v1, v2, range_val))

    return distances


def interpolate_q_in_range(q1, q2, joint_type, joint_ranges, n_interp):
    """
    Interpolates between two sets of joint values (`q1` and `q2`) based on joint types and ranges.

    This function generates intermediate joint configurations by interpolating between `q1`
    and `q2` for each joint. Linear interpolation is used for linear joints, and rotational
    interpolation (with wrapping) is used for rotational joints, respecting joint limits.

    Parameters:
    q1 (list or array): The starting joint values.
    q2 (list or array): The ending joint values.
    joint_type (list of bools): A list where each boolean indicates whether a joint is linear (True)
                                or rotational (False).
    joint_ranges (list of tuples): The valid range for each joint, given as a tuple (lowBound, highBound).
    n_interp (int): The number of interpolation steps (excluding the final point `q2`).

    Returns:
    numpy.ndarray: A 2D array of interpolated joint configurations. Each row represents a set
                   of joint values at an interpolated step. The array has shape `(n_interp, len(q1))`.

    Raises:
    ValueError: If `q1` and `q2` do not have the same length.
    """
    # Ensure q1 and q2 are of the same length
    if len(q1) != len(q2):
        raise ValueError("q1 and q2 must be of the same length.")

    # Calculate the valid distances between q1 and q2 for each joint, respecting joint types and ranges
    valid_distances = calc_dist_in_range(q1, q2, joint_type, joint_ranges)

    # Initialize an array to hold the interpolated joint configurations
    q_interp = np.zeros((n_interp, len(q1)))

    # Interpolate for each joint individually
    for i in range(len(q1)):
        # Generate n_interp + 1 evenly spaced points between 0 and the distance for the joint
        points = np.linspace(0, valid_distances[i], n_interp + 1)

        # Fill in the interpolated values for the current joint, excluding the last point (q2)
        for j, n in enumerate(
            points[:-1]
        ):  # Exclude the last point to avoid duplicating q2
            interpolated_value = q1[i] + n
            q_interp[j, i] = interpolated_value

    # Wrap angles if necessary for rotational joints (assuming wrap_angle_list handles multiple joints)
    q_interp = wrap_angle_list(q_interp)

    return q_interp


def round_list(input_list, decimals=2):
    """
    Recursively rounds numerical values in a list (or nested lists) to a specified number of decimal places.

    This function takes a list (which can contain nested lists and tuples) and rounds all
    numerical values (integers and floats) to the specified number of decimal places.
    Non-numerical values remain unchanged.

    Parameters:
    input_list (list): The list (or nested list) containing elements to be rounded.
    decimals (int, optional): The number of decimal places to round to. Default is 2.

    Returns:
    list: A new list with rounded numerical values. Nested lists and tuples are also handled.

    Example:
    >>> round_list([1.2345, 2.3456, [3.4567, (4.5678, 5.6789)]], 2)
    [1.23, 2.35, [3.46, (4.57, 5.68)]]
    """
    rounded_list = []

    # Iterate through each element in the input list
    for elem in input_list:
        if isinstance(elem, list):
            # If the element is a list, recursively round its elements
            rounded_list.append(round_list(elem, decimals))
        elif isinstance(elem, tuple):
            # If the element is a tuple, convert it to a list, round its elements, and convert back to a tuple
            rounded_list.append(tuple(round_list(list(elem), decimals)))
        elif isinstance(elem, (int, float)):
            # If the element is a number (int or float), round it
            rounded_list.append(round(elem, decimals))
        else:
            # If the element is neither a list, tuple, int, nor float, append it as is
            rounded_list.append(elem)

    return rounded_list


def floats_equal(a, b, tol=1e-9):
    """
    Check if two floats or lists of floats are equal within a given tolerance.

    This function compares two floating-point numbers or two lists of floating-point
    numbers and checks if they are equal within a specified tolerance. The comparison
    accounts for potential floating-point inaccuracies by allowing for small differences
    (defined by `tol`).

    Parameters:
    a (float or list of floats): First number or list of numbers to compare.
    b (float or list of floats): Second number or list of numbers to compare.
    tol (float, optional): The tolerance within which the numbers are considered equal.
                           Default is 1e-9.

    Returns:
    bool: True if the numbers or lists are equal within the given tolerance;
          False otherwise.

    Raises:
    ValueError: If the inputs are not both floats or both lists of the same length.

    Example:
    >>> floats_equal(0.123456789, 0.123456788)
    True

    >>> floats_equal([1.000000001, 2.000000001], [1.0, 2.0])
    True

    >>> floats_equal([1.0, 2.0], [1.0, 3.0])
    False
    """
    # Check if both a and b are lists
    if isinstance(a, list) and isinstance(b, list):
        # If lists have different lengths, they cannot be equal
        if len(a) != len(b):
            return False
        # Compare corresponding elements in both lists using the tolerance
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    # If both a and b are floats, compare them directly using the tolerance
    elif isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol

    # If the inputs are not both floats or lists of floats, raise an error
    else:
        raise ValueError(
            "Both inputs must be either floats or lists of floats of the same length."
        )


def nearest_neighbor(points, target_value, return_index=False):
    """
    Finds the nearest neighbor of a target value from an array of points using a KD-tree.

    This function uses a KD-tree to efficiently find the nearest neighbor of a given target value
    from a set of points. The target value and points can be in any dimensional space (1D, 2D, etc.).
    The function supports an option to return either the nearest point or the index of the nearest point.

    Parameters:
    points (np.ndarray): A 2D array of points where each row represents a point in space.
                         If a 1D array is provided, it will be reshaped into a 2D array.
    target_value (array-like): The target value for which the nearest neighbor is searched.
                               It can be a 1D or multi-dimensional array depending on the dimensionality
                               of the points.
    return_index (bool, optional): If True, the function returns the index of the nearest point
                                   in the `points` array. If False (default), the function returns
                                   the nearest point itself.

    Returns:
    np.ndarray or int: The nearest point in the array to the target value (if `return_index` is False),
                       or the index of that point (if `return_index` is True).

    Example:
    >>> points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> target = [3.1, 4.1]
    >>> nearest_neighbor(points, target)
    array([3., 4.])

    >>> nearest_neighbor(points, target, return_index=True)
    1
    """
    # Ensure that `points` is a NumPy array
    points = np.asarray(points)
    target_value = np.asarray(target_value)

    # Ensure `points` is a 2D array (n_points, n_dimensions)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    # Ensure `target_value` is a 2D array (1, n_dimensions) for querying the KD-tree
    if target_value.ndim == 1:
        target_value = target_value.reshape(1, -1)

    # Create a KDTree object using the array of points
    kdtree = KDTree(points)

    # Query the KDTree for the nearest neighbor to the target value
    _, index = kdtree.query(target_value)

    # Return either the index or the nearest point
    if return_index:
        return index[0]  # Return the index of the nearest neighbor
    else:
        return points[index][0]  # Return the nearest point itself


def flatten_list(nested_list):
    """
    Flattens a nested list (a list of lists) into a single list.

    This function takes a nested list, where each element is a list, and flattens it into
    a single list containing all the elements of the sublists.

    Parameters:
    nested_list (list of lists): A list containing other lists as elements.

    Returns:
    list: A flat list containing all the elements from the sublists.

    Example:
    >>> flatten_list([[1, 2], [3, 4], [5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    # Flatten the nested list by iterating over each sublist and each element in the sublist
    return [element for sublist in nested_list for element in sublist]


def unflatten_list(flat_list, final_list_structure):
    """
    Reshapes a flat list into a nested list structure based on a specified structure.

    This function takes a flat list and reshapes it into a nested list, where each
    sublist's length is defined by the corresponding element in `final_list_structure`.
    The sum of the elements in `final_list_structure` should match the length of the `flat_list`.

    Parameters:
    flat_list (list): The flat list that needs to be reshaped.
    final_list_structure (list of ints): A list of integers where each integer defines the
                                         length of the corresponding sublist in the resulting list.

    Returns:
    list: A reshaped list of lists, where each sublist has the specified length.

    Raises:
    ValueError: If the total size of the `final_list_structure` doesn't match the size of `flat_list`.

    Example:
    >>> flat_list = [1, 2, 3, 4, 5, 6]
    >>> final_list_structure = [2, 2, 2]
    >>> unflatten_list(flat_list, final_list_structure)
    [[1, 2], [3, 4], [5, 6]]

    >>> flat_list = [1, 2, 3, 4, 5]
    >>> final_list_structure = [2, 3]
    >>> unflatten_list(flat_list, final_list_structure)
    [[1, 2], [3, 4, 5]]
    """
    # Check if the total size of final_list_structure matches the length of flat_list
    if sum(final_list_structure) != len(flat_list):
        raise ValueError(
            "The total size of final_list_structure must match the size of flat_list."
        )

    result = []  # Initialize the result list
    start = 0  # Track the starting index of each sublist

    # Reshape the flat list based on the lengths in final_list_structure
    for length in final_list_structure:
        end = start + length  # Determine the end index for the sublist
        result.append(flat_list[start:end])  # Append the sublist
        start = end  # Update the start index for the next iteration

    return result

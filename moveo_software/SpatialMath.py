import numpy as np
import math
import AuxiliarFunctions as AF
from typing import Union, List, Tuple


def Trans(x: float, y: float, z: float) -> np.ndarray:
    """
    Creates a homogeneous transformation matrix for translation along the x, y, and z axes.

    Parameters:
    - x (float): Translation along the x-axis.
    - y (float): Translation along the y-axis.
    - z (float): Translation along the z-axis.

    Returns:
    - np.ndarray: A 4x4 numpy array representing the translation transformation matrix.
    """
    T = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    return T


def RotX(a: float) -> np.ndarray:
    """
    Creates a homogeneous transformation matrix for a rotation about the x-axis.

    Parameters:
    - a (float): Rotation angle in radians.

    Returns:
    - np.ndarray: A 4x4 numpy array representing the rotation transformation matrix about the x-axis.
    """
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(a), -math.sin(a), 0],
            [0, math.sin(a), math.cos(a), 0],
            [0, 0, 0, 1],
        ]
    )
    return Rx


def RotY(a: float) -> np.ndarray:
    """
    Creates a homogeneous transformation matrix for a rotation about the y-axis.

    Parameters:
    - a (float): Rotation angle in radians.

    Returns:
    - np.ndarray: A 4x4 numpy array representing the rotation transformation matrix about the y-axis.
    """
    Ry = np.array(
        [
            [math.cos(a), 0, math.sin(a), 0],
            [0, 1, 0, 0],
            [-math.sin(a), 0, math.cos(a), 0],
            [0, 0, 0, 1],
        ]
    )
    return Ry


def RotZ(a: float) -> np.ndarray:
    """
    Creates a homogeneous transformation matrix for a rotation about the z-axis.

    Parameters:
    - a (float): Rotation angle in radians.

    Returns:
    - np.ndarray: A 4x4 numpy array representing the rotation transformation matrix about the z-axis.
    """
    Rz = np.array(
        [
            [math.cos(a), -math.sin(a), 0, 0],
            [math.sin(a), math.cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return Rz


def TfmInverse(T: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a homogeneous transformation matrix.

    Parameters:
    - T (np.ndarray): A square homogeneous transformation matrix (typically 4x4).

    Returns:
    - np.ndarray: The inverse of the input transformation matrix.
    """
    nFil, nCol = T.shape  # Get the number of rows and columns.

    # Extract the rotation matrix R and the position vector p from the transformation matrix T.
    R = T[0 : (nFil - 1), 0 : (nCol - 1)]  # Upper-left 3x3 rotation matrix.
    p = T[0 : (nFil - 1), (nCol - 1) : nCol]  # Last column (position vector).

    # Transpose the rotation matrix R.
    Rt = np.transpose(R)

    # Compute the new position vector V.
    V = np.dot(-Rt, p)

    # Initialize the inverse transformation matrix with zeros.
    Tinv = np.zeros((nFil, nCol))

    # Copy the transposed rotation matrix to the upper-left submatrix of Tinv.
    for i in range(0, nFil - 1):
        for j in range(0, nCol - 1):
            Tinv[i, j] = Rt[i, j]

    # Copy the position vector V to the last column of Tinv.
    for k in range(0, nFil - 1):
        Tinv[k, nCol - 1] = V[k, 0]

    # Set the bottom-right element of Tinv to 1 (to maintain homogeneous transformation matrix format).
    Tinv[nFil - 1, nCol - 1] = 1

    return Tinv


def GetXYZ(T: np.ndarray) -> np.ndarray:
    """
    Extracts the X, Y, and Z coordinates from a homogeneous transformation matrix.

    Parameters:
    - T (np.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
    - np.ndarray: A 1D numpy array containing the X, Y, and Z coordinates.
    """
    X = T[0, 3]  # Extract the X coordinate.
    Y = T[1, 3]  # Extract the Y coordinate.
    Z = T[2, 3]  # Extract the Z coordinate.

    return np.array([X, Y, Z])  # Return the coordinates as a numpy array.


def GetRPY(T: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Extracts the Roll, Pitch, and Yaw (RPY) angles from a homogeneous transformation matrix.

    Parameters:
    - T (np.ndarray): A 4x4 homogeneous transformation matrix.
    - tol (float, optional): A tolerance value to handle numerical errors. Defaults to 1e-6.

    Returns:
    - np.ndarray: A 1D numpy array containing the Roll, Pitch, and Yaw angles in radians.
    """
    # Extract the rotation matrix R from the transformation matrix T.
    R = T[0:3, 0:3]

    # Extract the individual components of the rotation matrix.
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]

    # Check if the pitch angle is near the singularity (gimbal lock).
    if abs(1 - abs(r31)) > tol:
        # Standard case: pitch angle is not near a singularity.
        Pitch = math.atan2(-r31, math.sqrt(r32**2 + r33**2))
        Roll = math.atan2(r21 / math.cos(Pitch), r11 / math.cos(Pitch))
        Yaw = math.atan2(r32 / math.cos(Pitch), r33 / math.cos(Pitch))
    else:
        # Singularity case (gimbal lock): pitch is close to ±90 degrees.
        Roll = 0
        if r31 < 0:
            Pitch = math.pi / 2
            Yaw = Roll + math.atan2(r12, r13)
        else:
            Pitch = -math.pi / 2
            Yaw = -Roll + math.atan2(-r12, r13)

    return np.array([Roll, Pitch, Yaw])


def GetMatFromPose(pose: np.ndarray) -> np.ndarray:
    """
    Constructs a homogeneous transformation matrix from a combined list or array of XYZ coordinates and RPY angles.

    Parameters:
    - pose (array-like): A list or array containing [X, Y, Z, Roll, Pitch, Yaw], where XYZ are coordinates
                         and RPY are angles in radians.

    Returns:
    - np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    # Unpack the pose into position (x, y, z) and orientation (roll, pitch, yaw).
    x, y, z, roll, pitch, yaw = pose

    # Construct the homogeneous transformation matrix by combining translation and rotation matrices.
    T = Trans(x, y, z) @ RotZ(roll) @ RotY(pitch) @ RotX(yaw)

    return T


def extrct_MatrixVectorComp(matrix: np.ndarray) -> np.ndarray:
    """
    Extracts vector components from a 2x2 or 3x3 matrix.

    For a 2x2 matrix, it extracts components from the off-diagonal elements.
    For a 3x3 matrix, it extracts components from the last column.

    Parameters:
    - matrix (np.ndarray): A 2x2 or 3x3 matrix from which to extract vector components.

    Returns:
    - np.ndarray: A numpy array containing the extracted vector components.

    Raises:
    - ValueError: If the input matrix is not 2x2 or 3x3.
    """
    if matrix.shape == (2, 2):
        # Extract components for 2x2 matrix: [matrix[0, 1], matrix[1, 0]].
        x = matrix[0, 1]
        y = matrix[1, 0]
        return np.array([x, y])
    elif matrix.shape == (3, 3):
        # Extract components for 3x3 matrix: [matrix[0, 2], matrix[1, 2], matrix[2, 2]].
        x = matrix[0, 2]
        y = matrix[1, 2]
        z = matrix[2, 2]
        return np.array([x, y, z])
    else:
        raise ValueError("The matrix must be of shape 2x2 or 3x3.")


def approach(point: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """
    Adds a distance vector to a point in 6D space (x, y, z, roll, pitch, yaw).

    Parameters:
    - point (np.ndarray): A numpy array of six elements [x, y, z, roll, pitch, yaw].
    - distance (np.ndarray): A numpy array of six elements representing the distance
        to add in each corresponding dimension.

    Returns:
    - np.ndarray: A numpy array of six elements representing the new point after applying the distance.
    """
    return np.add(point, distance)


def vector(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """
    Calculate the vector from point A to point B.

    Parameters:
    - point_a (np.ndarray): The coordinates of point A. Can be 2D or 3D.
    - point_b (np.ndarray): The coordinates of point B. Can be 2D or 3D.

    Returns:
    - np.ndarray: The vector from point A to point B.

    Raises:
    - ValueError: If point_a and point_b do not have the same number of dimensions.
    """
    # Ensure both points have the same dimensions
    if point_a.shape != point_b.shape:
        raise ValueError("Both points must have the same number of dimensions")

    # Calculate the vector from point A to point B
    return point_b - point_a


def vector_norm(vector: Union[List[float], Tuple[float]]) -> float:
    """
    Compute the Euclidean norm (magnitude) of a vector.

    Parameters:
    - vector (list or tuple): The vector for which the norm is calculated.
      Can be a list or tuple of numerical components.

    Returns:
    - float: The Euclidean norm (magnitude) of the vector.
    """
    sum_of_squares = 0
    for component in vector:
        sum_of_squares += component**2
    return sum_of_squares**0.5


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    # Test Translation
    T = Trans(2, 4, -2)
    print("Translation Matrix of [2,-4,-2]:")
    print(T)
    print()

    # Test Rotation about X, Y, and Z axes
    Rx = RotX(math.radians(60))
    Ry = RotY(math.radians(60))
    Rz = RotZ(math.radians(60))
    print("Rotation Matrix X of 60º:")
    print(Rx)
    print()
    print("Rotation Matrix Y of 60º:")
    print(Ry)
    print()
    print("Rotation Matrix Z of 60º:")
    print(Rz)
    print()

    # Test Inversion of a transformation matrix
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # T = np.array([[0, 1, 0, 2], [-1, 0, 0, -5], [0, 0, 1, 3], [0, 0, 0, 1]])
    Tinv = TfmInverse(T)
    print("Inverse of Translation Matrix:")
    print(Tinv)
    print()

    # Test extraction of previous T Matrix
    RPY = GetRPY(T)
    print("Yaw, Pitch, Roll from T Matrix:")
    print(np.array(RPY))
    print()

    # Test composition of transformations and inversion
    roll = math.pi / 2
    pitch = 0
    yaw = math.radians(60)
    T_composed = RotZ(roll) @ RotY(pitch) @ RotX(yaw)
    T_composed_inv = TfmInverse(T_composed)
    print("Composed Transformation Matrix:")
    print(T_composed)
    print()
    print("Inverse of Composed Transformation Matrix:")
    print(T_composed_inv)
    print()

    # Test extraction of Roll, Pitch, and Yaw from a rotation matrix
    RPY = GetRPY(T_composed)
    print("Roll, Pitch, Yaw from Composed Matrix:")
    print(np.array(RPY))
    print()

    # Test manipulation of 6D points
    point = np.array([1, 2, 3, np.pi / 6, np.pi / 4, np.pi / 3])
    distance = np.array([0.5, 0.5, 0, 0.1, 0.1, 0])
    new_point = approach(point, distance)
    print("Original Point:")
    print(point)
    print("New Point after adding distance:")
    print(new_point)

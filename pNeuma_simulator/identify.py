import numpy as np
from numba import jit


@jit(nopython=True)
def identify(matrix: np.ndarray, image_rad: np.ndarray, ID: int) -> np.ndarray:
    """
    Identifies and replaces pixels in the matrix with the given ID based on a condition.
    Args:
        matrix (ndarray): The input matrix.
        image_rad (ndarray): The flattened array of image radii.
        ID (int): The ID to replace the pixels with.
    Returns:
        ndarray: The modified matrix with replaced pixels.
    """
    # matrix to array
    ar_matrix = matrix.flatten()
    ar_rad = image_rad.flatten()
    ar_matrix[np.where(ar_rad < 1)] = ID
    matrix = ar_matrix.reshape(matrix.shape)
    return matrix

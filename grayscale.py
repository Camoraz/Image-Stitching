import numpy as np

def to_grayscale(img):
    """Convert a BGR image to grayscale using standard formula."""
    # img assumed shape (H, W, 3)
    return np.clip(
        0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0],
        0, 255
    ).astype(np.uint8)

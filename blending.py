import numpy as np

def linear_blend(img1, img2, overlap_width):
    """Blend the left part of img1 with img2 linearly"""
    height = img1.shape[0]
    alpha = np.linspace(1, 0, overlap_width).reshape(1, overlap_width, 1)
    alpha = np.repeat(alpha, height, axis=0)
    alpha = np.repeat(alpha, 1, axis=2)

    overlap_img1 = img1[:, :overlap_width].astype(np.float32)
    overlap_img2 = img2[:, :overlap_width].astype(np.float32)

    # Exposure correction
    mean1 = np.mean(overlap_img1, axis=(0, 1), keepdims=True)
    mean2 = np.mean(overlap_img2, axis=(0, 1), keepdims=True)
    scale = mean2 / (mean1 + 1e-6)
    overlap_img1 *= scale

    blended = (alpha * overlap_img1 + (1 - alpha) * overlap_img2)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

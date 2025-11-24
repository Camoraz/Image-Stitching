# stitch_images_cpp.py
import cv2
import numpy as np
from grayscale import to_grayscale
import bfmatcher_cpp
from blending import linear_blend
from crop import crop_black_borders

def stitch_images(img1, img2):
    # Grayscale conversion
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)

    # SIFT keypoints and descriptors (still use OpenCV)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Convert descriptors to float32 if needed
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # Brute-force k-NN matching using C++ module
    matches = bfmatcher_cpp.knn_match(des1, des2, k=2)
    good_matches = bfmatcher_cpp.ratio_test(matches, ratio=0.75)

    # Extract points for homography
    src_pts = np.float32([kp1[m[0]].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[1]].pt for m in good_matches]).reshape(-1, 1, 2)

    # Homography (OpenCV)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp img1
    height, width = img2.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H, (width * 2, height))

    # Blend overlapping region
    overlap_width = width
    blended_overlap = linear_blend(warped_img1, img2, overlap_width)
    warped_img1[:, :overlap_width] = blended_overlap
    warped_img1[0:height, 0:width] = blended_overlap

    # Crop black borders
    result = crop_black_borders(warped_img1)
    return result

if __name__ == "__main__":
    img1 = cv2.imread('img/P1011370.JPG')
    img2 = cv2.imread('img/P1011371.JPG')

    panorama = stitch_images(img1, img2)
    cv2.imwrite('panorama_final_cpp.jpg', panorama)

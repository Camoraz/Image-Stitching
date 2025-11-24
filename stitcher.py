import cv2
import numpy as np

def stitch_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = img2.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, H, (width * 2, height))

    # Automatic overlap detection (simple method: use width of img2)
    overlap_width = width

    # Linear blending mask
    alpha = np.linspace(1, 0, overlap_width).reshape(1, overlap_width, 1)
    alpha = np.repeat(alpha, height, axis=0)
    alpha = np.repeat(alpha, 1, axis=2)

    overlap_warped = warped_img1[:, :overlap_width]
    overlap_img2 = img2.copy()

    # Exposure correction (simple scaling)
    mean_warped = np.mean(overlap_warped, axis=(0, 1), keepdims=True)
    mean_img2 = np.mean(overlap_img2, axis=(0, 1), keepdims=True)
    scale = mean_img2 / (mean_warped + 1e-6)
    overlap_warped = np.clip(overlap_warped * scale, 0, 255).astype(np.uint8)

    # Blend
    blended_overlap = (alpha * overlap_warped + (1 - alpha) * overlap_img2).astype(np.uint8)
    warped_img1[:, :overlap_width] = blended_overlap
    warped_img1[0:height, 0:width] = blended_overlap

    # Crop black borders
    gray_warped = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    result = warped_img1[y:y+h, x:x+w]

    return result

img1 = cv2.imread('img/P1011370.JPG')
img2 = cv2.imread('img/P1011371.JPG')

panorama = stitch_images(img1, img2)
cv2.imwrite('panorama_final.jpg', panorama)

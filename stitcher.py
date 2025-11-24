import cv2
import numpy as np
from grayscale import to_grayscale
import bfmatcher_cpp
from blending import linear_blend
from crop import crop_black_borders


def stitch_images(img1, img2):

    # Grayscale
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # BF matcher
    matches = bfmatcher_cpp.knn_match(des1, des2, k=2)
    good = bfmatcher_cpp.ratio_test(matches, ratio=0.75)

    if len(good) < 4:
        print("Not enough good matches")
        return None

    src_pts = np.float32([kp1[m[0]].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[1]].pt for m in good]).reshape(-1, 1, 2)

    # Homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ---- FIX 1: Compute output canvas bounding box ----
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    all_corners = np.concatenate((warped_corners,
                                  np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)),
                                 axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 10)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 10)

    # translation if negative coords exist
    translation = [-xmin, -ymin]
    translate_matrix = np.array([[1, 0, translation[0]],
                                 [0, 1, translation[1]],
                                 [0, 0, 1]], dtype=np.float32)

    # ---- FIX 2: Warp img1 into final canvas ----
    result = cv2.warpPerspective(img1, translate_matrix @ H, (xmax - xmin, ymax - ymin))

    # ---- FIX 3: Paste img2 properly ----
    result[translation[1]:translation[1] + h2,
           translation[0]:translation[0] + w2] = img2

    # ---- FIX 4: Blend ONLY the overlapping area ----
    mask1 = (result > 0).any(axis=2)
    mask2 = np.zeros_like(mask1)
    mask2[translation[1]:translation[1] + h2,
          translation[0]:translation[0] + w2] = True

    overlap_mask = mask1 & mask2
    ys, xs = np.where(overlap_mask)

    for y, x in zip(ys, xs):
        alpha = 0.5
        result[y, x] = (alpha * result[y, x] + (1 - alpha) * img2[y - translation[1], x - translation[0]])

    # Crop output
    result = crop_black_borders(result)

    return result


if __name__ == "__main__":
    img1 = cv2.imread('img/room1.jpg')
    img2 = cv2.imread('img/room2.jpg')

    panorama = stitch_images(img1, img2)
    cv2.imwrite('panorama_final_cpp.jpg', panorama)

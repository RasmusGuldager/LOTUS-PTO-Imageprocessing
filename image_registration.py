import cv2
import numpy as np


class ImageRegistration:
    def __init__(self, n_features=2000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)


        self.ref_kp, self.ref_des, self.ref_shape = None, None, None
        self.last_h = np.eye(3)  # Initial identity matrix
        self.last_match_count = 0

    def set_reference(self, img_gray):
        self.ref_shape = img_gray.shape
        self.ref_kp, self.ref_des = self.sift.detectAndCompute(img_gray, None)

    def align(self, target_gray):
        kp, des = self.sift.detectAndCompute(target_gray, None)
        matches = self.matcher.knnMatch(self.ref_des, des, k=2)

        good = [m for m, n in matches if m.distance < 0.6 * n.distance]
        self.last_match_count = len(good)

        if len(good) > 15:
            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
            self.last_h = H
            print(f"Found {len(good)} good matches. Homography computed.")
        else:
            H = self.last_h  # Fallback to previous known good alignment
            print(f"Only {len(good)} matches found. Using last known homography.")

        warped = cv2.warpPerspective(
            target_gray, H, (self.ref_shape[1], self.ref_shape[0])
        )
        return warped, H

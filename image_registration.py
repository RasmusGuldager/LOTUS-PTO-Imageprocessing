import cv2
import numpy as np
import os
from preprocessing import Preprocessor
from image_provider import ImageProvider


class ImageRegistration:
    def __init__(self, n_features=2000):
        # We use SIFT. It handles reflections and small shifts underwater very well.
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        
        # Flann-based matcher is faster than BFMatcher for large feature sets
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Store the reference image data once, reuse many times
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_shape = None

    def set_reference_image(self, ref_img_gray: np.ndarray):
        """Processes and stores features for the reference frame (Baseline)."""
        self.ref_shape = ref_img_gray.shape
        self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(ref_img_gray, None)
        print(f"Reference established. Found {len(self.ref_keypoints)} stable features.")

    def register_image(self, target_img_gray, good_match_ratio=0.5, ransac_thresh=5.0):
        """Aligns target_img_gray to the stored reference. Returns aligned image."""
        if self.ref_descriptors is None:
            raise ValueError("Must set reference image before registering.")

        target_kp, target_desc = self.sift.detectAndCompute(target_img_gray, None)
        matches = self.matcher.knnMatch(self.ref_descriptors, target_desc, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return target_img_gray, None

        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)

        aligned_img = cv2.warpPerspective(target_img_gray, H_matrix, (self.ref_shape[1], self.ref_shape[0]))
        
        return aligned_img, H_matrix

    @staticmethod
    def generate_difference_heatmap(ref_aligned, target_aligned):
        """Generates a visualization of changes between two aligned images."""
        # Simple absolute difference (pixel-by-pixel subtraction)
        diff = cv2.absdiff(ref_aligned, target_aligned)
        
        # Apply normalization just for visualization so we can see small changes
        diff_viz = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply a colormap (e.g., JET) to make it look like a heatmap
        heatmap = cv2.applyColorMap(diff_viz, cv2.COLORMAP_JET)
        
        return heatmap


def test_real_movement_registration(folder_path):
    print("--- Testing Real-World Camera Movement ---")
    provider = ImageProvider(folder_path, chunk_size=2)
    provider.get_next_chunk()
    
    # 1. Hent billeder
    ref_path = provider.get_next_chunk()[0]
    ref_img = cv2.imread(ref_path)
    ref_flat = Preprocessor.flatten_illumination(ref_img, sigma=70)

    for _ in range(1): provider.get_next_chunk()
    
    target_path = provider.get_next_chunk()[0]
    target_img = cv2.imread(target_path)
    target_flat = Preprocessor.flatten_illumination(target_img, sigma=70)

    # 2. Registration
    reg_engine = ImageRegistration(n_features=3000)
    reg_engine.set_reference_image(ref_flat)
    aligned_target, H_matrix = reg_engine.register_image(target_flat)

    if H_matrix is None:
        print("Registration Failed!"); return

    # 3. Visualiseringer
    h, w = ref_flat.shape
    heatmap_error = ImageRegistration.generate_difference_heatmap(ref_flat, target_flat)
    heatmap_success = ImageRegistration.generate_difference_heatmap(ref_flat, aligned_target)

    # Checkerboard
    checkerboard = np.zeros((h, w), dtype=np.uint8)
    tile_size = 200 # Større tiles gør det nemmere at se fejl
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                checkerboard[y:y+tile_size, x:x+tile_size] = ref_flat[y:y+tile_size, x:x+tile_size]
            else:
                checkerboard[y:y+tile_size, x:x+tile_size] = aligned_target[y:y+tile_size, x:x+tile_size]
    
    checker_bgr = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
    aligned_bgr = cv2.cvtColor(aligned_target, cv2.COLOR_GRAY2BGR)

    # --- FIX AF LAYOUT ---
    # Top række: To heatmaps (Bredde: w + w = 2w)
    top_row = np.hstack((heatmap_error, heatmap_success))
    
    # Bund række: Checkerboard + Det rettede billede (Bredde: w + w = 2w)
    bottom_row = np.hstack((checker_bgr, aligned_bgr))

    # Nu passer bredden! (2w mod 2w)
    combined = np.vstack((top_row, bottom_row))

    # --- RESIZE TIL SKÆRM ---
    # Dine billeder er åbenbart gigantiske (3548px brede hver!)
    # Vi skalerer det ned til noget, der kan være på en laptop-skærm.
    screen_res = (1600, 900) # Eller hvad der passer dig
    scale_width = screen_res[0] / combined.shape[1]
    scale_height = screen_res[1] / combined.shape[0]
    scale = min(scale_width, scale_height)
    
    new_size = (int(combined.shape[1] * scale), int(combined.shape[0] * scale))
    combined_small = cv2.resize(combined, new_size)

    # Tilføj tekst efter resize for at holde den læsbar
    cv2.putText(combined_small, "VENSTRE: Uden alignment | HOJRE: Med alignment", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Alignment Verification", combined_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    PATH = './lotus_kristineberg_prototype/images'
    test_real_movement_registration(PATH)
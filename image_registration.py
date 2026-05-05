import cv2
import numpy as np
import os

from preprocessing import Preprocessor
from image_provider import ImageProvider
from visualizer import Visualizer


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

    def register_image(self, target_img_gray, good_match_ratio=0.6, ransac_thresh=5.0):
        """Aligns target_img_gray to the stored reference. Returns aligned image."""
        if self.ref_descriptors is None:
            raise ValueError("Must set reference image before registering.")

        target_kp, target_desc = self.sift.detectAndCompute(target_img_gray, None)
        matches = self.matcher.knnMatch(self.ref_descriptors, target_desc, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

        self.last_match_count = len(good_matches)

        if len(good_matches) < 10:
            return target_img_gray, None

        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)

        aligned_img = cv2.warpPerspective(target_img_gray, H_matrix, (self.ref_shape[1], self.ref_shape[0]))
        
        return aligned_img, H_matrix


def run_alignment_time_series(folder_path):
    # 1. Setup
    provider = ImageProvider(folder_path, chunk_size=2)
    reg_engine = ImageRegistration(n_features=3000)
    
    # 2. Sæt MASTER REFERENCE (Baseline)
    ref_chunk = provider.get_next_chunk()
    if not ref_chunk:
        print("Ingen billeder fundet."); return
        
    ref_img = cv2.imread(ref_chunk[0])
    ref_flat = Preprocessor.flatten_illumination(ref_img, sigma=70)
    reg_engine.set_reference_image(ref_flat)

    window_name = "Time-Series Alignment Browser"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Master Reference sat: {os.path.basename(ref_chunk[0])}")
    print("Controls: SPACE/Any key = Næste | Q = Afslut")

    # 3. Loop gennem resten af serien
    while True:
        chunk = provider.get_next_chunk()
        if chunk is None:
            print("Slut på tidsserien."); break
        
        target_path = chunk[0]
        target_img = cv2.imread(target_path)
        target_flat = Preprocessor.flatten_illumination(target_img, sigma=70)
        
        # Registrering mod master baseline
        aligned, H = reg_engine.register_image(target_flat)
        
        if H is not None:
            # --- VI BYGGER SELV VORES FRAMES HER ---
            # Checkerboard til at tjekke kanter
            checker = Visualizer.create_checkerboard(ref_flat, aligned)
            
            # Heatmaps til at se "støj" vs "rigtig ændring"
            h_before = Visualizer.generate_difference_heatmap(ref_flat, target_flat)
            h_after = Visualizer.generate_difference_heatmap(ref_flat, aligned)

            # Sammensæt listen af frames og titler
            frames_to_show = [target_flat, checker, h_before, h_after]
            titles = [
                f"Billede: {os.path.basename(target_path)}",
                f"Checkerboard (Matches: {reg_engine.last_match_count})",
                "Før Alignment (Movement Error)",
                "Efter Alignment (Potential Growth)"
            ]

            # --- BRUG DEN UNIVERSELLE METODE ---
            # Her styrer vi dashboardet via display_height som ønsket
            dashboard = Visualizer.create_dashboard(
                frames_to_show, 
                titles, 
                rows=2, cols=2, 
                display_height=1950 # Tilpas denne til din skærm
            )
            
            cv2.imshow(window_name, dashboard)
        else:
            print(f"Alignment fejlede for: {os.path.basename(target_path)}")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    PATH = './lotus_kristineberg_prototype/images'
    run_alignment_time_series(PATH)
import cv2
import numpy as np
import os

from visualizer import Visualizer
from preprocessing import Preprocessor


def run_comprehensive_test(folder_path):
    all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    
    baseline_mertens = None
    baseline_flat = None
    baseline_hsv = None
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    
    for i in range(0, len(all_files), 5):
        chunk = all_files[i:i+5]
        if len(chunk) < 5: break
            
        raw_ref = cv2.imread(chunk[2])
        
        # 1. Processing
        mertens_img = Preprocessor.calculate_exposure_fusion(chunk)
        flat_img = Preprocessor.flatten_illumination(mertens_img)
        clahe_img = Preprocessor.apply_clahe(mertens_img)
        current_hsv = cv2.cvtColor(mertens_img, cv2.COLOR_BGR2HSV)
        
        # 2. Establish baselines
        if baseline_mertens is None:
            baseline_mertens = cv2.cvtColor(mertens_img, cv2.COLOR_BGR2GRAY)
            baseline_flat = flat_img.copy()
            baseline_hsv = current_hsv.copy()
            continue
            
        # 3. Temporal Differences (Stability check)
        mertens_gray = cv2.cvtColor(mertens_img, cv2.COLOR_BGR2GRAY)
        diff_mertens = cv2.absdiff(baseline_mertens, mertens_gray)
        _, mask_mertens = cv2.threshold(diff_mertens, 30, 255, cv2.THRESH_BINARY)
        
        diff_flat = cv2.absdiff(baseline_flat, flat_img)
        _, mask_flat = cv2.threshold(diff_flat, 30, 255, cv2.THRESH_BINARY)

        # 4. HSV Difference
        diff_h = cv2.absdiff(baseline_hsv[:,:,0], current_hsv[:,:,0])
        diff_s = cv2.absdiff(baseline_hsv[:,:,1], current_hsv[:,:,1])
        hs_diff = cv2.add(diff_h, diff_s)
        _, hs_mask = cv2.threshold(hs_diff, 50, 255, cv2.THRESH_BINARY)
        _, v_mask = cv2.threshold(current_hsv[:,:,2], 40, 255, cv2.THRESH_BINARY)
        clean_hs_mask = cv2.bitwise_and(hs_mask, v_mask)
        
        # 5. MOG2 and Canny
        mog_mask = backSub.apply(mertens_img, learningRate=0.01)
        _, mog_mask_clean = cv2.threshold(mog_mask, 250, 255, cv2.THRESH_BINARY)
        
        edges = cv2.Canny(flat_img, 50, 150)
        
        # 6. Apply Colormaps for visualization
        heat_mertens = cv2.applyColorMap(mask_mertens, cv2.COLORMAP_JET)
        heat_flat = cv2.applyColorMap(mask_flat, cv2.COLORMAP_JET)
        heat_hsv = cv2.applyColorMap(clean_hs_mask, cv2.COLORMAP_JET)
        heat_mog = cv2.applyColorMap(mog_mask_clean, cv2.COLORMAP_JET)
        
        # 7. Construct Dashboard
        frames = [
            raw_ref, mertens_img, flat_img,
            clahe_img, heat_mertens, heat_flat,
            heat_mog, edges, heat_hsv
        ]
        
        titles = [
            "1. Raw (Ref)", "2. Mertens Fusion", "3. Flattened",
            "4. CLAHE", "5. Diff (Mertens)", "6. Diff (Flattened)",
            "7. MOG2 Mask", "8. Canny Edges", "9. HSV Diff"
        ]
        
        display_img = Visualizer.create_3x3_dashboard(frames, titles, display_width=1800)
        cv2.imshow("Comprehensive Pipeline Analysis", display_img)
        
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_comprehensive_test('./exposure/images')
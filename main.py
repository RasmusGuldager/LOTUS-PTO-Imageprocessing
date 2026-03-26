# HSV is resistant to intensity changes, but vulnerable to spectral changes.
# Spektral variation


# DTU study:
# Image processing: Checkboard calibration -> median stacking -> mertens algorithm -> gaussian smoothing (250 sigma) -> 
# Pixel classification: Ilastik ->
import cv2
import numpy as np
import os

# --- MODUL 1: BILLEDFORBEDRING (MERTENS & BLUR) ---

def calculate_exposure_fusion(image_paths):
    """Fletter billeder med Mertens algoritmen for perfekt belysning."""
    img_list = [cv2.imread(f) for f in image_paths]
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    return np.clip(res_mertens * 255, 0, 255).astype('uint8')

def simple_noise_reduction(image):
    """Anvender en let Gaussian blur for at fjerne sensor-støj."""
    return cv2.GaussianBlur(image, (5, 5), 0)


# --- MODUL 2: VISUALISERING (DASHBOARD) ---

def create_analysis_dashboard(frames, titles, display_width=1280):
    """Samler billeder i en 2x2 matrix til præsentation."""
    processed = []
    quad_w = display_width // 2
    
    for img, title in zip(frames, titles):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Normalisering sikrer, at vi kan se de mørke områder på skærmen
        img_viz = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        h, w = img_viz.shape[:2]
        aspect = h / w
        img_viz = cv2.resize(img_viz, (quad_w, int(quad_w * aspect)))
        
        cv2.putText(img_viz, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed.append(img_viz)

    top = np.hstack((processed[0], processed[1]))
    bottom = np.hstack((processed[2], processed[3]))
    return np.vstack((top, bottom))


# --- HOVEDPROGRAM (STABILITETS-TEST) ---

def test_drastic_light_pipeline(folder_path):
    all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    
    baseline_hsv = None
    
    for i in range(0, len(all_files), 5):
        chunk = all_files[i:i+5]
        if len(chunk) < 5: break
            
        # 1. Behandl billedet (Flet og rens)
        mertens_img = calculate_exposure_fusion(chunk)
        processed_img = simple_noise_reduction(mertens_img)
        
        # 2. Konverteringer
        current_hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        current_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Sæt Baseline (Kun første gang)
        if baseline_hsv is None:
            baseline_hsv = current_hsv.copy()
            print("Baseline etableret! Tester mod drastiske lysskift...")
            continue 
            
        # --- ANALYSE A: HSV Forskellen (Med V-Gate) ---
        diff_h = cv2.absdiff(baseline_hsv[:,:,0], current_hsv[:,:,0])
        diff_s = cv2.absdiff(baseline_hsv[:,:,1], current_hsv[:,:,1])
        hs_diff = cv2.add(diff_h, diff_s)
        
        # Sæt en høj tolerance for farvespring
        _, hs_mask = cv2.threshold(hs_diff, 60, 255, cv2.THRESH_BINARY) 
        
        # DEN MAGISKE V-GATE: Ignorer alle pixels, der er mørke (Value < 50)
        _, v_mask = cv2.threshold(current_hsv[:,:,2], 50, 255, cv2.THRESH_BINARY)
        
        # Behold kun farveændringer, der IKKE befinder sig i mørke
        clean_hs_mask = cv2.bitwise_and(hs_mask, v_mask)
        clean_hs_mask = cv2.GaussianBlur(clean_hs_mask, (5,5), 0)
        heatmap_hs = cv2.applyColorMap(clean_hs_mask, cv2.COLORMAP_JET)

        # --- ANALYSE B: Struktur/Kant (Bevis for stabilitet) ---
        # Her udtager vi kanterne direkte. Du vil se dine formler lyse skarpt op.
        # Selv hvis rummet bliver mørkt, vil disse kanter forblive synlige og stabile.
        current_edges = cv2.Laplacian(current_gray, cv2.CV_64F)
        current_edges = cv2.convertScaleAbs(current_edges)
        
        # Gør kanterne mere bastante, så chefen let kan se dem
        _, strong_edges = cv2.threshold(current_edges, 30, 255, cv2.THRESH_BINARY)
        heatmap_edge = cv2.applyColorMap(strong_edges, cv2.COLORMAP_JET)

        # --- VISUALISERING ---
        raw_ref = cv2.imread(chunk[2])
        display_img = create_analysis_dashboard(
            frames=[raw_ref, processed_img, heatmap_hs, heatmap_edge],
            titles=[
                "1. Raw Image (Reference)", 
                "2. Processed (Mertens)", 
                "3. HSV Diff (V-Gate Active)", 
                "4. Current Edges (Structure)"
            ]
        )

        cv2.imshow("Drastisk Lys-Check", display_img)
        
        # Tiden sat til 100ms for et glidende "timelapse"-look
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Husk at opdatere mappenavnet, hvis det er nødvendigt
    test_drastic_light_pipeline('./exposure/images')
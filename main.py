# HSV is resistant to intensity changes, but vulnerable to spectral changes.
# Spektral variation


# DTU study:
# Image processing: Checkboard calibration -> median stacking -> mertens algorithm -> gaussian smoothing (250 sigma) -> 
# Pixel classification: Ilastik ->
import cv2
import numpy as np
import os

# --- MODUL 1: BILLEDFORBEDRING (MERTENS) ---
def calculate_exposure_fusion(image_paths):
    img_list = [cv2.imread(f) for f in image_paths]
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    return np.clip(res_mertens * 255, 0, 255).astype('uint8')

# --- MODUL 2: LYS-UDJÆVNING (FLATTENING) ---
def flatten_illumination(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    light_map = cv2.GaussianBlur(gray, (0, 0), sigmaX=50, sigmaY=50)
    
    gray_float = gray.astype(np.float32)
    light_map_float = np.maximum(light_map.astype(np.float32), 1.0)
    
    flat_float = gray_float / light_map_float
    return cv2.normalize(flat_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- MODUL 3: DASHBOARD ---
def create_analysis_dashboard(frames, titles, display_width=1280):
    processed = []
    quad_w = display_width // 2
    for img, title in zip(frames, titles):
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_viz = cv2.resize(img, (quad_w, int(quad_w * (img.shape[0] / img.shape[1]))))
        cv2.putText(img_viz, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processed.append(img_viz)
    return np.vstack((np.hstack((processed[0], processed[1])), np.hstack((processed[2], processed[3]))))

# --- HOVEDPROGRAM ---
def test_ultimate_pipeline(folder_path):
    all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    baseline_flat = None
    
    for i in range(0, len(all_files), 5):
        chunk = all_files[i:i+5]
        if len(chunk) < 5: break
            
        # 1. MERTENS FUSION (Få detaljer i hjørnerne)
        mertens_img = calculate_exposure_fusion(chunk)
        
        # 2. FLATTENING (Fjern skiftende lys og skygger)
        current_flat = flatten_illumination(mertens_img)
        
        # Sæt baseline ved første frame
        if baseline_flat is None:
            baseline_flat = current_flat.copy()
            print("Baseline etableret! Tester den fulde pipeline...")
            continue
            
        # 3. STABILITETS-TEST (AbsDiff)
        diff_flat = cv2.absdiff(baseline_flat, current_flat)
        _, stable_mask = cv2.threshold(diff_flat, 30, 255, cv2.THRESH_BINARY)
        heatmap = cv2.applyColorMap(stable_mask, cv2.COLORMAP_JET)

        # 4. STRUKTUR-TEST (Canny)
        edges = cv2.Canny(current_flat, 50, 150)
        edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_AUTUMN)

        # 5. VISUALISERING
        # Vi viser Mertens i Panel 1, så du kan se hvor flot det er, 
        # inden det bliver "smadret fladt" i Panel 2.
        display_img = create_analysis_dashboard(
            frames=[mertens_img, current_flat, heatmap, edges_colored],
            titles=[
                "1. Mertens Fusion (Flotte detaljer)", 
                "2. Flattened (Klar til analyse)", 
                "3. Stabilt Heatmap", 
                "4. Canny Edges"
            ]
        )

        cv2.imshow("Ultimate Pipeline Test", display_img)
        if cv2.waitKey(200) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_ultimate_pipeline('./exposure/images')
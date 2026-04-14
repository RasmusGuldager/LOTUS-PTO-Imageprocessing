import cv2
import numpy as np
import os

from visualizer import Visualizer
from preprocessing import Preprocessor
from image_provider import ImageProvider



def simple_pipeline(folder_path):
    # 1. Initialize the provider with chunks of 5
    provider = ImageProvider(folder_path, chunk_size=2)
    
    print("--- Starting Browser ---")
    print("Controls: Press 'Q' to Quit | Press any other key for the next chunk")

    while True:
        # 2. Get next chunk
        image_paths = provider.get_next_chunk()
        
        # Break if we run out of images
        if image_paths is None:
            print("Reached the end of the folder.")
            break

        # 3. Simple Processing Chain
        # Step A: Load one raw image for reference
        raw_img = cv2.imread(image_paths[0])
        
        # Step B: Merge the 5 images (Exposure Fusion)
        fused = Preprocessor.calculate_exposure_fusion(image_paths)
        
        # Step C: Flatten the light
        flat = Preprocessor.flatten_illumination(fused, sigma=50)
        
        # Step D: Final Contrast boost
        # We convert 'flat' to BGR because your apply_clahe expects BGR
        enhanced = Preprocessor.apply_clahe(cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR))

        blurred = Preprocessor.gaussian_blur(fused, sigma=2)

        # 4. Visualize the 4 steps in a 2x2 grid
        frames = [raw_img, fused, flat, enhanced, blurred]
        titles = ["1. Raw Sample", "2. Fused Chunk", "3. Light Flattened", "4. CLAHE Enhanced", "5. Gaussian Blurred"]
        
        dashboard = Visualizer.create_dashboard(frames, titles)

        # 5. Interaction Logic
        cv2.imshow("Underwater Processing Browser", dashboard)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    PATH_TO_IMAGES = './lotus_kristineberg_prototype/images' 
    #PATH_TO_IMAGES = './exposure/images' 
    
    
    if os.path.exists(PATH_TO_IMAGES):
        simple_pipeline(PATH_TO_IMAGES)
    else:
        print(f"Error: Folder '{PATH_TO_IMAGES}' not found.")
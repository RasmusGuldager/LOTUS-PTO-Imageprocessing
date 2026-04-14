import cv2
import numpy as np

class Preprocessor:
    @staticmethod
    def calculate_exposure_fusion(image_paths):
        # Use Mertens exposure fusion to combine images
        img_list = [cv2.imread(f) for f in image_paths]
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        return np.clip(res_mertens * 255, 0, 255).astype('uint8')
    
    @staticmethod
    def median_stack(image_paths):
        # Apply median stacking for all three rgb channels
        img_list = [cv2.imread(f) for f in image_paths]
        stack = np.stack(img_list, axis=3)
        median_img = np.median(stack, axis=3).astype('uint8')
        return median_img

    @staticmethod
    def average_stack(image_paths):
        # Apply average stacking for all three rgb channels
        img_list = [cv2.imread(f) for f in image_paths]
        stack = np.stack(img_list, axis=3)
        avg_img = np.mean(stack, axis=3).astype('uint8')
        return avg_img
    
    @staticmethod
    def gaussian_blur(image, sigma=50):
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def flatten_illumination(img, sigma=50):
        # Check if the image is already grayscale
        if len(img.shape) == 2:
            gray = img
        else:
            # Convert to grayscale for light flattening
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray_float = gray.astype(np.float32)
        
        # Lav light map
        light_map = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        light_map_float = np.maximum(light_map.astype(np.float32), 1.0)
        
        # Division for at udjævne lyset
        flat_float = gray_float / light_map_float
        return cv2.normalize(flat_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def apply_clahe(img_bgr, clip_limit=2.0, tile_grid=(8,8)):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(gray)
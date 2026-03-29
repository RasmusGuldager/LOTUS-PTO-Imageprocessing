import cv2
import numpy as np

class Preprocessor:
    @staticmethod
    def calculate_exposure_fusion(image_paths):
        img_list = [cv2.imread(f) for f in image_paths]
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        return np.clip(res_mertens * 255, 0, 255).astype('uint8')

    @staticmethod
    def flatten_illumination(img_bgr, sigma=50):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        light_map = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        gray_float = gray.astype(np.float32)
        light_map_float = np.maximum(light_map.astype(np.float32), 1.0)
        
        flat_float = gray_float / light_map_float
        return cv2.normalize(flat_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def apply_clahe(img_bgr, clip_limit=2.0, tile_grid=(8,8)):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(gray)
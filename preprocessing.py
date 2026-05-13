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
    def downscale(img, scale=0.3):
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def process_pair(pair_dict, flash_threshold=55):
        """Phase 1: Handles flash failure and merges."""
        img_dim = cv2.imread(pair_dict['dim'])
        img_bright = cv2.imread(pair_dict['bright'])

        # Check if flash worked
        if img_bright is not None and np.mean(img_bright) > flash_threshold:
            merge_mertens = cv2.createMergeMertens()
            res = merge_mertens.process([img_dim, img_bright])
            return np.clip(res * 255, 0, 255).astype('uint8')
        else:
            return img_dim # Fallback to dim if flash failed

    @staticmethod
    def flatten_illumination(img_gray, sigma=50):
        img_float = img_gray.astype(np.float32)
        blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Division normalization: Result = (Original / Blur)
        flat = img_float / np.maximum(blur, 1.0)
        return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def apply_clahe(gray_img, clip_limit=2.0):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        return clahe.apply(gray_img)
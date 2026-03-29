import numpy as np
import cv2


class Visualizer:
    @staticmethod
    def create_3x3_dashboard(frames, titles, display_width=1920):
        """Creates a 3x3 grid of images. Pads with black frames if len < 9."""
        processed = []
        cell_w = display_width // 3
        
        # Pad lists if they are shorter than 9
        while len(frames) < 9:
            frames.append(np.zeros_like(frames[0]))
            titles.append("Empty")

        for img, title in zip(frames[:9], titles[:9]):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            img_viz = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Calculate aspect ratio based resize
            h, w = img_viz.shape[:2]
            aspect = h / w
            cell_h = int(cell_w * aspect)
            img_viz = cv2.resize(img_viz, (cell_w, cell_h))
            
            cv2.putText(img_viz, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            processed.append(img_viz)

        # Build the 3x3 grid
        row1 = np.hstack((processed[0], processed[1], processed[2]))
        row2 = np.hstack((processed[3], processed[4], processed[5]))
        row3 = np.hstack((processed[6], processed[7], processed[8]))
        
        return np.vstack((row1, row2, row3))


    @staticmethod
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
import cv2
import numpy as np


class Visualizer:
    @staticmethod
    def create_dashboard(frames, titles, rows=2, cols=2, display_height=1080):
        """
        Opretter et grid (f.eks. 2x2 eller 3x3).
        Tvinger alle billeder til samme størrelse for at undgå vstack/hstack fejl.
        """
        num_cells = rows * cols
        processed = []

        # Beregn præcis cellestørrelse baseret på den ønskede totalbredde
        cell_h = display_height // rows

        # Vi tager udgangspunkt i det første billede for at finde aspect ratio
        h_orig, w_orig = frames[0].shape[:2]
        aspect = w_orig / h_orig
        cell_w = int(cell_h * aspect)

        # Klargør frames (fyld op med sorte billeder hvis der mangler nogen)
        temp_frames = list(frames)
        temp_titles = list(titles)
        while len(temp_frames) < num_cells:
            temp_frames.append(np.zeros((h_orig, w_orig, 3), dtype=np.uint8))
            temp_titles.append("Empty")

        for i in range(num_cells):
            img = temp_frames[i].copy()
            title = temp_titles[i]

            # Konverter til farve hvis det er grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Normalisering (gør det synligt)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Resize til den præcise cellestørrelse
            img_resized = cv2.resize(img, (cell_w, cell_h))

            # Tilføj tekst
            cv2.putText(
                img_resized,
                title,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            processed.append(img_resized)

        # Saml rækkerne dynamisk
        grid_rows = []
        for r in range(rows):
            row_start = r * cols
            row_data = np.hstack(processed[row_start : row_start + cols])
            grid_rows.append(row_data)

        return np.vstack(grid_rows)

    @staticmethod
    def generate_difference_heatmap(ref, target):
        """Skaber et heatmap der viser forskellen mellem to billeder."""
        diff = cv2.absdiff(ref, target)
        diff_viz = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(diff_viz, cv2.COLORMAP_JET)

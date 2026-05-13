import cv2
import numpy as np
import os

from visualizer import Visualizer
from preprocessing import Preprocessor
from image_provider import ImageProvider
from image_registration import ImageRegistration


def process_images(data_path, output_path):
    # Setup
    provider = ImageProvider(data_path)
    registration = ImageRegistration(n_features=3000)
    os.makedirs(output_path, exist_ok=True)

    #Establish Baseline (Picture set 0)
    first_pair = provider.get_pair(2)
    master_img = Preprocessor.process_pair(first_pair)
    master_img = Preprocessor.downscale(master_img, scale=0.3)
    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    registration.set_reference(master_gray)

    for i in range(3, len(provider)):
        pair = provider.get_pair(i)
        ts = provider.timestamps[i]

        frame = Preprocessor.process_pair(pair)
        frame_small = Preprocessor.downscale(frame)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        aligned, H = registration.align(gray)

        flat = Preprocessor.flatten_illumination(aligned)
        final = Preprocessor.apply_clahe(flat)

        cv2.imwrite(f"{output_path}/aligned_{ts}.png", final)

        if i % 10 == 0:
            print(f"Processed: {ts} | Matches: {getattr(registration, 'last_match_count', 'N/A')}")


if __name__ == "__main__": 
    process_images('./images', './processed_output')

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


class ImageOverlayProcessor:
    def __init__(self, original_folder, mask_folder, output_folder, all_consider='', image_count=0):
        self.original_folder = original_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.all_consider = all_consider
        self.image_count = image_count
        self.valid_extensions = ('.png', '.jpg', '.jpeg')
        ensure_directory(self.output_folder)
        self.original_images = self._filter_original_images()

    def _filter_original_images(self):
        all_images = sorted(
            [img for img in os.listdir(self.original_folder) if img.lower().endswith(self.valid_extensions)]
        )
        if self.all_consider:
            return all_images
        filtered_images = []
        count = 0
        for img in all_images:
            if img.split('_')[0] == self.all_consider:
                if count >= self.image_count:
                    filtered_images.append(img)
                count += 1
        return filtered_images

    def load_image_and_mask(self, image_name):
        original_image_path = os.path.join(self.original_folder, image_name)
        mask_image_name = os.path.splitext(image_name)[0] + '.png'
        mask_image_path = os.path.join(self.mask_folder, mask_image_name)
        if not os.path.exists(mask_image_path):
            return None, None
        original_image = cv2.imread(original_image_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        return original_image, mask_image

    def overlay_mask_on_image(self, original_image, mask_image):
        if len(mask_image.shape) == 2:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(original_image, 0.5, mask_image, 0.5, 0)

    def process_image(self, img_name):
        original_image, mask_image = self.load_image_and_mask(img_name)
        if original_image is not None and mask_image is not None:
            combined_image = self.overlay_mask_on_image(original_image, mask_image)
            output_image_path = os.path.join(self.output_folder, img_name)
            cv2.imwrite(output_image_path, combined_image)

    def process_all_images(self):
        with tqdm(total=len(self.original_images), desc="Processing Images") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.process_image, img_name): img_name for img_name in self.original_images}
                for future in futures:
                    future.result()
                    pbar.update(1)

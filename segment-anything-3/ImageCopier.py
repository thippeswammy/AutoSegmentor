import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


class ImageCopier:
    def __init__(self, original_folder, mask_folder, overlap_images_folder, output_original_folder, output_mask_folder):
        self.original_folder = original_folder
        self.mask_folder = mask_folder
        self.overlap_images_folder = overlap_images_folder
        self.output_original_folder = output_original_folder
        self.output_mask_folder = output_mask_folder
        self.valid_extensions = ('.png', '.jpg', '.jpeg')

    def copy_image(self, src, dst):
        if os.path.exists(dst):
            base, ext = os.path.splitext(os.path.basename(src))
            counter = 1
            while os.path.exists(dst):
                new_filename = f"{base}_{counter}{ext}"
                dst = os.path.join(os.path.dirname(dst), new_filename)
                counter += 1
        shutil.copy2(src, dst)

    def _get_overlap_filenames(self):
        return {os.path.splitext(filename)[0].lower() for filename in os.listdir(self.overlap_images_folder)
                if filename.lower().endswith(self.valid_extensions)}

    def _filter_images_to_copy(self, images):
        overlap_filenames = self._get_overlap_filenames()
        return [img for img in images if os.path.splitext(os.path.basename(img))[0].lower() in overlap_filenames]

    def copy_images(self):
        ensure_directory(self.output_original_folder)
        ensure_directory(self.output_mask_folder)
        original_images = [os.path.join(self.original_folder, filename) for filename in os.listdir(self.original_folder)
                           if filename.endswith(self.valid_extensions)]
        mask_images = [os.path.join(self.mask_folder, filename) for filename in os.listdir(self.mask_folder)
                       if filename.endswith(self.valid_extensions)]
        original_images_to_copy = self._filter_images_to_copy(original_images)
        mask_images_to_copy = self._filter_images_to_copy(mask_images)
        with tqdm(total=len(original_images_to_copy), desc='Copying Original Images') as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.copy_image, img,
                                           os.path.join(self.output_original_folder, os.path.basename(img))): img
                           for img in original_images_to_copy}
                for future in futures:
                    future.result()
                    pbar.update(1)
        with tqdm(total=len(mask_images_to_copy), desc='Copying Mask Images') as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.copy_image, img,
                                           os.path.join(self.output_mask_folder, os.path.basename(img))): img
                           for img in mask_images_to_copy}
                for future in futures:
                    future.result()
                    pbar.update(1)

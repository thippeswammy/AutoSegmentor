import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import create_yolo_structure

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageAugmentations:
    """Class to apply various image augmentations."""

    @staticmethod
    def apply_gaussian_blur(image, size=random.choice([3, 5])):
        """Apply Gaussian blur using a kernel for three-channel images."""
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)  # Add batch dimension
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def apply_average_blur(image, size=random.choice([3, 5])):
        """Apply average blur for three-channel images."""
        kernel = torch.ones((3, 1, size, size), dtype=torch.float32).to(device) / (size * size)
        image = image.unsqueeze(0)  # Add batch dimension
        return torch.nn.functional.conv2d(image, kernel, padding=size // 2, groups=3).squeeze(0)

    @staticmethod
    def add_gaussian_noise(image, mean=0.5, sigma=0.01):
        """Add Gaussian noise to the image."""
        noise = torch.randn(image.size()).to(device) * sigma + mean
        return (image + noise).clamp(0, 1)

    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """Add salt and pepper noise to the image."""
        noisy_img = image.clone()
        num_salt = int(salt_prob * image.numel())
        num_pepper = int(pepper_prob * image.numel())

        salt_coords = [torch.randint(0, dim, (num_salt,)).to(device) for dim in image.shape]
        pepper_coords = [torch.randint(0, dim, (num_pepper,)).to(device) for dim in image.shape]

        noisy_img[salt_coords] = 1  # Salt
        noisy_img[pepper_coords] = 0  # Pepper
        return noisy_img


class YoloProcessor:
    """Class to handle YOLO data processing, including augmentation and label management."""

    def __init__(self, config):
        try:
            fullPath, dataset_folder = create_yolo_structure.create_yolo_folder_structure(
                folder_name=config['folder_name'],
                main_path=config['dataset_saving_working_dir'],
                num_classes=config['class_names']
            )
        except Exception as e:
            logging.error(f"Error creating YOLO folder structure: {e}")
            raise

        self.train_image_count = self.val_image_count = self.test_image_count = 0
        self.train_save_path = os.path.join(fullPath, 'train')
        self.val_save_path = os.path.join(fullPath, 'valid')
        self.test_save_path = os.path.join(fullPath, 'test')
        self.SOURCE_img_type_ext = config['SOURCE_img_type_ext']
        self.SOURCE_mask_type_ext = config['SOURCE_mask_type_ext']
        self.SOURCE_mask_folder_name = config['SOURCE_mask_folder_name']
        self.SOURCE_original_folder_name = config['SOURCE_original_folder_name']
        self.ToDataTypeFormate = config['ToDataTypeFormate']
        self.augmenter = ImageAugmentations()
        self.color_to_label = config['color_to_label']
        self.FromDataType = config['FromDataType']
        self.class_names = config['class_names']
        self.class_to_id = config['class_to_id']
        self.train_split = config['train_split']
        self.source_dir_original_img = config['dataset_path'] + "/" + self.SOURCE_original_folder_name
        self.source_dir_mask_img = config['dataset_path'] + "/" + self.SOURCE_mask_folder_name
        self.test_split = config['test_split']
        self.val_split = config['val_split']
        self.main_path = config['dataset_saving_working_dir']
        self.factTimes = config['augment_times']
        self.num_threads = config['num_threads']
        self.keepValDatasetOriginal = config['Keep_val_dataset_original']
        self.DESTINATION_img_type_ext = config['DESTINATION_img_type_ext']
        self.DESTINATION_label_type_ext = config['DESTINATION_label_type_ext']

        if not os.path.exists(self.source_dir_original_img) or not os.path.exists(self.source_dir_mask_img):
            logging.error(
                f"Source directories '{self.source_dir_original_img}' or '{self.source_dir_mask_img}' do not exist.")
            raise FileNotFoundError(
                f"Source directories '{self.source_dir_original_img}' or '{self.source_dir_mask_img}' not found.")

    def distribute_files_with_threads(self):
        """Distribute files into training, validation, and test sets using multithreading."""
        image_paths = self.collect_image_paths(self.source_dir_original_img)
        if not image_paths:
            logging.error("No image files were found in the source directory.")
            return

        total_files = len(image_paths * self.factTimes)
        # if total_files / 10 > 10000:
        #     self.val_split = 10000 / total_files
        #     self.test_split = 1000 / total_files
        self.test_image_count = int(total_files * self.test_split)
        self.val_image_count = int(total_files * self.val_split)
        self.train_image_count = total_files - self.test_image_count - self.val_image_count

        len_ind = os.path.basename(image_paths[0]).index('.')
        file_infos = [(os.path.basename(file_path)[:len_ind], file_path) for file_path in image_paths]
        random.shuffle(file_infos)
        with tqdm(total=total_files, desc="Processing Images") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self.process_single_file, file_info, self.factTimes) for file_info in
                           file_infos]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Exception during processing: {e}")
                    pbar.update(self.factTimes)
        # for file_info in file_infos:
        #     self.process_single_file(file_info, self.factTimes)

    @staticmethod
    def collect_image_paths(directory):
        """Collect all image file paths from the given directory."""
        image_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, filename))
        logging.info(f"Found {len(image_paths)} images in the directory: {directory}.")
        return image_paths

    def process_single_file(self, file_info, Times):
        file_basename, file_path = file_info
        try:
            self.process_and_save(file_basename, file_path, Times)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {file_basename}: {e}")

    def get_label_path(self, image_source_path):
        """Construct the path for the label file based on the image path."""
        label_path = (image_source_path.replace(self.SOURCE_original_folder_name, self.SOURCE_mask_folder_name)
                      ).replace(".png", self.SOURCE_mask_type_ext).replace('.jpg', self.SOURCE_mask_type_ext).replace(
            '.jpeg', self.SOURCE_mask_type_ext)
        return label_path

    def process_and_save(self, file_name, image_source_path, Times):
        """Process and save images and their corresponding labels."""
        if not os.path.exists(image_source_path):
            logging.warning(f"Image file not found: {image_source_path}")
            return

        label_source_path = self.get_label_path(image_source_path)
        if os.path.exists(label_source_path):
            yolo_polygons_points_txt = self.process_mask_to_yolo_txt(label_source_path, self.class_to_id)
            for i in range(1, Times + 1):
                save_img_path, save_label_path = self.get_destination_paths(file_name, i)
                if save_img_path and save_label_path:
                    augmented_img = self.apply_augmentations(image_source_path, i)
                    if augmented_img is not None:
                        try:
                            if isinstance(augmented_img, torch.Tensor):
                                image_np = np.array(augmented_img.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
                            else:
                                image_np = np.array(augmented_img)  # PIL Image to numpy array
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                            self.save_yolo_format(save_label_path, yolo_polygons_points_txt)
                            cv2.imwrite(save_img_path, image_np)
                        except Exception as e:
                            logging.error(f"Error saving image {save_img_path}: {e}")
        else:
            logging.warning(f"Label file not found for image: {image_source_path}")

    def get_destination_paths(self, file_name, num):
        """Get the destination paths for saving images and labels."""
        save_img_path, save_label_path = '', ''
        choices = []
        if self.train_image_count > 0:
            choices.append(0)
        elif self.val_image_count > 0:
            choices.append(1)
        elif self.test_image_count > 0:
            choices.append(2)
        if self.keepValDatasetOriginal and num == 1:
            choice = 1
        else:
            choice = random.choice(choices)

        if choice == 0:
            save_img_path = os.path.join(self.train_save_path, 'images',
                                         f'{file_name}_{num}{self.DESTINATION_img_type_ext}')
            save_label_path = os.path.join(self.train_save_path, 'labels',
                                           f'{file_name}_{num}{self.DESTINATION_label_type_ext}')
            self.train_image_count -= 1
        elif choice == 1:
            save_img_path = os.path.join(self.val_save_path, 'images',
                                         f'{file_name}_{num}{self.DESTINATION_img_type_ext}')
            save_label_path = os.path.join(self.val_save_path, 'labels',
                                           f'{file_name}_{num}{self.DESTINATION_label_type_ext}')
            self.val_image_count -= 1
        elif choice == 2:
            save_img_path = os.path.join(self.test_save_path, 'images',
                                         f'{file_name}_{num}{self.DESTINATION_img_type_ext}')
            save_label_path = os.path.join(self.test_save_path, 'labels',
                                           f'{file_name}_{num}{self.DESTINATION_label_type_ext}')
            self.test_image_count -= 1
        return save_img_path, save_label_path

    def apply_augmentations(self, source_img_path, num):
        """Apply augmentations using PyTorch and return augmented image."""

        try:
            img = Image.open(source_img_path).convert("RGB")
            if self.keepValDatasetOriginal and num == 1:
                return img
            if num == 1:
                bright = [0.6, 0.9]
                contrast = [0.6, 0.8]
            elif num == 2:
                bright = [0.65, 1.1]
                contrast = [0.9, 1.1]
            elif num == 3:
                bright = [0.5, 0.9]
                contrast = [0.9, 1.1]
            elif num == 4:
                bright = [0.8, 1.1]
                contrast = [0.7, 0.8]
            elif num == 5:
                bright = [0.7, 1.1]
                contrast = [0.8, 1.1]
            else:
                bright = [0.99, 1.11]
                contrast = [0.99, 1.11]

            augmentations = T.Compose([
                T.ColorJitter(brightness=(bright[0], bright[1]), contrast=(contrast[0], contrast[1])),
                T.ToTensor(),
            ])
            img_tensor = augmentations(img).to(device)
            if num % 6 == 0:
                img_tensor = self.augmenter.apply_gaussian_blur(img_tensor)
            elif num % 6 == 1:
                img_tensor = self.augmenter.apply_average_blur(img_tensor)
            elif num % 6 == 2:
                img_tensor = self.augmenter.add_gaussian_noise(img_tensor, random.uniform(0, 0.5),
                                                               random.uniform(0.005, 0.04))
            elif num % 6 == 3:
                img_tensor = self.augmenter.add_salt_pepper_noise(img_tensor, random.uniform(0.005, 0.04),
                                                                  random.uniform(0.001, 0.05))
            else:
                return img
            return img_tensor
        except Exception as e:
            logging.error(f"Error applying augmentations on {source_img_path}: {e}")
            return None

    def process_mask_to_yolo_txt(self, mask_file_path, class_map):
        """Convert the mask file to YOLO format."""
        mask_image = cv2.imread(mask_file_path)
        image_height, image_width = mask_image.shape[:2]
        polygons = self.get_polygons(mask_image)
        yolo_polygons_txt = self.convert_polygons_to_yolo(image_width, image_height, polygons)

        return yolo_polygons_txt

    def get_polygons(self, mask_image):
        """
        Extract polygons for each label in the mask image.
        """
        polygons = []
        for color, label in self.color_to_label.items():
            mask = np.all(mask_image == color, axis=-1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 0:
                    polygons.append((label, contour.reshape(-1, 2)))
        return polygons

    @staticmethod
    def convert_polygons_to_yolo(img_width, img_height, polygons):
        """
        Convert polygon coordinates to YOLO format.
        """
        yolo_polygons = []
        for label, polygon in polygons:
            normalized_polygon = [(x / img_width, y / img_height) for (x, y) in polygon]
            yolo_polygons.append((label, normalized_polygon))
        return yolo_polygons

    @staticmethod
    def save_yolo_format(save_label_path, yolo_polygons):
        """Save the YOLO formatted text to the specified path."""
        try:
            with open(save_label_path, 'a') as f:
                for label, polygon in yolo_polygons:
                    polygon_str = ' '.join(f"{x} {y}" for x, y in polygon)
                    f.write(f"{label} {polygon_str}\n")
        except Exception as e:
            logging.error(f"Error saving YOLO label file {save_label_path}: {e}")


if __name__ == '__main__':
    CONFIG = {
        "dataset_path": r"F:\RunningProjects\SAM2\segment-anything-3\working_dir",
        "SOURCE_mask_folder_name": "render",
        "SOURCE_original_folder_name": "images",
        "SOURCE_mask_type_ext": '.png',
        "SOURCE_img_type_ext": '.jpeg',
        "augment_times": 10,  # Number of augmentations per image
        "test_split": 0.0,  # Percentage of data for testing
        "val_split": 0.1,  # Percentage of data for validation
        "train_split": 0.9,  # Percentage of data for training
        "Keep_val_dataset_original": True,  # for keeping the original dataset has original
        "num_threads": os.cpu_count() - 2,  # Number of threads for parallel processing
        "class_to_id": {
            'road': 0,
        },
        "color_to_label": {
            (255, 255, 255): 0,
        },
        "dataset_saving_working_dir": r'F:\RunningProjects\SAM2\DatasetManager',
        "folder_name": 'road',
        "class_names": ['road'],
        "DESTINATION_img_type_ext": '.jpg',
        "DESTINATION_label_type_ext": '.txt',
        "FromDataType": '',
        "ToDataTypeFormate": '',
    }

    try:
        processor = YoloProcessor(config=CONFIG)
        processor.distribute_files_with_threads()
    except Exception as e:
        logging.error(f"Critical error: {e}")

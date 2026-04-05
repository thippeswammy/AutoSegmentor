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
        self.annotation_manager = config.get('annotation_manager') # Pass manager directly
        self.export_types = config.get('export_types', ['mask']) # ['box', 'mask', 'pose']
        
        # Paths for specialized labels
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
        self.source_dir_original_img = os.path.join(config['dataset_path'], self.SOURCE_original_folder_name)
        self.source_dir_mask_img = os.path.join(config['dataset_path'], self.SOURCE_mask_folder_name)
        self.test_split = config['test_split']
        self.val_split = config['val_split']
        self.main_path = config['dataset_saving_working_dir']
        self.factTimes = config.get('augment_times', 1)
        self.num_threads = config.get('num_threads', 4)
        self.keepValDatasetOriginal = config.get('Keep_val_dataset_original', True)
        self.enabled_augmentations = config.get('enabled_augmentations', ['color', 'gauss_blur', 'avg_blur', 'gauss_noise', 'sp_noise'])
        self.DESTINATION_img_type_ext = config.get('DESTINATION_img_type_ext', '.jpg')
        self.DESTINATION_label_type_ext = config.get('DESTINATION_label_type_ext', '.txt')

        if not os.path.exists(self.source_dir_original_img):
             logging.error(f"Source directory '{self.source_dir_original_img}' does not exist.")
             raise FileNotFoundError(f"Source directory '{self.source_dir_original_img}' not found.")
             
        # Cache image dimensions from the first available image
        sample_img_paths = self.collect_image_paths(self.source_dir_original_img)
        if sample_img_paths:
            sample_img = cv2.imread(sample_img_paths[0])
            if sample_img is not None:
                self.img_h, self.img_w = sample_img.shape[:2]
            else:
                self.img_h, self.img_w = 720, 1280 # Fallback
        else:
            self.img_h, self.img_w = 720, 1280 # Fallback

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
        
        # 1. Process Mask -> Segmentation / Box
        yolo_segmentation = None
        yolo_box = None
        if os.path.exists(label_source_path):
            yolo_segmentation = self.process_mask_to_yolo_txt(label_source_path, self.class_to_id)
            # Box can be derived from segment or mask processing
            yolo_box = self.process_mask_to_yolo_box_txt(label_source_path, self.class_to_id)
        elif 'mask' in self.export_types or 'box' in self.export_types:
            logging.warning(f"Mask label file not found for image: {image_source_path}")

        # 2. Process Pose -> Keypoints
        yolo_pose = None
        if 'pose' in self.export_types and self.annotation_manager:
            try:
                frame_idx = int(os.path.splitext(file_name)[0].split('_')[-1])
            except ValueError:
                frame_idx = 0 # Fallback
            yolo_pose = self.process_pose_to_yolo_txt(frame_idx)
            if not yolo_pose:
                logging.warning(f"Pose keypoints not found for image: {image_source_path}")

        # 3. Save to split
        for i in range(1, Times + 1):
            dst = self.get_destination_paths(file_name, i)
            if dst:
                augmented_img = self.apply_augmentations(image_source_path, i)
                if augmented_img is not None:
                    try:
                        if isinstance(augmented_img, torch.Tensor):
                            image_np = np.array(augmented_img.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
                        else:
                            image_np = np.array(augmented_img)
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        cv2.imwrite(dst['image'], image_np)
                        
                        if 'mask' in self.export_types and yolo_segmentation:
                            self.save_yolo_format(dst['label_mask'], yolo_segmentation)
                        
                        if 'box' in self.export_types and yolo_box:
                            self.save_yolo_format(dst['label_box'], yolo_box)
                            
                        if 'pose' in self.export_types and yolo_pose:
                            self.save_yolo_format(dst['label_pose'], yolo_pose)
                            
                    except Exception as e:
                        logging.error(f"Error saving {dst['image']}: {e}")

    def get_destination_paths(self, file_name, num):
        """Get the destination paths for saving images and multi-type labels."""
        choices = []
        if self.train_image_count > 0: choices.append(0)
        if self.val_image_count > 0: choices.append(1)
        if self.test_image_count > 0: choices.append(2)
        
        # If no choices left, default to train (0)
        choice = random.choice(choices) if choices else 0

        # Note: If keepValDatasetOriginal is True AND choice == 1 (val), 
        # the augmentation function will return the original image instead of blurring it.
        # But we DO NOT force choice=1 purely because num==1, as that breaks the split ratio.

        base_path = self.train_save_path if choice == 0 else (self.val_save_path if choice == 1 else self.test_save_path)
        
        if choice == 0: self.train_image_count -= 1
        elif choice == 1: self.val_image_count -= 1
        else: self.test_image_count -= 1
        
        return {
            'image': os.path.join(base_path, 'images', f'{file_name}_{num}{self.DESTINATION_img_type_ext}'),
            'label_box': os.path.join(base_path, 'labels_box', f'{file_name}_{num}{self.DESTINATION_label_type_ext}'),
            'label_mask': os.path.join(base_path, 'labels_mask', f'{file_name}_{num}{self.DESTINATION_label_type_ext}'),
            'label_pose': os.path.join(base_path, 'labels_pose', f'{file_name}_{num}{self.DESTINATION_label_type_ext}')
        }

    def apply_augmentations(self, source_img_path, num):
        """Apply augmentations using PyTorch and return augmented image."""

        try:
            img = Image.open(source_img_path).convert("RGB")
            if self.keepValDatasetOriginal and num == 1:
                return img
                
            # If no augmentations enabled by user, just return original
            if not self.enabled_augmentations:
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

            if 'color' in self.enabled_augmentations:
                augmentations = T.Compose([
                    T.ColorJitter(brightness=(bright[0], bright[1]), contrast=(contrast[0], contrast[1])),
                    T.ToTensor(),
                ])
            else:
                augmentations = T.Compose([T.ToTensor()])
                
            img_tensor = augmentations(img).to(device)

            # Determine filter/noise based on num mod using only selected extra augmentations
            extra_augs = [aug for aug in self.enabled_augmentations if aug != 'color']
            
            if extra_augs:
                chosen_aug = extra_augs[num % len(extra_augs)]
                if chosen_aug == 'gauss_blur':
                    img_tensor = self.augmenter.apply_gaussian_blur(img_tensor)
                elif chosen_aug == 'avg_blur':
                    img_tensor = self.augmenter.apply_average_blur(img_tensor)
                elif chosen_aug == 'gauss_noise':
                    img_tensor = self.augmenter.add_gaussian_noise(img_tensor, random.uniform(0, 0.5), random.uniform(0.005, 0.04))
                elif chosen_aug == 'sp_noise':
                    img_tensor = self.augmenter.add_salt_pepper_noise(img_tensor, random.uniform(0.005, 0.04), random.uniform(0.001, 0.05))

            return img_tensor
        except Exception as e:
            logging.error(f"Error applying augmentations on {source_img_path}: {e}")
            return None

    def process_mask_to_yolo_txt(self, mask_file_path, class_map):
        """Convert the mask file to YOLO format."""
        mask_image = cv2.imread(mask_file_path)
        if mask_image is None: return None
        image_height, image_width = mask_image.shape[:2]
        polygons = self.get_polygons(mask_image)
        return self.convert_polygons_to_yolo(image_width, image_height, polygons)

    def process_mask_to_yolo_box_txt(self, mask_file_path, class_map):
        """Convert the mask file to YOLO BBox format."""
        mask_image = cv2.imread(mask_file_path)
        if mask_image is None: return None
        image_height, image_width = mask_image.shape[:2]
        
        boxes = []
        for color, label in self.color_to_label.items():
            mask = np.all(mask_image == color, axis=-1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0 and h > 0:
                    cx = (x + w/2) / image_width
                    cy = (y + h/2) / image_height
                    nw = w / image_width
                    nh = h / image_height
                    boxes.append((label, [cx, cy, nw, nh]))
        return boxes

    def process_pose_to_yolo_txt(self, frame_idx):
        """Convert keypoints for a frame to YOLO Pose format."""
        prompt = self.annotation_manager.get_prompt_for_frame(frame_idx)
        if not prompt: return None
        
        w, h = self.img_w, self.img_h
        pose_lines = []
        kps = prompt.get("pose_keypoints", [])
        if kps:
            # YOLO Pose format: class_id cx cy w h k1_x k1_y v1 ...
            # Calculate a box around the keypoints
            xs = [k["x"] for k in kps if k["x"] >= 0]
            ys = [k["y"] for k in kps if k["y"] >= 0]
            if not xs: return None
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bw, bh = max_x - min_x, max_y - min_y
            cx, cy = (min_x + max_x) / 2 / w, (min_y + max_y) / 2 / h
            nw, nh = (bw + 10) / w, (bh + 10) / h # 10px padding
            
            line = [cx, cy, nw, nh]
            for k in kps:
                vis = 2 if k.get("visible", True) and k["x"] >= 0 else 0
                nx = k["x"] / w if k["x"] >= 0 else 0
                ny = k["y"] / h if k["y"] >= 0 else 0
                line.extend([nx, ny, vis])
            
            class_id = 0
            if kps[0].get("label"):
                 class_id = (kps[0]["label"] // 1000) - 1
            
            pose_lines.append((class_id, line))
        return pose_lines

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
    def save_yolo_format(save_label_path, yolo_data):
        """Save the YOLO formatted text to the specified path."""
        try:
            with open(save_label_path, 'w') as f: # Use 'w' instead of 'a' since we save per type
                for label, coords in yolo_data:
                    coords_str = ' '.join(f"{c}" for c in coords)
                    f.write(f"{label} {coords_str}\n")
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

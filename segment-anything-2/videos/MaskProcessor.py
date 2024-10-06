import cv2
import numpy as np


def convert_to_color(mask):
    """Convert the binary mask to a color image."""
    # Create a 3-channel image with the same height and width as the mask
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return color_mask


class MaskProcessor:
    def __init__(self, mask_path):
        self.mask_path = mask_path
        self.mask = self.load_mask()

    def load_mask(self):
        """Load the mask image in grayscale."""
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file {self.mask_path} not found.")
        return mask

    def process_mask(self):
        """Process the mask by filling holes and ensuring binary output."""
        # Convert to binary mask
        _, binary_mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological closing to fill holes
        kernel = np.ones((9, 9), np.uint8)  # Adjust kernel size as needed
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Ensure the mask is strictly binary (0s and 255s)
        _, binary_output = cv2.threshold(closed_mask, 200, 255, cv2.THRESH_BINARY)

        return binary_output

    def save_mask(self, output_path, mask):
        """Save the processed mask to a file."""
        cv2.imwrite(output_path, mask)

    def display_masks(self, original_mask, smoothed_mask):
        """Display the original and smoothed masks in full-screen."""
        # Create a named window and set it to full screen
        cv2.namedWindow('Original Mask', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Original Mask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Original Mask', original_mask)

        # Create another named window for the smoothed mask
        cv2.namedWindow('Smoothed Mask', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Smoothed Mask', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Smoothed Mask', smoothed_mask)

        # Wait for a key press to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the input and output paths
    input_mask_path = '../rendered_frames/road44_00002.png'  # Update with your input mask file name
    output_mask_path = 'smoothed_road_mask.png'  # Output file name
    output_color_mask_path = 'color_road_mask.png'  # Color output file name

    # Create a MaskProcessor instance
    processor = MaskProcessor(input_mask_path)

    # Process the mask
    smoothed_mask = processor.process_mask()

    # Convert the smoothed mask to a color image
    color_mask = convert_to_color(smoothed_mask)

    # Save the processed masks
    processor.save_mask(output_mask_path, smoothed_mask)
    processor.save_mask(output_color_mask_path, color_mask)

    # Optional: Display the original and processed masks
    processor.display_masks(processor.mask, smoothed_mask)

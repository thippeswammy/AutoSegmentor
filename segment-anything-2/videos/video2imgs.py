import os

import cv2
from tqdm import tqdm

VIDEO_NUMBER = 31


def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def main(val=VIDEO_NUMBER):
    video_path = f'D:\downloadFiles\\front_3\\video{val}.mp4'
    output_dir = 'F:\RunningProjects\SAM2\segment-anything-2\\videos\\road_imgs'
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)  # Load the video
    cap = cv2.VideoCapture(video_path)  # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize the progress bar
    with tqdm(total=total_frames, desc='Extracting Frames', unit='frame') as pbar:
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Save each frame as a .png file with a filename like road3_00000.png, etc.
            frame_filename = os.path.join(output_dir, f'road{VIDEO_NUMBER}_{frame_count:05d}.png')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1  # Update the progress bar
            pbar.update(1)

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_dir}'")


if __name__ == "__main__":
    main(31)

'''
    video18 = road1 ==> 40
    video19 = road2 ==> 40
    video2  = road3 ==> 40
    video23 = road4 ==> 40
    
    video31 = road31 ==> 80
    video32 = road32 ==> 80
    video33 = road33 ==> 80
    video34 = road34 ==> 80
    video35 = road35 ==> 80
    video36 = road36 ==> 80
    video37 = road37 ==> 80
    video38 = road38 ==> 80
    video39 = road39 ==> 80
    video40 = road40 ==> 80
    video41 = road41 ==> 80
    video42 = road42 ==> 80
    video43 = road43 ==> 80
    video44 = road44 ==> 80
    video45 = road45 ==> 80
    video46 = road46 ==> 80
    video47 = road47 ==> 80
    video48 = road48 ==> 80
    video49 = road49 ==> 80
    video50 = road50 ==> 80
    video51 = road51 ==> 80
    video52 = road52 ==> 80
    
'''

import csv
import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from pathlib import Path

from detectors import YOLOv8_World, LightGlue
from detectors import SAM2

def create_directory(directory_name, verbose=False):
    """
        Check if a folder already exists, if not create a folder in the provided path.
    """
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
        if verbose:
            print(f'created directory {directory_name}')
    return

def extract_frames(base_path, csv_path, video_path, extracted_frames_path, stop_frame: int = None, verbose=False):
    """Extract frames from video with optimized sequential reading"""
    
    df = pd.read_csv(os.path.join(base_path, csv_path))
    world_indices = sorted(df['world_index'].values)  # Sort indices for sequential reading
    
    # Pre-check existing frames
    existing_frames = {int(f.split('_')[1].split('.')[0]) 
                      for f in os.listdir(extracted_frames_path) 
                      if f.startswith('frame_')}
    needed_indices = [idx for idx in world_indices if idx not in existing_frames]
    
    if not needed_indices:
        return
        
    cap = cv2.VideoCapture(video_path)
    current_frame = 0

    if verbose:
        print(f'Extracting frames from {video_path} to {extracted_frames_path}')
        print(f'Total frames to extract: {len(needed_indices)}')
    
    for idx in tqdm(needed_indices):
        if stop_frame is not None and current_frame >= stop_frame:
            if verbose:
                print(f'Stop frame reached at {stop_frame}')
            break
            
        # Skip frames if needed
        while current_frame < idx:
            cap.grab()  # Fast frame skip without decoding
            current_frame += 1
            
        ret, frame = cap.read()
        current_frame += 1
        
        if ret:
            cv2.imwrite(os.path.join(extracted_frames_path, f'frame_{idx}.jpg'), frame)
    
    cap.release()

def create_video(frame_path, output_path, frame_size=None, frame_rate=10):
    """
    Recombine frames into a video with optimized I/O and processing.
    """
    # Use Path object for better path handling
    frame_path = Path(frame_path)
    
    # Get and sort files in one pass using list comprehension
    image_files = sorted(
        [f for f in frame_path.glob('*.png')],
        key=lambda x: int(x.stem.split('_')[1])
    )
    
    if not image_files:
        raise ValueError("No PNG files found in directory")
        
    # Read first frame and get size
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read first frame: {image_files[0]}")
        
    frame_size = frame_size or first_frame.shape[:2][::-1]
    
    # Use more efficient codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)
    
    try:
        # Process frames in batches for better I/O
        batch_size = 100
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            frames = [cv2.imread(str(f)) for f in batch]
            for frame in frames:
                if frame is not None:
                    out.write(frame)
    finally:
        out.release()

def mask_coverage_in_bbox(mask, bbox):
    """Calculate the portion of the bounding box that's covered by the mask."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    roi = mask[y1:y2, x1:x2]
    roi = ToTensor()(roi)
    return torch.sum(roi).item() / (roi.shape[0] * roi.shape[1])


def predict_annotation(base_path, fixation_path, frame_path, th_coverage=0.4, savefig_path=None, create_sequence_images=False, verbose=False):
    """
    Process video frames, perform segmentation and object detection, and save the results.

    Args:
        base_path (str): Base directory path.
        fixation_path (str): Path to the fixation CSV file.
        frame_path (str): Directory containing video frames.
        th_coverage (float): Threshold for mask coverage to consider a valid detection.
        savefig_path (str, optional): Directory to save result images.
        create_sequence_images (bool, optional): If True, saves visualized frames.
        verbose (bool, optional): Print intermediate processing info.

    Returns:
        pd.DataFrame: DataFrame with results for each frame.
    """
    # Store results for each frame
    data_list = []

    # Load fixation data
    df = pd.read_csv(os.path.join(base_path, fixation_path))
    
    segmenter = SAM2()
    detector = YOLOv8_World()
    #detector = LightGlue(reference_image_path="ref_church3.jpg", threshold=30)

    if verbose:
        print(f"Loaded fixation data from {fixation_path} with {len(df)} entries.")
        
    # Process each frame
    for idx in df['world_index'].unique():
        image_path = os.path.join(frame_path, f'frame_{idx}.jpg')

        if not os.path.exists(image_path):
            if verbose:
                print(f"Frame {idx} not found.")
            continue

        if verbose:
            print(f"Processing frame {idx}.")

        # Load the image
        image = Image.open(image_path)
        image_np = np.array(image)

        # Convert BGR to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Extract gaze point
        gaze_data = df[df['world_index'] == idx]
        x_mean = gaze_data['x_mean'].values[0]
        y_mean = gaze_data['y_mean'].values[0]

        x_coordinate = int(x_mean * image.width)
        y_coordinate = int(y_mean * image.height)

        prompt_point = np.array([[x_coordinate, y_coordinate]])
        prompt_label = np.array([1])

        best_mask = segmenter.predict(prompt_point, prompt_label, image)

        boxes, classes, class_names = detector.detect(image_np)

        # Select the best matching class based on overlap
        candidates = []
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)
            if x_min <= prompt_point[0, 0] <= x_max and y_min <= prompt_point[0, 1] <= y_max:
                coverage = mask_coverage_in_bbox(best_mask, box)
                if coverage > th_coverage: # Threshold for valid overlap
                    candidates.append((coverage, i))

        if not candidates:
            max_iou_box = None
            max_iou_label = None
            coverage = None
        else:
            _, max_iou_idx = max(candidates, key=lambda x: x[0])
            max_iou_box = boxes[max_iou_idx]
            max_iou_label = class_names[classes[max_iou_idx]]

        # Append data to the result list
        data_list.append({
            'frame_number': idx,
            'point_x': x_coordinate,
            'point_y': y_coordinate,
            'duration': gaze_data['time'].values[0],
            'yolo_label': max_iou_label,
            'bounding_box_max_iou': max_iou_box if max_iou_box is not None else None,
            'mask_coverage': coverage,
            'manual_label': None
        })

        if create_sequence_images and savefig_path:
            # Save visualization
            os.makedirs(savefig_path, exist_ok=True)
            overlay = image_np.copy()

            # Create green mask overlay
            green_mask = np.zeros_like(image_np, dtype=np.uint8)
            green_mask[best_mask] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 1, green_mask, 0.5, 0)

            # Draw bounding boxes and labels
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box)
                color = (0, 255, 255) if max_iou_box is not None and np.array_equal(box, max_iou_box) else (255, 0, 0)
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)

                if max_iou_box is not None and np.array_equal(box, max_iou_box):
                    label_text = f"{max_iou_label}" if max_iou_label else "Unknown"
                    cv2.putText(
                        overlay, label_text, (x_min, y_min - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1
                    )

            # Draw fixation point
            cv2.circle(overlay, (x_coordinate, y_coordinate), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot

            # Save the visualization image
            output_path = os.path.join(savefig_path, f'frame_{idx}.png')
            cv2.imwrite(output_path, overlay)

    # Save results to a CSV file
    df_export = pd.DataFrame(data_list)
    df_export.to_csv('detection.csv')

    return df_export

def get_dispersion(points):
    """
    Calculate the dispersion of a given list of points.
    """
    dispersion = 0
    if len(points) == 0:
        return dispersion
    xx = []
    yy = []

    # access x_mean and y_mean
    for i in points:
        xx.append(i[3])
        yy.append(i[4])
    argxmin = np.min(xx)
    argxmax = np.max(xx)

    argymin = np.min(yy)
    argymax = np.max(yy)

    dispersion = ((argxmax - argxmin) + (argymax - argymin)) / 2

    return dispersion

def check_range(value, min_max=(0, 1)):
    """
    Check if a given value is outside a specific range. Return the range value instead.
    """
    if value < min_max[0]:
        return None
    elif value > min_max[1]:
        return None
    else:
        return value

def get_sum_x_y(points):
    """
    Calculate the mean x and y coordinate of all provided gaze points.
    """
    x = 0
    y = 0
    if len(points) == 0:
        return x, y
    for i in points:
        x += i['norm_pos_x']
        y += i['norm_pos_y']

    # accomodate a small bug, for mean fixations out of the range of 0 and 1
    xx, yy = check_range(x / len(points)), check_range(y / len(points))
    return xx, yy

def get_distance(points, x, y):
    """
    Calculate the Euclidean distance between a list of given points and a point.
    """
    if len(points) == 0:
        return 0
    res = 0
    for i in points:
        res += math.sqrt(((i['norm_pos_x'] - x) ** 2) +
                         ((i['norm_pos_y'] - y) ** 2))
    return res / len(points)

def get_time(points):
    """
    Calculate the time-duration from a list of points.
    """
    time = 0
    if len(points) == 0:
        return time
    first = points[0]['gaze_timestamp']
    last = points[len(points) - 1]['gaze_timestamp']

    time = (last - first) / 1000000
    return time

def write_export_file(filename, data, csv_field, verbose=False):
    """
    Write a csv file, containing the provided fields and data.
    """
    if verbose:
        print(f'writing file to {filename}')
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_field)
        for i in data:
            if len(i) > 0:
                csvwriter.writerow(i)
    return

def idt(data_path, dis_threshold, dur_threshold, file_basename, output_path, verbose=False):
    """
    Calculate the fixation and saccades of eye-tracking data by reading a csv file containing the gaze positions.
    """

    # Load the data
    data = np.genfromtxt('{}'.format(data_path), delimiter=',', skip_header=1,
                         names=['gaze_timestamp', 'world_index', 'confidence', 'norm_pos_x', 'norm_pos_y'])

    for i in data:
        i['gaze_timestamp'] = round((i['gaze_timestamp'] * 1000000), 2)

    # define the fields for the exported dataframe containing all necessary columns
    fixation_fields = ['id', 'time', 'world_index', 'x_mean', 'y_mean', 'start_frame', 'end_frame', 'dispersion']
    saccades_fields = ['id', 'first_gaze_timestamp', 'first_world_index', 'first_confidence', 'first_norm_pos_x',
                       'first_norm_pos_y', 'last_gaze_timestamp', 'last_world_index', 'last_confidence',
                       'last_norm_pos_x', 'last_norm_pos_y']

    fixation_export, saccades_export, x = list(), list(), list()

    nano_dur_threshold = dur_threshold * 1000
    first = data[0]['gaze_timestamp']

    # calculate the fixation and saccades for each data entry
    for index, val in enumerate(data):
        second_time = val['gaze_timestamp']
        vl = second_time - first
        if nano_dur_threshold > vl or get_dispersion(x) < dis_threshold:
            x.append(val)
        else:
            new_first = x[0]
            new_last = x[len(x) - 1]
            xx, yy = get_sum_x_y(x)
            # check if the calculation is outside [0, 1]
            if xx is None or yy is None:
                if verbose:
                    print(f'An error in the fixation calculation is suspected. The fixation point at {int(val["world_index"])} has been skipped.')
                x = []
                continue
            xx_d = get_distance(x, xx, yy)
            fixation_export.append(
                [index, get_time(x), int(new_first['world_index']), xx, yy, int(new_first['world_index']),
                 int(new_last['world_index']), xx_d])
            saccades_export.append([index, new_first['gaze_timestamp'] / 1000000, int(new_first['world_index']),
                                    new_first['confidence'], new_first['norm_pos_x'], new_first['norm_pos_y'],
                                    new_last['gaze_timestamp'] / 1000000, int(new_last['world_index']),
                                    new_last['confidence'], new_last['norm_pos_x'], new_last['norm_pos_y']])
            x = []

    # write the created data onto files for later use
    write_export_file(filename=f'{output_path}/{file_basename}_fixation.csv', data=fixation_export,
                      csv_field=fixation_fields, verbose=verbose)
    write_export_file(filename=f'{output_path}/{file_basename}_saccades.csv', data=saccades_export,
                      csv_field=saccades_fields, verbose=verbose)

    return


import os
import cv2
import pickle
import numpy as np
import pandas as pd

initial_latitude = 21.1938
initial_longitude = 81.3509
cell_size_mts = 30

with open('./grid_size.pkl', 'rb') as f:
    grid_size = pickle.load(f)
    
with open('./total_population.pkl', 'rb') as f:
    total_population = pickle.load(f)

print("total_population : ", total_population)
 
def non_max_suppression(contours, overlap_thresh=0.3):
    if len(contours) == 0:
        return []
    selected_contours = []
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    areas = [w * h for x, y, w, h in bounding_boxes]
    sorted_indices = np.argsort(areas)[::-1]
    while len(sorted_indices) > 0:
        largest_index = sorted_indices[0]
        selected_contours.append(contours[largest_index])
        x1, y1, w1, h1 = bounding_boxes[largest_index]
        rect1 = (x1, y1, x1 + w1, y1 + h1)
        iou = [calculate_iou(rect1, bounding_boxes[i]) for i in sorted_indices[1:]]
        filtered_indices = np.where(np.array(iou) <= overlap_thresh)[0]
        sorted_indices = sorted_indices[filtered_indices + 1]
    return selected_contours

def calculate_iou(box1, box2):
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def detect_color_regions(image_path, color_ranges):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_counts = {color: 0 for color in color_ranges.keys()}
    area_sums = {color: 0 for color in color_ranges.keys()}
    
    # Get image dimensions
    height, width, _ = image.shape
    total_area = width * height
    
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = non_max_suppression(contours)
        
        # Ensure that the sum of areas does not exceed the total area of the image
        total_area_color = sum(cv2.contourArea(contour) for contour in contours)
        if total_area_color > total_area:
            contours = adjust_areas(contours, total_area, total_area_color)
        
        color_counts[color] = len(contours)
        for contour in contours:
            area_sums[color] += cv2.contourArea(contour)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    # Optionally, display the image with contours
    # cv2.imshow("Color Regions", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return color_counts, area_sums, total_area

def adjust_areas(contours, total_area, total_area_color):
    adjustment_factor = total_area / total_area_color
    adjusted_contours = []
    for contour in contours:
        new_contour = contour * adjustment_factor
        adjusted_contours.append(new_contour.astype(int))
    return adjusted_contours



# color_ranges = {
#     'red': ([0, 100, 100], [10, 255, 255]),  
#     'blue': ([100, 50, 50], [140, 255, 255]),
#     'yellow': ([28, 150, 150], [35, 255, 255]),
#     'green': ([30, 70, 150], [60, 255, 255]) 
# }

color_ranges = {
    'red': ([0, 100, 100], [10, 255, 255]),  
    'blue': ([100, 50, 50], [140, 255, 255]),
    'yellow': ([18, 150, 150], [45, 255, 255]),
    'green': ([45, 70, 150], [70, 230, 255]) 
}

output_tiles_folder = './output_tiles'
data = []

for filename in sorted(os.listdir(output_tiles_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(output_tiles_folder, filename)
        color_counts, area_sums, image_pixels = detect_color_regions(image_path, color_ranges)
        location = tuple(map(int, filename.split('.')[0].split('_')[1:]))

        
        data.append({
            'location': location,
            'red_count': color_counts['red'],
            'blue_count': color_counts['blue'],
            'yellow_count': color_counts['yellow'],
            'green_count': color_counts['green'],
            'red_pixels': area_sums['red'],
            'blue_pixels': area_sums['blue'],
            'yellow_pixels': area_sums['yellow'],
            'green_pixels': area_sums['green'],
            'image_pixels': image_pixels
        })
df = pd.DataFrame(data)
print(df)
sorted_df = df.sort_values(by='green_pixels')
print(sorted_df)

with open('./Testing/cc_dataframe.pkl', 'wb') as f:
    pickle.dump(df, f)

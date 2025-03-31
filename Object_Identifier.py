import cv2 as cv
import numpy as np

image = cv.imread('coke_can.jpg')
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 100, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(image, contours, -1, (0, 255, 0), 3)

green_mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 255) & (image[:, :, 2] == 0)

green_mask[:3, :] = False  
green_mask[-3:, :] = False  
green_mask[:, -3:] = False  

green_mask = green_mask.astype(np.uint8) * 255
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(green_mask, connectivity=4)

new_labels = np.zeros_like(labels)

new_label = 1 
max_area = 0
for label in range(1, num_labels):
    area = stats[label, cv.CC_STAT_AREA]
    max_area = max(area, max_area)

insignificant_objects = []
label_index = 0
#Discard anything less than 10%
for label in range(1, num_labels):
    area = stats[label, cv.CC_STAT_AREA]
    
    if area <= min(300, max_area*0.1) and (stats[label][0]>2 and stats[label][1]>2):
        insignificant_objects.append((label, label_index))
    label_index+=1


for label, _ in insignificant_objects:
    labels[labels == label] = 0

contours_filtered = []

for label in range(1, num_labels):
    if np.any(labels == label):
        mask = np.uint8(labels == label) * 255
        cnt, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_filtered.extend(cnt)

image_filtered = cv.imread('coke_can.jpg')
cv.drawContours(image_filtered, contours_filtered, -1, (0, 255, 0), 3)

def is_encapsulated(box1, box2):
    min_x1, max_x1, min_y1, max_y1 = box1
    min_x2, max_x2, min_y2, max_y2 = box2
    return (min_x1 >= min_x2 and max_x1 <= max_x2 and min_y1 >= min_y2 and max_y1 <= max_y2)

def is_too_small(box):
    min_x, max_x, min_y, max_y = box
    return (max_x - min_x < 5) or (max_y - min_y < 5)

bounding_boxes = []
for i in range(len(contours_filtered)):
    contour = contours_filtered[i]
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)

    x_values = contour[:, 0] 
    y_values = contour[:, 1]  

    min_x = np.min(x_values)  
    max_x = np.max(x_values) 
    min_y = np.min(y_values)  
    max_y = np.max(y_values)
    bounding_boxes.append((i,(min_x, max_x, min_y, max_y)))
# Assumes there is no overlapping objects
filtered_boxes = []
print(bounding_boxes)
added_boxes = set()
for i, box in bounding_boxes:
    if is_too_small(box):
        continue
    is_encapsulated_box = False
    for j, other_box in bounding_boxes:
        if i != j and is_encapsulated(box, other_box):
            area_box = (box[1] - box[0]) * (box[3] - box[2])
            area_other_box = (other_box[1] - other_box[0]) * (other_box[3] - other_box[2])
            if area_box > area_other_box:
                if j not in added_boxes:
                    filtered_boxes.append((i, box))
                    added_boxes.add(i)
            else:
                if i not in added_boxes:
                    filtered_boxes.append((j, other_box))
                    added_boxes.add(j)
            is_encapsulated_box = True
            break
    if not is_encapsulated_box and i not in added_boxes:
        filtered_boxes.append((i, box))
        added_boxes.add(i)

contours_objects = []
for i in range(len(contours_filtered)):
    if i in added_boxes:
        contours_objects.append(contours_filtered[i])
image_huge = cv.imread('coke_can.jpg')
cv.drawContours(image_huge,contours_objects, -1, (0, 255, 0), 3)
added_boxes = list(added_boxes)

ordered_boxes = sorted(added_boxes, key=lambda x: bounding_boxes[x][1][0])
print(ordered_boxes)
coin_x_change = bounding_boxes[ordered_boxes[0]][1][1] - bounding_boxes[ordered_boxes[0]][1][0]
coin_y_change = bounding_boxes[ordered_boxes[0]][1][3] - bounding_boxes[ordered_boxes[0]][1][2]

#Change this based on coin/initial object of known length size
known_size = 2
mm_per_pixel = known_size / ((coin_x_change+coin_y_change)/2)
for i in (added_boxes):
    top_left = (bounding_boxes[i][1][0], bounding_boxes[i][1][2])
    bottom_right = (bounding_boxes[i][1][1], bounding_boxes[i][1][3])
    print(top_left, bottom_right)
    cv.rectangle(image_huge, top_left, bottom_right, (0, 255, 0), 3)
    cv.rectangle(image_huge, top_left, bottom_right, (0, 255, 0), 3)
    
    size = (bottom_right[0] - top_left[0]) * mm_per_pixel 
    
    # Add text above the top left of the rectangle
    text_position_top = (top_left[0], top_left[1] - 10)
    cv.putText(image_huge, f"{size:.2f} mm", text_position_top, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    size = (bottom_right[1] - top_left[1]) * mm_per_pixel 
    # Add text on the side of the rectangle
    text_position_side = (bottom_right[0] + 10, bottom_right[1])
    cv.putText(image_huge, f"{size:.2f} mm", text_position_side, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)


cv.imshow('Image with Contours', image_huge)
cv.waitKey(0)
cv.destroyAllWindows()
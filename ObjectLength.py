import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def drawCoutour (count, pixel_per_cm, image):
    for cnt in count:
        tl, tr, w, h = cv2.boundingRect(cnt)
        tlval = (tl, tr)
        trval = (tl+w, tr)
        brval = (tl+w, tr+h)
        blval = (tl, tr+h)
        surroundings = np.array([tlval, trval, brval, blval])
        cv2.drawContours(image, [surroundings], -1, (0, 0, 255), 3)
         
        vertical_mid = (tl+w//2, tr-10)
        horizontal_mid = (tl+w+10, tr+h//2)
        print(vertical_mid, horizontal_mid)
        
        width = euclidean(tlval, trval)/pixel_per_cm
        height = euclidean(trval, brval)/pixel_per_cm
            
        cv2.putText(image, "{:.1f}cms".format(width), vertical_mid,
            cv2.FONT_ITALIC, 0.5, (25, 23, 0), 2)
        cv2.putText(image, "{:.1f}cms".format(height), horizontal_mid,
            cv2.FONT_ITALIC, 0.5, (25, 23, 0), 2)
 
def show_images(images):
	for i, image in enumerate(images):
		cv2.imshow("image ", image)
	cv2.waitKey(0)

image_name = "foot.jpg"

# Read image and preprocessing, including edge detection
image = cv2.imread(image_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

edged = cv2.Canny(blur, 50, 100)
ret, thresh = cv2.threshold(blur, 100, 255, 0)

count, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count_sorted = sorted(count, key=lambda x: cv2.boundingRect(x)[0])
count = [count_sorted[i] for i in range(len(count_sorted)) if cv2.contourArea(count_sorted[i]) > 150]
# Get reference object dimensions
ref_object = count[0]
tl, tr, h, w = cv2.boundingRect(ref_object)
pixel_dist = euclidean((tl, tr+w), (tl, tr))
#Can adjust this number based on leftmost object size
dist_in_cm = 21.5
#Uses pixels for measurement
pixel_per_cm = pixel_dist/dist_in_cm

drawCoutour (count, pixel_per_cm, image)
show_images([image])

cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read()
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

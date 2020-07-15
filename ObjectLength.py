import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import utlis


#Insert the image path name
image_name = "foot.jpg"

# Read image and preprocessing
image = cv2.imread(image_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Find the # of contours
count = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = imutils.grab_contours(count)

# Sort contours from left to right as leftmost contour is reference object
(count, _) = contours.sort_contours(count)

# Remove small contours, removes noise
count = [x for x in count if cv2.contourArea(x) > 90]

# Get reference object dimensions
ref_object = count[0]
surroundings = cv2.minAreaRect(ref_object)
surroundings = cv2.boxPoints(surroundings )
surroundings = np.array(surroundings , dtype="int")
surroundings = perspective.order_points(surroundings )
(tl, tr, br, bl) = surroundings
pixel_dist = euclidean(tl, tr)
#Can adjust this number based on leftmost object size
dist_in_cm = 2.0
#Uses pixels for measurement
pixel_per_cm = pixel_dist/dist_in_cm


# Draw all contours that were counted
utlis.drawCoutour (count, pixel_per_cm, image)
#Check utlis for definition
utlis.show_images([image])

#Can extend for video capture for an extension
cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read() #returns ret and the frame
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

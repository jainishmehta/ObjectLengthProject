import cv2
#This will show the image
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils

def drawCoutour (count, pixel_per_cm, image):
    for cnt in count:
        surroundings = cv2.minAreaRect(cnt)
        surroundings = cv2.boxPoints(surroundings )
        surroundings = np.array(surroundings , dtype="int")
        surroundings = perspective.order_points(surroundings )
        (tl, tr, br, bl) = surroundings
        #Draw the boxes
        cv2.drawContours(image, [surroundings .astype("int")], -1, (0, 0, 255), 2)
         
        #Calculate the midpoints for the vertical and horizontal sections
        vertical_mid = (tr[0] + int(abs(br[0]-tr[0])/2), tr[1] + int(abs( br[1]-tr[1])/2))
        horizontal_mid = (tl[0] + int(abs( tl[0]-tr[0] )/2), tl[1] + int(abs( tl[1]-tr[1] )/2))
            
        #Calculate the pixel per meter value  
        width = euclidean(tl, tr)/pixel_per_cm
        height = euclidean(tr, br)/pixel_per_cm
            
         # Put the information as text on the image  
        cv2.putText(image, "{:.1f}cms".format(height), (int(vertical_mid[0] + 10), int(vertical_mid[1])),
            cv2.FONT_ITALIC, 0.5, (25, 23, 0), 2)
        cv2.putText(image, "{:.1f}cms".format(width), (int(horizontal_mid [0] - 15), int(horizontal_mid [1] - 10)),
            cv2.FONT_ITALIC, 0.5, (25, 23, 0), 2)
 
def show_images(images):
	for i, image in enumerate(images):
		cv2.imshow("image ", image)
		#waitKey(0) will display the window infinitely until any keypress
	cv2.waitKey(0)

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
drawCoutour (count, pixel_per_cm, image)
#Check utlis for definition
show_images([image])

#Can extend for video capture eventually
cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read() #returns ret and the frame
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

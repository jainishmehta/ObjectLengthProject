import cv2

#This will show the image
import numpy as np
from imutils import perspective
from scipy.spatial.distance import euclidean


def show_images(images):
	for i, image in enumerate(images):
		cv2.imshow("image ", image)
		#waitKey(0) will display the window infinitely until any keypress
	cv2.waitKey(0)

def drawCoutour (count, pixel_per_cm, image):
    for cnt in count:
        surroundings = cv2.minAreaRect(cnt)
        surroundings = cv2.boxPoints(surroundings )
        surroundings = np.array(surroundings , dtype="int")
        surroundings = perspective.order_points(surroundings )
        (tl, tr, br, bl) = surroundings
        #Draw the boxes
        cv2.drawContours(image, [surroundings .astype("int")], -1, (0, 0, 255), 2)
        horizontal_mid = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        vertical_mid = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        width = euclidean(tl, tr)/pixel_per_cm
        height = euclidean(tr, br)/pixel_per_cm
        cv2.putText(image, "{:.1f}cms".format(width), (int(horizontal_mid [0] - 15), int(horizontal_mid [1] - 10)),
            cv2.FONT_ITALIC, 0.5, (25, 23, 0), 2)
        cv2.putText(image, "{:.1f}cms".format(height), (int(vertical_mid[0] + 10), int(vertical_mid[1])),
            cv2.FONT_ITALIC, 0.5, (25, 23, 0), 2)

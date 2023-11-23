import cv2
import matplotlib.pyplot as plt
import numpy as np


# Watershed documentation

coins_overlap = cv2.imread(r"C:\aiprojects\computer-vision\coins_overlap.jpg")
assert coins_overlap is not None, "file could not be read, check with os.path.exists()"

gray = cv2.cvtColor(coins_overlap,cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (19, 19), 0)


#Threshold Processing 
ret, bin_img = cv2.threshold(blurred, 
                             0, 255,  
                             cv2.THRESH_OTSU) 

# Noise Removal

# Morphological Operations: On shapes and structures
# Erosion - shrinks the boundaries - for removing small isolated structures
# Dilation - expands the boundaries - for filling gaps, joining structures
# Opening = Erosion >> Dilation - First remove small structures and then seperate objects
# Closing = Dilation >> Erosion - Used to close the gaps between nearby objects

# Increasing iterations will increase the overall effect of OPENING

# kernel is the 3x3 matrix that gets convoluted on the image to get the desired effect

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
bin_img = cv2.morphologyEx(bin_img,  
                           cv2.MORPH_OPEN, 
                           kernel, 
                           iterations=3) 

cv2.imshow("Binary Image", np.hstack([gray, bin_img]))
cv2.waitKey(0)

# Enhancing the background by dilating the shape boundaries
sure_bg = cv2.dilate(bin_img,kernel,iterations=3)

# Finding the pixel values representing Euclidean distance of each foreground pixel ...
# ... from the nearest background pixel
# bin_img should be with black background and white foreground
# cv2.DIST_L2 = Euclidean distance metrix (sqrt((x2 - x1)^2 + (y2 - y1)^2))
# Last value (0) - kernel size - defines the neighbourhood within which the distanc is calculated.
# Reducing kernel size = can detect more detailed and smaller structures
# Increaseing kernel size = less sensitive to smaller structure detection.
# kernel size can be 0 / 3/ 5

dist_transform = cv2.distanceTransform(bin_img,cv2.DIST_L2,0)

# ret returns the threshold value
# pixel values below thresholding value (0.7 * max distance) is 0
# This will enhance the foreground objects and save in sure_fg
# 0.7 is empirical value and can change according to the application
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Converting into integer matrix
sure_fg = np.uint8(sure_fg)

# Unknown area represents the area where the algorithm is uncertain if it is ...
# ... a foreground or background
# Giving this area in the marker will tell the watershed algorithm to carefully decide in these regions.
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
# It assigns a label to each connected components
# ret >> number of connected components (In this case it should be 9 (9 coins))
# markers >> labeled image where each pixel is assigned a label that corresponds to connected component.
ret, markers = cv2.connectedComponents(bin_img)

# Add one to all labels so that sure background is not 0, but 1 because indexing starts from 0
markers = markers+1
# Now, mark the region of unknown with zero so unknown regions become the part of background.
markers[unknown==255] = 0

# Apply the watershed algorithm to the original image with the markers
markers = cv2.watershed(coins_overlap,markers)

# Rewriting the coin_overlap by changing pixel values at boundaries with a color
# markers == -1 is for boundaries.
coins_overlap[markers == -1] = [255,0,0]

cv2.imshow("Object Segmentation", coins_overlap)
cv2.waitKey(0)


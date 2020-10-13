import cv2
import numpy as np
import numpy.ma as ma
import os
from os import path
import sys

arglen = len(sys.argv)
if(arglen < 2):
    print("Require filename argument.")
    exit()

file = sys.argv[1]

img = cv2.imread(file, cv2.IMREAD_COLOR)
w = img.shape[1]
h = img.shape[0]

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


threshold_value, threshold_image = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("out",threshold_image)
cv2.waitKey(0)

label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_image, 8, cv2.CV_32S)

max_stat = max(stats, key=lambda stat: stat[4])

#a better method maybe?
def search_index_max_size(stats, max_val):
    for i in range(0, stats.shape[0]):
        if stats[i][4] == max_val:
            return i

max_idx = search_index_max_size(stats, max_stat[4])
        
MIN_AREA = 2300 - 843
MAX_AREA = 7000

#print(max_idx)
largest_component_pixels_array = ((labels == max_idx).astype(np.uint8) * 255)

hole_filled = 255 - largest_component_pixels_array

hole_filled = cv2.morphologyEx(hole_filled, cv2.MORPH_OPEN, (3,3))

cv2.imshow("out",hole_filled)
cv2.waitKey(0)

label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(hole_filled, 8, cv2.CV_32S)

def filter_stats_for_erythrocytes(stats):
    eligible_indexes = []
    for i in range(0, stats.shape[0]):
        area = stats[i][4]
        #print(area)
        if MIN_AREA < area and area < MAX_AREA:
            eligible_indexes.append(i)
    return eligible_indexes

erythrocyte_labels = filter_stats_for_erythrocytes(stats)

erythrocyte_binary_image = np.zeros((h,w), dtype=np.uint8)

if not path.exists("./out"):
    os.mkdir("./out")

for idx in erythrocyte_labels:

    stat = stats[idx]
    x, y, w, h, a = stat

    #Select pixels that contains label idx
    selected_label = (labels == idx).astype(np.uint8)
    
    #cv2.imshow("out",erythrocyte_binary_image*255)
    
    #append selected_label to erythrocyte_binary_image
    erythrocyte_binary_image = erythrocyte_binary_image + selected_label

    #mask to show original imag e compared to the detected erythrocytes
    masked = cv2.bitwise_and(img ,img, mask=selected_label)

    roi = masked[y:y+h, x:x+w ]
    

 
    cv2.imwrite("./out/%s.jpg" % (idx), roi)

    #cv2.imshow("out", roi)
    #cv2.waitKey(0)

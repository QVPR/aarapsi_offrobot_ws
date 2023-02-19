import cv2
import numpy as np

def labelImage(img_in, textstring, org_in, colour):
# Write textstring at position org_in, with colour and black border on img_in

    # Black border:
    img_A = cv2.putText(img_in, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=(0,0,0), thickness=7, lineType=cv2.LINE_AA)
    # Colour inside:
    img_B = cv2.putText(img_A, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=colour, thickness=2, lineType=cv2.LINE_AA)
    return img_B

def makeImage(query_raw, match_path, icon_to_use, icon_size=100, icon_dist=0):
# Produce image to be published via ROS

    match_img = cv2.imread(match_path)
    query_img = cv2.resize(query_raw, (match_img.shape[1], match_img.shape[0]), interpolation = cv2.INTER_AREA) # resize to match_img dimensions
    
    match_img_lab = labelImage(match_img, "Reference", (20,40), (100,255,100))
    query_img_lab = labelImage(query_img, "Query", (20,40), (100,255,100))

    if icon_size > 0:
        # Add Icon:
        img_slice = match_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :]
        # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
        icon_mask_inv = cv2.inRange(icon_to_use, (50,50,50), (255,255,255)) # get border (white)
        icon_mask = 255 - icon_mask_inv # get shape
        icon_mask_stack_inv = cv2.merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255 # stack into rgb layers, binary image
        icon_mask_stack = cv2.merge([icon_mask, icon_mask, icon_mask]) / 255 # stack into rgb layers, binary image
        opacity_icon = 0.8 # 80%
        # create new slice with appropriate layering
        img_slice = (icon_mask_stack_inv * img_slice) + \
                    (icon_mask_stack * icon_to_use) * (opacity_icon) + \
                    (icon_mask_stack * img_slice) * (1-opacity_icon)
        match_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :] = img_slice

    return np.concatenate((match_img_lab, query_img_lab), axis=1)
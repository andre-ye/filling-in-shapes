import cv2
import numpy as np
import random
import os
from tqdm.notebook import tqdm


# returns square with random shift and side length
def RandomSquare():
    
    side_length = random.randint(20, 70)
    left_top_corner_loc = (random.randint(10, 118-side_length), random.randint(10,118-side_length))
    
    outline_image = cv2.rectangle(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                                  pt1=left_top_corner_loc, 
                                  pt2=(left_top_corner_loc[0]+side_length, left_top_corner_loc[1]+side_length), 
                                  color=1, thickness=2)
    filled_image = cv2.rectangle(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                                 pt1=left_top_corner_loc, 
                                 pt2=(left_top_corner_loc[0]+side_length, left_top_corner_loc[1]+side_length), 
                                 color=1, thickness=-1)
    
    return outline_image, filled_image


# returns rectangle with random shift and side lengths
def RandomRectangle():
    
    hor_side_len = random.randint(20,70)
    ver_side_len = random.randint(20,70)
    left_top_corner_loc = (random.randint(10,118-hor_side_len), random.randint(10,118-ver_side_len))

    outline_image = cv2.rectangle(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                                  pt1=left_top_corner_loc, 
                                  pt2=(left_top_corner_loc[0] + hor_side_len, left_top_corner_loc[1] + ver_side_len), 
                                  color=1, thickness=2)
    filled_image = cv2.rectangle(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                                 pt1=left_top_corner_loc, 
                                 pt2=(left_top_corner_loc[0] + hor_side_len, left_top_corner_loc[1] + ver_side_len), 
                                 color=1, thickness=-1)
    
    return outline_image, filled_image


# returns circle with random shift and radius
def RandomCircle():
    
    radius = random.randint(10,35)
    center_loc = (random.randint(10+radius, 118-radius), (random.randint(10+radius, 118-radius)))
    
    outline_image = cv2.circle(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                               center_loc, radius, color=1, thickness=4)
    filled_image = cv2.circle(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                               center_loc, radius, color=1, thickness=-1)
    
    return outline_image, filled_image


# returns ellipse with random shift and axis lengths
def RandomEllipse():
    
    major_axis_len = random.randint(10,35)
    minor_axis_len = random.randint(10,35)
    center_loc = (random.randint(10+major_axis_len, 118-major_axis_len), (random.randint(10+minor_axis_len, 118-minor_axis_len)))
    
    outline_image = cv2.ellipse(0 * np.ones(shape=[128, 128], dtype=np.uint8),
                               center_loc,
                               (major_axis_len, minor_axis_len),
                               0,0,360,1,4)
    filled_image = cv2.ellipse(0 * np.ones(shape=[128, 128], dtype=np.uint8),
                              center_loc,
                              (major_axis_len, minor_axis_len),
                              0,0,360,1,-1)
    
    return outline_image, filled_image


# returns star with random 
def RandomStar():
    
    # generating star shape w/ two circle method
    outer_circle_radius = random.randint(10, 35)
    inner_circle_radius = random.randint(3, outer_circle_radius - random.randint(5, outer_circle_radius-5))
    coordinates = []
    for k in range(1,6):
        coordinates.append(StarOuterPoint(k, outer_circle_radius))
        coordinates.append(StarInnerPoint(k, inner_circle_radius))
    coordinates = np.array(coordinates, np.int32)

    hshift, vshift = 64, 64  # centers the star in the middle

    # calculate changes to centering star in the middle w/ padding 10
    hshift += random.randint(-abs(64+min(coordinates[:,0])), 64-max(coordinates[:,0]))
    vshift += random.randint(-abs(64+min(coordinates[:,1])), 64-max(coordinates[:,1]))

    # apply shifts
    hor_coor = coordinates[:,0] + hshift
    ver_coor = coordinates[:,1] + vshift
    coordinates = np.array(list(zip(hor_coor, ver_coor)))
    
    outline_image = cv2.polylines(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                  [coordinates], 
                  True, 1, thickness=4)
    filled_image = cv2.fillPoly(0 * np.ones(shape=[128, 128], dtype=np.uint8), 
                  pts=[coordinates], color=1)
    
    return outline_image, filled_image


# helper for RandomStar(): returns the position for outer point on star
def StarOuterPoint(k, r):
    return [r*np.cos(((2*np.pi*k)/5) + (np.pi/2)), r*np.sin(((2*np.pi*k)/5) + (np.pi/2))]

# helper for RandomStar(): returns the position for inner point on star
def StarInnerPoint(k, r):
    return [r*np.cos(((2*np.pi*k)/5) + (7*np.pi/10)), r*np.sin(((2*np.pi*k)/5) + (7*np.pi/10))]


# returns 2 randomly drawn lines (input and output are the same b/c should have no effect)
def RandomLines():
    
    current_image = 0 * np.ones(shape=[128, 128], dtype=np.uint8)
    
    for i in range(2):
        start_coor = (random.randint(0, 128), random.randint(0, 128))
        end_coor = (random.randint(0, 128), random.randint(0,128))
        current_image = cv2.line(current_image, 
                                start_coor, end_coor, color=1, thickness=2)
    
    return current_image, current_image

# returns filled circles (already filled circles should have no change)
def RandomFilledCircle():
    
    radius = random.randint(10,35)
    center_loc = (random.randint(10+radius, 118-radius), (random.randint(10+radius, 118-radius)))
    filled_image = cv2.circle(0 * np.ones(shape=[128,128], dtype=np.uint8), 
                               center_loc, radius, color=1, thickness=-1)
    
    return filled_image, filled_image


# returns random image rotation and slight zoom out
def RandomRotation(img, degrees):
    M = cv2.getRotationMatrix2D((64,64), degrees, 0.8)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
  
  
def generate_dataset(num_data, path):
    dataset_path = os.path.join(os.getcwd(), path)
    train_path, test_path = os.path.join(dataset_path, "train"), os.path.join(dataset_path, "test")
    os.makedirs(train_path)
    os.makedirs(test_path)

    for i in tqdm(range(num_data)):
        
        # choose and instantiate a random shape function
        outline_image, filled_image = random.choice([RandomSquare, 
                                                     RandomRectangle, 
                                                     RandomCircle, 
                                                     RandomEllipse, 
                                                     RandomStar, 
                                                     RandomLines,
                                                     RandomFilledCircle
                                                    ])()
        
        # generate file path
        train_filename = os.path.join(train_path, "{}.png".format(i))
        test_filename = os.path.join(test_path, "{}.png".format(i))
        
        # before storing, rotate both by the same random degree
        degree_rotate = random.randint(0,360)
        cv2.imwrite(train_filename, RandomRotation(outline_image, degree_rotate)*255)
        cv2.imwrite(test_filename, RandomRotation(filled_image, degree_rotate)*255)
        
generate_dataset(25_000, "images") # generates 25k images and stores in folder "images"

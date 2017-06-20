import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from data import load_smallset
from color_space import convert_color
from image_display import side_by_side_plot


# Define a function to return HOG features and visualization
def channel_hog_features(img, orientions, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orientions, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orientions, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=feature_vec)
        return features

def get_hog_features(image, orientions=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(channel_hog_features(image[:,:,channel],
                                                     orientions, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        return np.ravel(hog_features)
    else:
        return channel_hog_features(image[:,:,hog_channel], orientions,
                                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
def demo():
    # Read in our vehicles and non-vehicles
    cars, notcars = load_smallset()

    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    image_YCrCb = convert_color(image, "YCrCb")
    channel1 = image_YCrCb[:, :, 0]
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = channel_hog_features(channel1, orient,
                                           pix_per_cell, cell_per_block,
                                           vis=True, feature_vec=False)

    side_by_side_plot(im1=image, im2=hog_image, im1_title="Example Car Image", im2_title="HOG Visualization", fontsize=16)

#demo()
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pickle

from hog_features import get_hog_features
from color_space import convert_color
from bin_spatial import bin_spatial
from color_histogram import color_hist
from data import load_dataset, read_image

class FeatureParameter():
    def __init__(self, color_space, orient, pix_per_cell, cell_per_block,
                 hog_channel, spatial_size, hist_bins, spatial_feat,
                 hist_feat, hog_feat):
        self.color_space = color_space # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient # HOG orientations
        self.pix_per_cell = pix_per_cell # HOG pixels per cell
        self.cell_per_block = cell_per_block # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_size # Spatial binning dimensions
        self.hist_bins = hist_bins # Number of histogram bins
        self.spatial_feat = spatial_feat # Spatial features on or off
        self.hist_feat = hist_feat # Histogram features on or off
        self.hog_feat = hog_feat # HOG features on or off

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_image_features(image, feature_parameter):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(image, feature_parameter.color_space)
    #3) Compute spatial features if flag is set
    if feature_parameter.spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=feature_parameter.spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if feature_parameter.hist_feat == True:
        hist_features, rhist, ghist, bhist, bin_centers = color_hist(feature_image, nbins=feature_parameter.hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if feature_parameter.hog_feat == True:
        hog_features = get_hog_features(feature_image, feature_parameter.orient, feature_parameter.pix_per_cell,
                                        feature_parameter.cell_per_block, feature_parameter.hog_channel)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, feature_parameter):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = read_image(file)
        image_features = single_image_features(image=image, feature_parameter=feature_parameter)
        features.append(image_features)
    # Return list of feature vectors
    return features

def train_classifier(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)


    print('Feature vector length:(training: ', len(X_train), ", validation: ", len(X_test), ")")
    # Use a linear SVC
    svc = LinearSVC(verbose=1)
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    #t=time.time()
    return (svc, X_scaler)

def save_classifier(clf, scaler, filename):
    output = open(filename, 'wb')
    pickle.dump({"classifier": clf, "scaler": scaler}, output)
    output.close()

def read_classifier(filename):
    p = pickle.load(open(filename, "rb"))
    clf = p["classifier"]
    scaler = p["scaler"]
    return (clf, scaler)

def run_train():
    cars, notcars = load_dataset()
    #cars, notcars = load_smallset()
    feature_parameter = selected_feature_parameter()

    car_features = extract_features(cars, feature_parameter=feature_parameter)
    notcar_features = extract_features(notcars, feature_parameter=feature_parameter)
    clf, X_scaler = train_classifier(car_features=car_features, notcar_features=notcar_features)
    save_classifier(clf=clf, scaler=X_scaler, filename="classifier.p")

def selected_feature_parameter():
    ### TODO: Tweak these parameters and see how the results change.
    return FeatureParameter(
        color_space = 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9,  # HOG orientations
        pix_per_cell = 8, # HOG pixels per cell
        cell_per_block = 2, # HOG cells per block
        hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16), # Spatial binning dimensions
        hist_bins = 16,    # Number of histogram bins
        spatial_feat = True, # Spatial features on or off
        hist_feat = True, # Histogram features on or off
        hog_feat = True # HOG features on or off
    )
#run_train()


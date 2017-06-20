import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from color_space import convert_color
from hog_features import channel_hog_features
from bin_spatial import bin_spatial
from color_histogram import color_hist
from sliding_window import draw_boxes
from search_classify import selected_feature_parameter, read_classifier
from data import read_image

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, y_start_stop, scale, clf, X_scaler, feature_parameter):
    hot_windows = subsample_search(img, y_start_stop, scale, clf, X_scaler, feature_parameter)

    draw_image = np.copy(img)
    draw_image = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    return draw_image

# Define a single function that can extract features using hog sub-sampling and make predictions
def subsample_search(img, y_start_stop, scale, clf, X_scaler, feature_parameter):
    draw_img = np.copy(img)
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    y_start, y_stop = y_start_stop
    pix_per_cell, cell_per_block, orient, spatial_size, hist_bins = (feature_parameter.pix_per_cell, feature_parameter.cell_per_block,
                                                                     feature_parameter.orient, feature_parameter.spatial_size, feature_parameter.hist_bins)

    img_to_search = img[y_start:y_stop,:,:]
    img_to_search = convert_color(img_to_search, space=feature_parameter.color_space)
    if scale != 1:
        imshape = img_to_search.shape
        img_to_search = cv2.resize(img_to_search, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = img_to_search[:,:,0]
    ch2 = img_to_search[:,:,1]
    ch3 = img_to_search[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = channel_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = channel_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = channel_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    on_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            #hog_features = hog_feat1

            x_left = xpos*pix_per_cell
            y_top = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_to_search[y_top:y_top+window, x_left:x_left+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features, rh, gh, bh, bincen = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            features = np.hstack((spatial_features, hist_features, hog_features))
            test_features = X_scaler.transform(features.reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                x_left_scaled = np.int(x_left*scale)
                y_top_scaled = np.int(y_top*scale)
                window_scaled = np.int(window*scale)
                on_windows.append(((x_left_scaled, y_top_scaled+y_start),(x_left_scaled+window_scaled,y_top_scaled+window_scaled+y_start)))
                #cv2.rectangle(draw_img,(x_left_scaled, y_top_scaled+y_start),(x_left_scaled+window_scaled,y_top_scaled+window_scaled+y_start),(0,0,255),6)

    return on_windows

def demo():
    feature_parameter = selected_feature_parameter()
    #scale = 1.5
    scale = 1
    clf, X_scaler = read_classifier("classifier.p")
    #img = read_image('test_images/bbox-example-image.jpg')
    img = read_image('test_images/test1.jpg')
    #y_start_stop = (400, 400+32)
    y_start_stop = (400, img.shape[0])

    draw_image = find_cars(img, y_start_stop, scale, clf, X_scaler, feature_parameter)

    plt.imshow(draw_image)
    plt.show()

#demo()
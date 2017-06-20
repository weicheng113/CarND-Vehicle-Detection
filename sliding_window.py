import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    x_start, x_stop = x_start_stop
    y_start, y_stop = y_start_stop
    # Compute the span of the region to be searched
    x_span = x_stop-x_start
    y_span = y_stop-y_start
    # Compute the number of pixels per step in x/y
    x_pixels_per_step, y_pixels_per_step = (np.array(xy_window) * xy_overlap).astype(int)
    # Compute the number of windows in x/y
    x_n_windows = int(x_span/x_pixels_per_step - 1)
    y_n_windows = int(y_span/y_pixels_per_step - 1)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    # Calculate each window position
    # Append window position to list
    for y_i in range(y_n_windows):
        for x_i in range(x_n_windows):
            x0 = x_start + x_i * x_pixels_per_step
            x1 = x0 + xy_window[0]
            y0 = y_start + y_i * y_pixels_per_step
            y1 = y0 + xy_window[1]
            window_list.append(((x0, y0), (x1, y1)))
    # Return the list of windows
    return window_list

def demo():
    image = mpimg.imread('test_images/bbox-example-image.jpg')
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()

#demo()
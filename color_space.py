import cv2
import numpy as np

def convert_color(image, space="RGB"):
    space_map = {'HSV': cv2.COLOR_RGB2HSV, 'HLS': cv2.COLOR_RGB2HLS,
                 'LUV': cv2.COLOR_RGB2LUV, 'YUV': cv2.COLOR_RGB2YUV,
                 'YCrCb': cv2.COLOR_RGB2YCrCb}
    if space == 'RGB':
        return np.copy(image)
    elif space in space_map:
        code = space_map.get(space)
        return cv2.cvtColor(image, code)
    else:
        raise Exception("Unsupported color space '", space, "'")
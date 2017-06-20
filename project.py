from moviepy.editor import VideoFileClip
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from classifier import selected_feature_parameter, read_classifier
from hog_subsample import subsample_search, draw_boxes
from heat_map import heatmap_filter, draw_labeled_bboxes
from data import read_image
from image_display import side_by_side_plot

class VehicleDetection():
    def __init__(self):
        self.feature_parameter = selected_feature_parameter()
        self.clf, self.X_scaler = read_classifier("classifier.p")
        self.recent_5_hot_windows = [None, None, None, None, None]
        #self.frame_number = 0
    def detect(self, image):
        #y_max = image.shape[0]

        hot_windows = []
        search_parameters = [#(8, (400, 400+32)),
                             #(16, (400, 400+64)),
                             (32, (400, 400+64)),
                             (64, (400,400+128)),
                             (96, (400,400+256)),
                             (128, (400,400+256))]
                             #(160, (400,400+256))]
        for search_parameter in search_parameters:
            search_window_size, y_start_stop = search_parameter
            scale = search_window_size / 64
            hot_windows.extend(subsample_search(image, y_start_stop, scale, self.clf, self.X_scaler, self.feature_parameter))

        self.recent_5_hot_windows = self.recent_5_hot_windows[1:]
        self.recent_5_hot_windows.append(hot_windows)
        combine_hot_windows, number_of_valid = self.combine_hot_windows(self.recent_5_hot_windows)
        heat_threshold = 5 + number_of_valid * 1
        heatmap = heatmap_filter(image, combine_hot_windows, heat_threshold)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        #heatmap0 = heatmap_filter(image, hot_windows, 0)
        #side_by_side_plot(im1=image, im2=heatmap0, im2_cmap='hot', im1_title="Frame {}".format(self.frame_number), im2_title="Heatmap {}".format(self.frame_number), fontsize=16)
        #self.frame_number += 1
        #if self.recent_5_hot_windows[0] is not None:
        #    side_by_side_plot(im1=heatmap, im2=draw_img, im1_cmap='hot', im1_title="Integrated Heatmap", im2_title="Integrated Bounding Boxes", fontsize=16)

        #draw_img = draw_boxes(image, hot_windows)
        return draw_img
    def combine_hot_windows(self, hot_windows_array):
        combined = []
        number_of_valid = 0
        for hot_wondows in hot_windows_array:
            if hot_wondows is not None:
                combined.extend(hot_wondows)
                number_of_valid += 1
        return (combined, number_of_valid)


def demo():
    vehicle_detection = VehicleDetection()

    image = read_image('test_images/test1.jpg')
    #image = read_image('test_images/screen2.png')
    draw_image = vehicle_detection.detect(image)

    side_by_side_plot(im1=image, im2=draw_image, im2_cmap='hot', im1_title="Example Image", im2_title="Bounding Boxes", fontsize=16)
    #plt.imshow(draw_image)
    #plt.show()

def process_video(file_path, out_dir):
    filename = file_path.replace('\\', '/').split("/")[-1]
    output_file_path = os.path.join(out_dir, filename.replace(".mp4",'_processed.mp4'))

    clip1 = VideoFileClip(file_path)
    vehicle_detection = VehicleDetection()
    video_clip = clip1.fl_image(vehicle_detection.detect)
    video_clip.write_videofile(output_file_path, audio=False)

#demo()
#process_video("project_video.mp4", ".")
process_video("test_video.mp4", ".")
from moviepy.editor import VideoFileClip
import os
import numpy as np
from keras import backend as K
from scipy.ndimage.measurements import label
from heat_map import heatmap_filter, draw_labeled_bboxes
from dl import search_cars, create_model

class VehicleDetection():
    def __init__(self):
        model = create_model((260, 1280, 3))
        model.load_weights("model_-11-0.01.h5")
        self.model = model
        self.recent_5_hot_windows = [None, None, None, None, None]
    def detect(self, image):
        hot_windows = search_cars(self.model, image)
        self.recent_5_hot_windows = self.recent_5_hot_windows[1:]
        self.recent_5_hot_windows.append(hot_windows)

        combine_hot_windows, number_of_valid = self.combine_hot_windows(self.recent_5_hot_windows)
        heat_threshold = 5 + number_of_valid * 3
        heatmap = heatmap_filter(image, combine_hot_windows, heat_threshold)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        return draw_img
    def combine_hot_windows(self, hot_windows_array):
        combined = []
        number_of_valid = 0
        for hot_wondows in hot_windows_array:
            if hot_wondows is not None:
                combined.extend(hot_wondows)
                number_of_valid += 1
        return (combined, number_of_valid)

def process_video(file_path, out_dir):
    filename = file_path.replace('\\', '/').split("/")[-1]
    output_file_path = os.path.join(out_dir, filename.replace(".mp4",'_processed_dl.mp4'))

    clip1 = VideoFileClip(file_path)
    vehicle_detection = VehicleDetection()
    video_clip = clip1.fl_image(vehicle_detection.detect)
    video_clip.write_videofile(output_file_path, audio=False)

    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()

#demo()
#process_video("project_video.mp4", ".")
#process_video("test_video.mp4", ".")
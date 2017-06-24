import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from image_display import side_by_side_plot
#from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

def load_smallset():
    car_files = glob.glob('vehicles_smallset/**/*.jpeg')
    notcar_files = glob.glob('non-vehicles_smallset/**/*.jpeg')
    return (car_files, notcar_files)

def load_dataset():
    car_files = glob.glob('vehicles/**/*.png')
    notcar_files = glob.glob('non-vehicles/**/*.png')
    return (car_files, notcar_files)

def read_image(filename):
    if filename.endswith("jpg") or filename.endswith("jpeg"):
        return mpimg.imread(filename)
    elif filename.endswith("png"):
        image = cv2.imread(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise Exception("unknown image type of file '", filename, "'")

def read_images(filenames):
    images = []
    for filename in filenames:
        image = read_image(filename)
        images.append(image)
    return images

# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict

def demo_smallset():
    cars, notcars = load_smallset()
    data_info = data_look(cars, notcars)

    print('Your function returned a count of',
          data_info["n_cars"], ' cars and',
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:',
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    side_by_side_plot(im1=car_image, im2=notcar_image, im1_title="Example Car Image", im2_title="Example Not-car Image", fontsize=16)

def demo_dataset():
    cars, notcars = load_dataset()

    print('Number of car samples: ', len(cars),
          "Number of non-car samples: ", len(notcars))
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])
    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.show()
#demo_smallset()
#demo_dataset()
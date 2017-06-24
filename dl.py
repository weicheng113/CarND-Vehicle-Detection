from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from data import load_smallset, load_dataset, read_images, read_image
from image_display import side_by_side_plot
from sliding_window import draw_boxes
from heat_map import heatmap_filter, draw_labeled_bboxes

def create_model(input_shape=(64,64,3)):
    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda image: image/255 - 0.5, input_shape=input_shape))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Dropout(0.5))
    # This acts like a 128 neuron dense layer
    model.add(Conv2D(128, (5, 5), activation='elu'))
    model.add(Dropout(0.5))
    # This is like a 1 neuron dense layer with tanh [-1, 1]
    model.add(Conv2D(1,(1,1), activation="sigmoid"))

    return model

def demo_model_summary():
    model = create_model()
    model.summary()

def train():
    model = create_model()
    model.summary()
    model.add(Flatten())

    cars, notcars = load_dataset()
    print("number of cars: ",len(cars), ", number of notcars: ", len(notcars))
    filenames = []
    filenames.extend(cars)
    filenames.extend(notcars)
    X = np.array(read_images(filenames))
    Y = np.concatenate([np.ones(len(cars)), np.zeros(len(notcars))])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=63)

    model_file="model_-{epoch:02d}-{val_loss:.2f}.h5"
    cb_checkpoint = ModelCheckpoint(filepath=model_file, verbose=1)
    cb_early_stopping = EarlyStopping(patience=2)
    optimizer = Adam()
    model.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=2, validation_data=(X_test, Y_test), callbacks=[cb_checkpoint, cb_early_stopping])
    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()

def demo_prediction():
    model = load_model("model_-14-0.04.h5")
    cars, notcars = load_smallset()
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = read_image(cars[car_ind])
    notcar_image = read_image(notcars[notcar_ind])

    car_prediction = model.predict(np.reshape(car_image, (1, 64,64,3)))
    notcar_prediction = model.predict(np.reshape(notcar_image, (1, 64,64,3)))
    side_by_side_plot(im1=car_image, im2=notcar_image, im1_title="prediction: {}".format(car_prediction),
                      im2_title="prediction: {}".format(notcar_prediction), fontsize=16)
    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()

def search_cars(model, img):
    # We crop the image to 440-660px in the vertical direction
    cropped = img[400:660, 0:1280]
    heat = model.predict(cropped.reshape(1,cropped.shape[0],cropped.shape[1],cropped.shape[2]))
    # This finds us rectangles that are interesting
    xx, yy = np.meshgrid(np.arange(heat.shape[2]),np.arange(heat.shape[1]))
    x = (xx[heat[0,:,:,0]>0.9])
    y = (yy[heat[0,:,:,0]>0.9])
    hot_windows = []
    # We save those rects in a list
    x_scale = cropped.shape[1]/heat.shape[2]
    y_scale = cropped.shape[0]/heat.shape[1]
    for i,j in zip(x,y):
        scaled_x = int(i*8)
        scaled_y = 400 + int(j*8)
        hot_windows.append(((scaled_x,scaled_y), (scaled_x+64,scaled_y+64)))
    return hot_windows

def demo_prediction2():
    image = read_image("test_images/test1.jpg")
    model = create_model((260, 1280, 3))
    model.load_weights("model_-11-0.01.h5")
    hot_windows = search_cars(model, image)
    heatmap = heatmap_filter(image, hot_windows, 3)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    side_by_side_plot(im1=image, im2=draw_img)
    # Temporary fix - AttributeError: 'NoneType' object has no attribute 'TF_NewStatus
    K.clear_session()
#demo_model_summary()
#train()
#demo_prediction()
#demo_prediction2()

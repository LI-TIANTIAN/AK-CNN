
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

sys.path.append('..')
matplotlib.use("Agg")

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=True,
        help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
        help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # activation function
        model.add(Dense(CLASS_NUM))
        model.add(Activation("softmax"))
        return model

# parameters
EPOCHS = 200
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 5
norm_size = 32


def eachFile(filepath):
   out = []
   for allDir in filepath:
      child = allDir
   return out

# def load_data(path):
#     print("[INFO] loading images...")
#     data = []
#     labels = []
#     # grab the image paths and randomly shuffle them
#     imagePaths = sorted(list(paths.list_images(path)))
#     random.seed(42)
#     random.shuffle(imagePaths)
#     # loop over the input images
#     for imagePath in imagePaths:
#         # load the image, pre-process it, and store it in the data list
#         image = cv2.imread(imagePath)
#         image = cv2.resize(image, (norm_size, norm_size))
#         image = img_to_array(image)
#         data.append(image)
#
#         # extract the class label from the image path and update the
#         # labels list
#         label = int(imagePath.split(os.path.sep)[-2])
#         labels.append(label)
#
#     # scale the raw pixel intensities to the range [0, 1]
#     data = np.array(data, dtype="float") / 255.0
#     labels = np.array(labels)
#
#     # convert the labels from integers to vectors
#     labels = to_categorical(labels, num_classes=CLASS_NUM)
#     return data, labels


# def train(aug, trainX, trainY, testX, testY, args):
#     # initialize the model
#     print("[INFO] compiling model...")
#     model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
#     opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#     model.compile(loss="categorical_crossentropy", optimizer=opt,
#                   metrics=["accuracy"])
#
#     # train the network
#     print("[INFO] training network...")
#     H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#                             validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#                             epochs=EPOCHS, verbose=1)
#
#     # save the model to disk
#     print("[INFO] serializing network...")
#     model.save(args["model"])
#
#     # plot the training loss and accuracy
#     plt.style.use("ggplot")
#     plt.figure()
#     N = EPOCHS
#     plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#     plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#     plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#     plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
#     plt.title("Training Loss and Accuracy on traffic-sign classifier")
#     plt.xlabel("Epoch #")
#     plt.ylabel("Loss/Accuracy")
#     plt.legend(loc="lower left")
#     plt.savefig(args["plot"])


if __name__ == '__main__':
        args = args_parse()
        train_file_path = './TRAIN/'
        test_file_path = './TEST/'
        train_datagen = ImageDataGenerator()
        train_generator = train_datagen.flow_from_directory(directory=train_file_path, target_size=(32, 32),
                                                            classes=eachFile(train_file_path))
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(directory=test_file_path, target_size=(32, 32),
                                                          classes=eachFile(test_file_path))
        model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
        opt = SGD(lr=0.0001, momentum=0.9)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                          metrics=["accuracy"])
        H = model.fit_generator(generator=train_generator, epochs=EPOCHS, validation_data=test_generator)
        print("[INFO] serializing network...")
        model.save('three_classes.h5')

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        #plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        #plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        #plt.title("Training Loss and Accuracy on traffic-sign classifier")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot1"])

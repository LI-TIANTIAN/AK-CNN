from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Concatenate
from keras import backend as K
import matplotlib
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from math import ceil
# sys.path.append('..')
# matplotlib.use("Agg")
from keras.models import Model
from keras.layers import Input, Dense, concatenate


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False,
        help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


class LeNet:
    @staticmethod
    def build(width, height, depth, classes = 5):
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

def generator_three_img(gen1):

    while True:
        X1i = gen1.next()
        X2i = X1i
        X3i = X1i
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]

def model1(width, height, depth):
    inputShape = (height, width, depth)
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)


    top_branch_input = Input(shape=inputShape, name='Top')
    top_branch_output = Conv2D(4, (7, 3), padding='same',strides=(2, 2),activation='relu')(top_branch_input)

    middle_branch_input = Input(shape=inputShape, name='Middle')
    middle_branch_output = Conv2D(8, (5, 5), padding='same', strides=(2, 2),activation='relu')(middle_branch_input)

    bottom_branch_input = Input(shape=inputShape, name='Bottom')
    bottom_branch_output = Conv2D(4, (3, 7), padding='same',  strides=(2, 2), activation='relu')(bottom_branch_input)

    concat = concatenate([top_branch_output, middle_branch_output,bottom_branch_output], name='Concatenate')

    conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(concat)
    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)
    flattern = Flatten()(conv2)
    dense1 = Dense(500, activation='relu')(flattern)
    dense2 = Dense(128, activation='relu')(dense1)
    final_model_output = Dense(CLASS_NUM, activation='softmax')(dense2)
    final_model = Model(inputs=[top_branch_input, middle_branch_input,bottom_branch_input], outputs=final_model_output,
                        name='Final_output')
    return final_model
    
    
def model2(width, height, depth):
    inputShape = (height, width, depth)
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)


    top_branch_input = Input(shape=inputShape, name='Top')
    top_branch_output = Conv2D(4, (7, 3), padding='same',strides=(2, 2))(top_branch_input)
    top_branch_output = LeakyReLU(0.2)(top_branch_output)

    middle_branch_input = Input(shape=inputShape, name='Middle')
    middle_branch_output = Conv2D(8, (5, 5), padding='same', strides=(2, 2))(middle_branch_input)
    middle_branch_output = LeakyReLU(0.2)(middle_branch_output)

    bottom_branch_input = Input(shape=inputShape, name='Bottom')
    bottom_branch_output = Conv2D(4, (3, 7), padding='same',  strides=(2, 2))(bottom_branch_input)
    bottom_branch_output = LeakyReLU(0.2)(bottom_branch_output)

    concat = concatenate([top_branch_output, middle_branch_output,bottom_branch_output], name='Concatenate')

    conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(concat)
    conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)
    flattern = Flatten()(conv2)
    dense1 = Dense(500, activation='relu')(flattern)
    dense2 = Dense(128, activation='relu')(dense1)
    final_model_output = Dense(CLASS_NUM, activation='softmax')(dense2)
    final_model = Model(inputs=[top_branch_input, middle_branch_input,bottom_branch_input], outputs=final_model_output,
                        name='Final_output')
    return final_model



class AKNet:
    @staticmethod
    def build(width, height, depth, classes):
        kernel1 = Sequential()
        kernel2 = Sequential()
        kernel3 = Sequential()
        model = Sequential()


        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        kernel1.add(Conv2D(8, (5, 5), padding='same',strides=(2,2), input_shape=inputShape, activation='relu'))
        kernel2.add(Conv2D(4, (3, 7), padding='same',strides=(2,2),input_shape=inputShape, activation='relu'))
        kernel3.add(Conv2D(4, (7, 3), padding='same',strides=(2,2), input_shape=inputShape, activation='relu'))

        merged = Concatenate([kernel1, kernel2,kernel3])

        model.add(merged)
        model.add(Conv2D(32,(3, 3), padding="same",strides=(2,2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same",strides=(2,2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Flatten())
        model.add(Dense(96))
        model.add(Dense(16))


        model.add(Dense(CLASS_NUM))
        model.add(Activation("softmax"))
        return model


# parameters
EPOCHS = 2000
INIT_LR = 1e-3
BS = 128
CLASS_NUM = 5
norm_size = 32
steps_per_epoch = ceil(1126618 / BS)
validation_step= ceil(281651 / BS)
def eachFile(filepath):
   out = []
   for allDir in filepath:
      child = allDir
   return out





if __name__ == '__main__':
        args = args_parse()
        file_path = '/data2/Lucy_dataset_HEVC/'
        checkpoint_filepath = '/data2/minh_hevc/HEVC_dataset_test/TrainingData/checkpoint/5_classes_Kevin.h5'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_acc',
            mode='max',
            save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)
        # train_file_path = './TRAIN/'
        # test_file_path = './TEST/'
        train_datagen = ImageDataGenerator(validation_split=0.25)
        train_generator = train_datagen.flow_from_directory(directory=file_path, target_size=(32, 32),
                                                            classes=eachFile(file_path),subset='training')
        # test_datagen = ImageDataGenerator()
        test_generator = train_datagen.flow_from_directory(directory=file_path, target_size=(32, 32),
                                                          classes=eachFile(file_path),subset='validation')

        print(train_generator.class_indices)

        # model = AKNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
        model = LeNet.build(width=norm_size, height=norm_size, depth=3)
        # opt = SGD(lr=0.001, momentum=0.9)

        opt = Adam(lr=0.001)
        #opt = RMSprop(lr=0.001, rho=0.9, decay=0.0)
        #opt = Adagrad(lr=0.01, decay=0.0)
        # opt = Adadelta(lr=0.01, rho=0.95, decay=0.0)

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()
        # H = model.fit_generator(generator_three_img(train_generator,), epochs=EPOCHS, validation_data=generator_three_img(test_generator,),steps_per_epoch=steps_per_epoch,validation_steps=validation_step,callbacks=[model_checkpoint_callback,es])
        H = model.fit_generator(train_generator, epochs=EPOCHS,
                                validation_data=test_generator, steps_per_epoch=steps_per_epoch,
                                validation_steps=validation_step, callbacks=[model_checkpoint_callback, es])

        print("[INFO] serializing network...")
        model.save('5_classes_Kevin.h5')

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = es.stopped_epoch + 1
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot"])
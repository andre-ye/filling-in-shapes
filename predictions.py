import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import cv2
import numpy as np

# LOADING MODEL

# standard convolution cell w/ 2 convolutions and 1 batchnorm; also serves as bottleneck
def conv_cell(input_layer, filters):
    
    x = layers.Conv2D(filters, (3,3), activation="relu", padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3,3), activation="relu", padding="same")(x)
    return x

# standard down-cell (encoding step) w/ convolution and pooling
def down_cell(input_layer, filters):
    
    c = conv_cell(input_layer, filters)
    p = layers.MaxPool2D((2, 2))(c)
    return c, p

# standard up-cell (decoding step) w/ upsampling, concatenation, and convolution
def up_cell(input_layer, concat_layer, filters):
    
    x = layers.UpSampling2D((2, 2))(input_layer)
    x = layers.Concatenate()([x, concat_layer])
    x = conv_cell(x, filters)
    return x


def UNet(input_shape=(128,128,1)):
    
    # layer naming:
    # uc = up convolution
    # dc = down convolution
    # b1 = bottleneck
    # p = pooling

    inputs = layers.Input(input_shape)

    uc1, p1 = down_cell(inputs, 8)
    uc2, p2 = down_cell(p1, 16)
    uc3, p3 = down_cell(p2, 32)
    uc4, p4 = down_cell(p3, 64)
    
    b = conv_cell(p4, 128)
    
    dc1 = up_cell(b, uc4, 64)
    dc2 = up_cell(dc1, uc3, 32)
    dc3 = up_cell(dc2, uc2, 16)
    dc4 = up_cell(dc3, uc1, 8)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(dc4)

    model = keras.models.Model(inputs, outputs)
    
    return model

model = UNet()
model.load_weights(os.path.join(os.getcwd(), "model-weights.h5")) # set to wherever model weights are


# PREDICTION

directory = os.path.join(os.getcwd(), "test_images\\train") # set to path for prediction folder
for (root,dirs,files) in os.walk(directory):
    for file in files[:10]:
        filename = os.path.join(directory, file)
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128,128))*255
        prediction = model.predict(np.array(image).reshape((1,128,128,1))).reshape(128,128,1)
        cv2.imshow("prediction", prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# DATA

# training data
outline_filenames = tf.data.Dataset.list_files("../images/train/*.png", shuffle=False)
outline_data = outline_filenames.map(lambda x: tf.io.decode_png(tf.io.read_file(x))/255)

filled_filenames = tf.data.Dataset.list_files("../images/train/*.png", shuffle=False)
filled_data = filled_filenames.map(lambda x: tf.io.decode_png(tf.io.read_file(x))/255)

complete_data = tf.data.Dataset.zip((outline_data, filled_data)).batch(32)

# test data generated to another folder
test_outline_filenames = tf.data.Dataset.list_files("../test_images/train/*.png", shuffle=False)
test_outline_data = test_outline_filenames.map(lambda x: tf.io.decode_png(tf.io.read_file(x))/255)

test_filled_filenames = tf.data.Dataset.list_files("../test_images/train/*.png", shuffle=False)
test_filled_data = test_filled_filenames.map(lambda x: tf.io.decode_png(tf.io.read_file(x))/255)

test_complete_data = tf.data.Dataset.zip((test_outline_data, test_filled_data)).batch(32)



# MODELING

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
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4), 
              loss="binary_crossentropy",
              metrics=["acc", "mse", "mae"])
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("best-unet.h5",
                                                        monitor="val_loss",
                                                        save_best_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.001)
model.fit(complete_data, epochs=30, 
          validation_data=test_complete_data, 
          callbacks=[model_checkpoint, reduce_lr])

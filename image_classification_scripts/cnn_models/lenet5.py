
# Architecture Description - taken from https://medium.com/datadriveninvestor/five-powerful-cnn-architectures
# -b939c9ddd57b

# Imports

import tensorflow as tf

NAME = 'Lenet 5'


# Main function

def get_model(input_shape, number_of_classes, optimizer='sgd'):

    formatted_shape = input_shape[1:]  # reference the shape formatted to fit with this network

    # !!!!!!!! LAYERS !!!!!!!! #

    input_layer = tf.keras.layers.Input(formatted_shape)

    conv1 = tf.keras.layers.Conv2D(filters=20, kernel_size=5, padding='same', activation=tf.keras.activations.relu)(input_layer)  # Layer
    # 1: A convolutional layer with kernel size of 5×5, stride of 1×1 and 6 kernels in total. So the input image of size 32x32x1 gives an output of 28x28x6. Total params in layer = 5 * 5 * 6 + 6 (bias terms)

    pool1 = tf.keras.layers.MaxPool2D()(conv1)  # Layer 2: A pooling layer with 2×2 kernel size, stride of 2×2 and 6 kernels in total. This pooling layer acted a little differently than what we discussed in previous post. The input values in the receptive were summed up and then were multiplied to a trainable parameter (1 per filter), the result was finally added to a trainable bias (1 per filter). Finally, sigmoid activation was applied to the output. So, the input from previous layer of size 28x28x6 gets sub-sampled to 14x14x6. Total params in layer = [1 (trainable parameter) + 1 (trainable bias)] * 6 = 12

    conv2 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, padding='same', activation=tf.keras.activations.relu)(pool1)  # Layer 3:
    # Similar to Layer 1, this layer is a convolutional layer with same configuration except it has 16 filters instead of 6. So the input from previous layer of size 14x14x6 gives an output of 10x10x16. Total params in layer = 5 * 5 * 16 + 16 = 416.

    pool2 = tf.keras.layers.MaxPool2D()(conv2)  # Layer 4: Again, similar to Layer 2, this layer is a pooling layer
    # with 16 filters this time around. Remember, the outputs are passed through sigmoid activation function. The input of size 10x10x16 from previous layer gets sub-sampled to 5x5x16. Total params in layer = (1 + 1) * 16 = 32

    flatten = tf.keras.layers.Flatten()(pool2)  # Layer 5: This time around we have a convolutional layer with 5×5
    # kernel size and 120 filters. There is no need to even consider strides as the input size is 5x5x16 so we will get an output of 1x1x120. Total params in layer = 5 * 5 * 120 = 3000

    dense1 = tf.keras.layers.Dense(500, activation=tf.keras.activations.relu)(flatten)  # Layer 6: This is a dense layer with 84
    # parameters. So, the input of 120 units is converted to 84 units. Total params = 84 * 120 + 84 = 10164. The activation function used here was rather a unique one. I’ll say you can just try out any of your choice here as the task is pretty simple one by today’s standards.

    output_layer = tf.keras.layers.Dense(number_of_classes, activation=tf.keras.activations.softmax)(dense1)  # Output
    # Layer: Finally, a dense layer with 10 units is used. Total params = 84 * 10 + 10 = 924.

    # !!!!!!!! MODEL !!!!!!!! #

    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model


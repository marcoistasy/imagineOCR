# Imports

import tensorflow as tf

NAME = 'Five Layer CNN'


# Main function

def get_model(input_shape, number_of_classes, optimizer=tf.keras.optimizers.Adam()):

    formatted_shape = input_shape[1:]  # reference the shape formatted to fit with this network

    # !!!!!!!! LAYERS !!!!!!!! #

    model = tf.keras.models.Sequential([

        # First layer
        tf.keras.layers.Conv2D(256, (3, 3), input_shape=formatted_shape, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second layer
        tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third layer
        tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Fourth layer
        tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Fifth layer
        tf.keras.layers.Flatten(),  # this converts our 3D feature maps to 1D feature vectors
        tf.keras.layers.Dense(64),

        # Output layer
        tf.keras.layers.Dense(number_of_classes, activation=tf.nn.softmax)

    ])

    # !!!!!!!! MODEL !!!!!!!! #

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

# Imports

import tensorflow as tf


NAME = 'Three Layer CNN'


# Main function

def get_model(input_shape, number_of_classes, optimizer=tf.keras.optimizers.Adam()):

    formatted_shape = input_shape[1:]  # reference the shape formatted to fit with this network

    # !!!!!!!! LAYERS !!!!!!!! #

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.keras.activations.relu, input_shape=formatted_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(number_of_classes, activation=tf.keras.activations.softmax)

    ])

    # !!!!!!!! MODEL !!!!!!!! #

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

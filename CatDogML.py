
#This model will train images of cats and dogs for classification

if __name__ == '__main__':
    # Cats v Dogs with custom splits
    # https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_addons as tfa
    import numpy as np


    def augmentimages(image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 255)
        image = tf.image.resize(image, (300, 300))
        return image, label


    count_data = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
    train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
    validation_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)
    test_data = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)

    # count_length = [i for i,_ in enumerate(count_data)][-1] + 1
    # print(count_length)

    # train_length = [i for i,_ in enumerate(train_data)][-1] + 1
    # print(train_length)

    # validation_length = [i for i,_ in enumerate(validation_data)][-1] + 1
    # print(validation_length)

    # test_length = [i for i,_ in enumerate(test_data)][-1] + 1
    # print(test_length)

    # augmented_training_data=train_data.map(augmentimages)
    # augmented_validation_data=validation_data.map(augmentimages)
    # train_batches = augmented_training_data.shuffle(1024).batch(32)
    # validation_batches = augmented_validation_data.batch(10)

    augmented_training_data = train_data.map(augmentimages)
    train_batches = augmented_training_data.shuffle(1024).batch(32)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_batches, epochs=25)



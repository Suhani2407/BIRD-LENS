import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

# Define data directories and parameters
train_data_dir = r'.vscode/dataset3/train'
test_data_dir = r'.vscode/dataset3/test'
image_size = (224, 224)
batch_size = 32
epochs = 20
num_classes = 20  # Replace with the actual number of bird classes in your dataset

# Create data generators
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_data_generator = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))  # Use num_classes

SGD = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

# Compile the model
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")


import time

# Generate a timestamp to include in the model name
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Save the model with a timestamp in the file name
model.save(f'bird_classification_model_{timestamp}.h5')



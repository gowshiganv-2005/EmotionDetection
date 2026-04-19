
# Train Emotion Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 48
batch_size = 64

train_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(48,48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size
)

test_data = test_gen.flow_from_directory(
    "dataset/test",
    target_size=(48,48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(7,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=3, validation_data=test_data)
model.save("emotion_model.h5")

print("Training complete")

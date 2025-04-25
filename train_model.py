import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

with open("pec/meta/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
num_classes = len(class_names)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'food_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)
model.save('food_recognition_model.h5')

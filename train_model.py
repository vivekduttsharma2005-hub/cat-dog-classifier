# train_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# Data setup (replace with your dataset paths)
# ------------------------------
train_dir = "dataset/train"  # folder with subfolders 'cats' and 'dogs'
val_dir = "dataset/val"

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(128,128), batch_size=16, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(128,128), batch_size=16, class_mode='binary')

# ------------------------------
# Small MobileNetV2 model
# ------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ------------------------------
# Train model (few epochs)
# ------------------------------
model.fit(train_gen, validation_data=val_gen, epochs=3)

# ------------------------------
# Save lightweight model as SavedModel
# ------------------------------
model.save("cat_dog_model_small")  # folder for Streamlit
print("✅ Model saved as cat_dog_model_small")
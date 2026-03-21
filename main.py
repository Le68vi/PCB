from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# DATA AUGMENTATION PIPELINE
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=15,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.15,
shear_range=0.1,
horizontal_flip=True,   # Set False if PCB orientation matters
brightness_range=[0.7, 1.3],
fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 2. DATA GENERATOR FUNCTION(Used by resnet)
def get_data_generators(batch_size=32):
    train_generator = train_datagen.flow_from_directory(
    'Dataset/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    seed=42
)

    validation_generator = val_datagen.flow_from_directory(
    'Dataset/val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

    print("Class indices:", train_generator.class_indices)

    return train_generator, validation_generator

#3. VISUALIZATION FUNCTION
def visualize_augmentations():

    train_generator, _ = get_data_generators()

    x_batch, y_batch = next(train_generator)

    plt.figure(figsize=(10, 6))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_batch[i])

    # Adjust label based on printed class_indices
        label = "Defect" if y_batch[i] == 0 else "Pass"

        plt.title(label)
        plt.axis('off')

        plt.suptitle("Augmented PCB Samples")
    plt.show()

#MAIN EXECUTION
if __name__ == "__main__":
    visualize_augmentations()

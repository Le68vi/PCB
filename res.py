from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Import data pipeline
from main import get_data_generators

#  BUILD MODEL
def build_resnet50_model():

    base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
    )

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

# Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )
    return model

# TRAIN MODEL
def train_model(model, train_gen, val_gen):

    callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'P_res_50.h5',   # NEW MODEL NAME
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

    history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    callbacks=callbacks
)

    return history

#  PLOT LEARNING CURVES
def plot_history(history):

    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss Curve")

    # Accuracy graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy Curve")

plt.show()


#  MAIN EXECUTION
if __name__ == "__main__":

    train_gen, val_gen = get_data_generators()# Load data
    model = build_resnet50_model()# Build model
    history = train_model(model, train_gen, val_gen) # Train model and save
    plot_history(history)# Plot results


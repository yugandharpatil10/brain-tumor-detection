import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Data loading and preprocessing 
print("\n\nDataset Loading and Preprocessing:\n\n")
print("The script loads MRI images from datasets/no/ (no tumor) and datasets/yes/ (tumor) folders.")
print("It filters for .jpg files, reads them using OpenCV, converts them to RGB, and resizes them to 64x64 pixels.")
print("mages are stored in a NumPy array with shape (n_samples, 64, 64, 3), and labels (0 for no tumor, 1 for tumor) are stored as (n_samples,).")
print("The data is split into 80% training and 20% testing sets, then normalized along the height axis to stabilize training.")


image_directory = 'datasets/'
no_tumour_images = os.listdir(image_directory + 'no/')
yes_tumour_images = os.listdir(image_directory + 'yes/')

dataset = []
label = []
INPUT_SIZE = 64

for i, image_name in enumerate(no_tumour_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumour_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

print("\nData preprocessing complete!")

# Define custom loss functions with shape handling
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.expand_dims(y_true, axis=-1)  # Reshape from (batch,) to (batch, 1)
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = self.alpha * tf.pow(1 - p_t, self.gamma) * ce_loss
        return tf.reduce_mean(focal_loss)

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.expand_dims(y_true, axis=-1)  # Reshape from (batch,) to (batch, 1)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        return 1 - dice

# Training function with explicit model saving
def train_model(model, name, x_train, y_train, x_test, y_test, epochs=10):
    history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=epochs,
                        validation_data=(x_test, y_test), shuffle=False)
    save_path = f'models/{name}10Epochs.h5'
    model.save(save_path)
    print(f"Model saved as {save_path}")
    return history

# Baseline Model (Your original CNN)
def build_baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Model 1: CNN with BatchNorm + Focal Loss + AdamW
def build_model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss=FocalLoss(), optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), metrics=['accuracy'])
    return model

# Model 2: MobileNetV2 + Binary Cross-Entropy + SGD
def build_model2():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    base_model.trainable = False
    inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])
    return model

# Model 3: Deeper CNN + Dice Loss + RMSprop
def build_model3():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss=DiceLoss(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])
    return model

# Train all models with unique names
models = {
    'BaselineCNN': build_baseline_model(),
    'CNNBNFocal': build_model1(),
    'MobileNetV2': build_model2(),
    'DeepCNNDice': build_model3()
}

# View model summary before training
for name, model in models.items():
    print(f"\n Model Summary of {name}...")
    model.summary()
    print()

# Train all models
print("Training started! \n\n")
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    history = train_model(model, name, x_train, y_train, x_test, y_test)
    results[name] = history.history

# Plotting results

print("\n\n Plotting Results: \n\n")
# def plot_results(results):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     for name, history in results.items():
#         ax1.plot(history['loss'], label=f'{name} Train Loss')
#         ax1.plot(history['val_loss'], label=f'{name} Val Loss')
#         ax2.plot(history['accuracy'], label=f'{name} Train Acc')
#         ax2.plot(history['val_accuracy'], label=f'{name} Val Acc')
#     ax1.set_title('Loss Comparison')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
#     ax2.set_title('Accuracy Comparison')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy')
#     ax2.legend()
#     plt.tight_layout()
#     plt.show()

# Compare final metrics
# plot_results(results)
# for name, history in results.items():
#     print(f"{name}: Final Val Loss: {history['val_loss'][-1]:.4f}, Final Val Acc: {history['val_accuracy'][-1]:.4f}")

# Print results to terminal (for output.txt)
def print_results(results):
    print("\n=== Training Results ===")
    for name, history in results.items():
        print(f"\nModel: {name}")
        print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc")
        print("-" * 50)
        for epoch in range(len(history['loss'])):
            train_loss = history['loss'][epoch]
            val_loss = history['val_loss'][epoch]
            train_acc = history['accuracy'][epoch]
            val_acc = history['val_accuracy'][epoch]
            print(f"{epoch + 1:5d} | {train_loss:.4f}     | {val_loss:.4f}   | {train_acc:.4f}    | {val_acc:.4f}")
        print(f"Final Val Loss: {val_loss:.4f}, Final Val Acc: {val_acc:.4f}")
    print("\n=======================")

# Save GUI plots to files
def save_plot_results(results):
    # Create a 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot and save for each model
    for name, history in results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(history['loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title(f'{name} Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(history['accuracy'], label='Train Acc')
        ax2.plot(history['val_accuracy'], label='Val Acc')
        ax2.set_title(f'{name} Accuracy Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        # Save the plot as a PNG file
        save_path = f'plots/{name}_results.png'
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved plot: {save_path}")

# Replace the original plot_results call with these
print_results(results)
save_plot_results(results)
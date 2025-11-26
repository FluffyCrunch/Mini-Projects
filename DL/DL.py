import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
import glob
import tensorflow as tf

# Helper functions for loading and processing images
def load_images(image_paths, target_size=(256, 256)):
    images = []
    for path in image_paths:
        try:
            img = load_img(path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return np.array(images)

def preprocess_images(rgb_images):
    # Convert RGB to LAB
    lab_images = []
    for img in rgb_images:
        lab_img = rgb2lab(img)
        # Normalize L channel to [0,1] and ab channels to [-1,1]
        l = lab_img[:,:,0] / 100.0
        ab = lab_img[:,:,1:] / 128.0
        lab_images.append((l, ab))
    return lab_images

# Define the colorization model
def build_colorization_model(input_shape=(256, 256, 1)):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    
    # Decoder (with upsampling)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Output layer
    output_layer = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    
    # Define model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# Function to visualize the results
def visualize_colorization(model, grayscale_images, original_images=None, num_samples=3):
    plt.figure(figsize=(12, 4 * num_samples))
    
    for i in range(min(num_samples, len(grayscale_images))):
        # Original grayscale image
        gray_img = grayscale_images[i]
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.title("Grayscale")
        plt.imshow(gray_img.squeeze(), cmap='gray')
        plt.axis('off')
        
        # Model prediction (colorized)
        pred_ab = model.predict(np.expand_dims(gray_img, axis=0))[0]
        # Rescale ab channels
        pred_ab = pred_ab * 128.0
        
        # Combine L and predicted ab channels
        colorized = np.zeros((gray_img.shape[0], gray_img.shape[1], 3))
        colorized[:, :, 0] = gray_img.squeeze() * 100.0  # L channel
        colorized[:, :, 1:] = pred_ab  # ab channels
        
        # Convert LAB to RGB
        colorized_rgb = lab2rgb(colorized)
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.title("Colorized")
        plt.imshow(colorized_rgb)
        plt.axis('off')
        
        # Original color image (if available)
        if original_images is not None:
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.title("Original")
            plt.imshow(original_images[i])
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main training function
def train_colorization_model(data_folder, epochs=5, batch_size=16):
    # Get image paths
    image_paths = glob.glob(os.path.join(data_folder, "*.jpg")) + \
                 glob.glob(os.path.join(data_folder, "*.png"))
    
    print(f"Found {len(image_paths)} images.")
    
    if len(image_paths) == 0:
        print(f"No images found in {data_folder}. Please check the path.")
        return None
    
    # Load and process images
    print("Loading images...")
    rgb_images = load_images(image_paths)
    print(f"Loaded {len(rgb_images)} images successfully.")
    
    print("Processing images...")
    lab_images = preprocess_images(rgb_images)
    
    # Prepare training data
    X = np.array([img[0].reshape(256, 256, 1) for img in lab_images])
    y = np.array([img[1] for img in lab_images])
    
    # Split data
    X_train, X_test, y_train, y_test, rgb_train, rgb_test = train_test_split(
        X, y, rgb_images, test_size=0.1, random_state=42)
    
    # Build and train the model
    print("Building model...")
    model = build_colorization_model()
    model.summary()
    
    print("Training model...")
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test))
    
    # Visualize results
    visualize_colorization(model, X_test, rgb_test)
    
    # Save model
    model.save('colorization_model.h5')
    print("Model saved as 'colorization_model.h5'")
    
    return model, history

# Function to colorize a single image
def colorize_image(model, image_path, output_path=None):
    # Load image and convert to grayscale
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    
    # Convert to LAB and get L channel
    lab_img = rgb2lab(img_array)
    l_channel = lab_img[:,:,0] / 100.0
    
    # Predict ab channels
    l_input = l_channel.reshape(1, 256, 256, 1)
    pred_ab = model.predict(l_input)[0] * 128.0
    
    # Combine L and predicted ab channels
    colorized = np.zeros((256, 256, 3))
    colorized[:,:,0] = l_channel * 100.0
    colorized[:,:,1:] = pred_ab
    
    # Convert back to RGB
    colorized_rgb = lab2rgb(colorized)
    
    # Display results
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Grayscale")
    plt.imshow(l_channel, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Colorized")
    plt.imshow(colorized_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save result if output path is provided
    if output_path:
        plt.imsave(output_path, colorized_rgb)
        print(f"Colorized image saved to {output_path}")
    
    return colorized_rgb

# Example usage
if __name__ == "__main__":
    # Path to folder containing training images
    data_folder = "./training_images"
    
    # Train the model
    model, history = train_colorization_model(data_folder, epochs=10)
    
    # Colorize a test image
    test_image_path = "./test_images/test_5.png"
    colorize_image(model, test_image_path, output_path="./colorized_test.png")

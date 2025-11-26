# Colorizing Old Black & White Images using Deep Learning

A deep learning project that automatically colorizes black and white images using a Convolutional Neural Network (CNN) architecture. The model is trained to predict color information from grayscale images using the LAB color space.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements an automatic image colorization system using deep learning. The model takes grayscale images as input and predicts the color channels (a and b in LAB color space) to produce colorized versions of black and white photographs.

## ✨ Features

- **Automatic Colorization**: Converts grayscale images to colorized versions
- **CNN-based Architecture**: Uses an encoder-decoder CNN architecture
- **LAB Color Space**: Works in LAB color space for better color prediction
- **Training Pipeline**: Complete training pipeline with data preprocessing
- **Visualization**: Built-in visualization tools to compare results
- **Model Saving**: Saves trained models for future use

## 🏗️ Architecture

The model uses an encoder-decoder architecture:

- **Encoder**: 
  - 3 convolutional layers with stride 2 for downsampling
  - Filters: 64 → 128 → 256
  
- **Decoder**:
  - 3 convolutional layers with upsampling
  - Filters: 128 → 64 → 32 → 2 (output channels)
  - Uses UpSampling2D layers to restore image resolution

- **Output**: 2 channels (a and b channels in LAB color space)

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-image
- scikit-learn

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Mini-Projects.git
cd Mini-Projects/DL
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow numpy matplotlib scikit-image scikit-learn
```

## 🚀 Usage

### Training the Model

1. Prepare your training images:
   - Place color images in a `training_images` folder
   - Supported formats: `.jpg`, `.png`

2. Run the training script:
```python
python DL.py
```

Or use the functions directly:
```python
from DL import train_colorization_model

# Train the model
model, history = train_colorization_model(
    data_folder="./training_images",
    epochs=10,
    batch_size=16
)
```

### Colorizing Images

After training, you can colorize new images:

```python
from DL import colorize_image

# Load trained model
from tensorflow.keras.models import load_model
model = load_model('colorization_model.h5')

# Colorize an image
colorize_image(
    model, 
    image_path="./test_images/test_5.png",
    output_path="./colorized_output.png"
)
```

## 📁 Project Structure

```
DL/
├── DL.py                    # Main implementation file
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── training_images/        # Training dataset (not included)
├── test_images/            # Test images (not included)
└── colorization_model.h5   # Trained model (generated after training)
```

## 🔬 How It Works

1. **Image Preprocessing**:
   - Images are loaded and resized to 256x256
   - Converted from RGB to LAB color space
   - L channel (lightness) is used as input
   - ab channels (color) are used as target

2. **Model Training**:
   - The model learns to predict ab channels from L channel
   - Uses Mean Squared Error (MSE) loss
   - Adam optimizer with learning rate 0.001

3. **Colorization**:
   - Input grayscale image is converted to LAB
   - Model predicts ab channels
   - L and predicted ab channels are combined
   - Converted back to RGB for display

## 📊 Results

The model produces colorized versions of grayscale images. Training results include:
- Training and validation loss curves
- Visual comparisons of grayscale, colorized, and original images
- Saved model for future inference

## 🔮 Future Improvements

- [ ] Add U-Net architecture for better feature preservation
- [ ] Implement transfer learning with pre-trained models
- [ ] Add support for higher resolution images
- [ ] Implement batch colorization
- [ ] Add web interface for easy usage
- [ ] Support for video colorization
- [ ] Fine-tuning with domain-specific datasets

## 📝 Notes

- The model works best with natural images
- Training time depends on dataset size and hardware
- For better results, use a larger and diverse training dataset
- Model performance improves with more training epochs

## 👤 Author

FluffyCrunch

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This is a mini-project for learning purposes. For production use, consider using more advanced architectures and larger datasets.


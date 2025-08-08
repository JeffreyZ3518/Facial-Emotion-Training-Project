# Facial Emotion Recognition Project

This project is part of my NVIDIA AI work, focused on training a deep learning model to classify human facial emotions using the **Jetson Orin Nano** platform. It includes both the training code and a demonstration video.

## Overview

The goal of this project was to create a model capable of recognizing **seven distinct emotions**:  
**Happy, Angry, Sad, Disgusted, Fearful, Neutral,** and **Surprised**.

### Tools and Technologies Used:
- Jetson Orin Nano for model training
- Linux environment
- Python
- Visual Studio Code
- Docker
- PyTorch (with ONNX model export)

## Datasets

The model was trained on facial expression datasets sourced from Kaggle:

- FER2013 Dataset:  
  https://www.kaggle.com/datasets/msambare/fer2013

- Emotion Detection FER Dataset:  
  https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

The data was preprocessed and formatted for compatibility with the NVIDIA Jetson inference training framework.

## Model Training

The model was trained in two iterations:

- **First training**: The initial model had limited accuracy when identifying emotions.
- **Second training**: We expanded the dataset, increased the number of training epochs, and improved preprocessing. This significantly enhanced model performance.

## Setup and Training Instructions

1. **Organize the Dataset**  
   Store the image data in an appropriate folder structure inside the Jetson Nano device.

2. **Launch the Docker Container**  
   ```bash
   cd ~/jetson-inference/
   ./docker/run.sh

3. **Navigate to the Classification Training Directory**  
   ```bash
   cd python/training/classification

4. **Train the Model**  
   ```bash
   python3 train.py --model-dir=models/emotion data/emotion

5. **Convert the Trained Model to ONNX Format**  
   ```bash
   python3 onnx_export.py --model-dir=models/emotion

6. **Save Model and Dataset Paths**  
   ```bash
   NET=models/emotion
   DATASET=data/emotion

7. **Run Inference on a Test Image**  
   ```bash
   imagenet.py \
    --model=$NET/resnet18.onnx \
    --labels=$NET/labels.txt \
    --input_blob=input_0 \
    --output_blob=output_0 \
    $DATASET/test/angry/angry.jpg angryOutput.jpg

## Setup and Training Instructions

After the second training phase—using a larger dataset and more epochs—the AI model demonstrated reliable performance in classifying the seven targeted emotions from facial images.

## Future Improvements

 - Incorporating real-time emotion detection
 - Exploring more advanced neural network architectures
 - Applying additional data augmentation or fine-tuning

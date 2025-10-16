# SARS-CoV-2 Detection in Genosensor Images via Deep Learning and Data Augmentation

This repository contains the official implementation of the paper:
> SARS-CoV-2 Detection in Genosensor Images via Deep Learning and Data Augmentation
> Accepted at the XX Workshop on Computer Vision (WVC), 2025.

## Overview
This study explores deep learning models for detecting SARS-CoV-2 sequences using genosensor images.
It builds upon the pioneering work of Soares et al.,  [Detection of a SARS-CoV-2 sequence with genosensors using data analysis based on information visualization and machine learning techniques](https://pubs.rsc.org/en/content/articlelanding/2021/qm/d1qm00665g), who introduces image-based detection for genosensors using scanning electron microscopy (SEM) images. 

Our work extends this approach by systematically exploring multiple CNN architectures and a variety of data augmentation strategies. By synthetically increasing the dataset size, we aim to enable DL models to learn more robust and generalizable features, surpassing both classical methods and previous DL attempts.

## Repository Structure
```
SARS-CoV-2-Detection-in-Genosensor-Images/
│
├── src/
│   ├── data.py           # Dataset loading 
│   ├── augmentations.py  # Data augmentation definitions
│   └── cnn.py            # Training and testing functions
│
├── experiments/
│   ├── experiments.py                     # Main experiment script
│   ├── gradcam.py                         # Grad-CAM visualization for models
│   ├── classification_reports_summary.py  # Aggregates classification reports
│   ├── results/                           # Output JSONs and reports
│   └── model_weights/                     # Saved model weights (.pt)
│
└── README.md
```

## Usage
This repository does not include the original genosensor dataset since it is not public. 

However, you can download the pre-trained .pt weights from [this link](https://drive.google.com/drive/folders/17lUfFH-nZ14RtNyJpacnObFrVW3W_JSF?usp=sharing). This allows replicating the experiments exactly or adapting the models for related genosensor image classification tasks.

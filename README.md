# Fashion Classification

## Overview
This repository contains a study on image classification using the Fashion MNIST dataset. It explores the effectiveness of custom Convolutional Neural Networks (CNNs) and pre-trained models for this task.

## Repository Composition

data/: Contains the Fashion MNIST dataset.
models/: Stores trained models for future use.
notebooks/: Jupyter notebooks used for experimentation and analysis.
scripts/: Source code including model definitions and utility functions.
docs/: Project documentation in Italian.
README.md: This file, providing an overview of the project.

## Dataset
Fashion MNIST consists of grayscale images of clothing items categorized into 10 classes. Each image is 28x28 pixels, making it suitable for benchmarking image classification algorithms.

## Approach
Two main approaches are explored:
- Custom CNN: A CNN architecture specifically designed for Fashion MNIST.
- Pre-trained Model: Fine-tuning the ResNet18 model pre-trained on ImageNet for Fashion MNIST.

## Experiments
The study involves training and evaluating both models using various configurations and optimization techniques. Performance metrics such as accuracy and confusion matrices are analyzed.

## Results
The custom CNN achieves an accuracy of 92.88% on the test set, outperforming the pre-trained ResNet18 model, which achieves 90.91% accuracy.

## Repository Composition

data/: Contains the Fashion MNIST dataset.
models/: Stores trained models for future use.
notebooks/: Jupyter notebooks used for experimentation and analysis.
scripts/: Source code including model definitions and utility functions.
docs/: Project documentation in Italian.
README.md: This file, providing an overview of the project.

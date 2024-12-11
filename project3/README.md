# Comparative Analysis of Machine Learning Models for Image Classification

## Project Overview

This repository contains the codebase and assets for the project titled:
**Comparative Analysis of Machine Learning Models for Image Classification: From Traditional Methods to Deep Learning Architectures**.

The project evaluates several machine learning models on image classification tasks, focusing on their accuracy, computational efficiency, and scalability.

## Repository Structure

```
| - models
|    |- cnn_rnn_model.keras       # Pretrained CNN-RNN model file
|    |- ffnn_model.keras          # Pretrained FFNN model file
|    |- rnn_model.keras           # Pretrained RNN model file
|    |- vggcnn_model.keras        # Pretrained VGG-based CNN model file
|
| - cnn.py                        # Code for Convolutional Neural Network (CNN) implementation
| - optimisers.py                 # Optimizers and utility functions for training
| - svm.py                        # Code for Support Vector Machine (SVM) implementation
| - utils.py                      # General utility functions for data preprocessing and evaluation
| - Project3.ipynb                # Jupyter Notebook for the project's main analysis and experiments
| - Project3_Oscar_Qian.pdf       # Final report summarizing findings and methodology
| - requirements.txt              # Python dependencies required for the project
```

## Requirements

To run the code, ensure you have Python 3.8 or later installed. Use the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Files and Usage

### 1. **cnn.py**

Contains the implementation of the Convolutional Neural Network (CNN). This file includes model architecture, training loops, and evaluation methods.

### 2. **optimisers.py**

Defines custom optimizers and other helper functions used for model training and optimization.

### 3. **svm.py**

Includes the implementation of a Support Vector Machine (SVM) for image classification tasks. Uses Scikit-learn for core functionalities.

### 4. **utils.py**

A utility script that provides functions for:

- Data preprocessing (e.g., image resizing, normalization)
- Evaluation metrics computation (e.g., accuracy, precision)
- Plotting results

### 5. **Project3.ipynb**

A Jupyter Notebook that brings together the entire workflow, including:

- Data loading and preprocessing
- Training and evaluation of different models
- Comparative analysis and visualization of results

### 6. **models/**

This folder contains the pretrained models saved in `.keras` format for reuse.

### 7. **Project3_Oscar_Qian.pdf**

The research report summarizing the comparative analysis, methodology, and results.

### 8. **requirements.txt**

Lists all the Python libraries required to run the project. Install these to set up the environment.

## Running the Code

### Step 1: Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2: Install Dependencies

Run the following command to install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebook

Launch the Jupyter Notebook to explore the experiments and results:

```bash
jupyter notebook Project3.ipynb
```

### Step 4: Analyze Results

Use the Jupyter Notebook or scripts to visualize the performance metrics and understand the trade-offs between different machine learning models.

## Pretrained Models

The `models/` directory contains pretrained model weights for CNN, RNN, FFNN, and VGG-based architectures. These can be loaded directly to save time during evaluation.

## License

This project is licensed under the MIT License.

## Author

Oscar Qian

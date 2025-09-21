# Handwritten-Digit-Recognition_Kaggle
This is a self-directed deep learning project dedicated to handwritten digit recognition—my key goal was to master Convolutional Neural Network (CNN) architectures, deep learning workflows, and model optimization through end-to-end implementation. By working with the classic MNIST dataset, I built, trained, and refined a CNN model to accurately classify handwritten digits (0-9), while solidifying core deep learning concepts.

# Project Overview
The project targets high-precision classification of handwritten digits using a custom-built CNN. Leveraging the MNIST dataset (grayscale images of digits 0-9), I focused on designing CNN architectures from scratch, rather than using pre-built models, to deepen my understanding of how convolutional layers, pooling layers, and fully connected layers work together to extract spatial features from image data.

# Key Workflow & Deep Learning Skill Development
### 1. Data Preprocessing
- Loaded and explored the MNIST dataset, analyzing image dimensions (28x28 grayscale) and class distribution.
  
- Normalized pixel values from [0, 255] to [0, 1] to stabilize model training and accelerate convergence.
  
- Reshaped image data to include a channel dimension (required for CNN input: (no. of samples, height, width, channels)).
  
- Split data into training and validation sets to evaluate model generalization.

### 2. CNN Architecture Design & Implementation
I iteratively designed and tested network structures, learning how layer choices impact performance:

- **Convolutional Layers**: Added 2 sequential convolutional layers to extract low-level (edges, lines) and high-level (digit shapes) spatial features.
  
- **Pooling Layers**: Integrated MaxPooling2D layers after each convolutional layer to reduce spatial dimensions, prevent overfitting, and lower computational cost.
  
- **Regularization**: Included Dropout layers to mitigate overfitting by randomly deactivating a subset of neurons during training.
  
- **Fully Connected Layers**: Added dense layers to map extracted features to 10 digit classes, with softmax activation for probability output.

All layers were implemented using TensorFlow/Keras, with careful tuning of hyperparameters (e.g., activation functions, kernel initializers) to optimize feature extraction.


### 3. Model Training & Optimization
- Trained the CNN using Adam optimizer and Sparse Categorical Crossentropy loss.
  
- Implemented early stopping to halt training if validation accuracy stopped improving, avoiding unnecessary computation.
  
- Evaluated model performance on the test set and iterated on architecture to boost accuracy.

# Results
- Final CNN model achieved 99.014% accuracy on the MNIST test set.
- Key takeaway: CNNs excel at image tasks because their hierarchical layer structure efficiently captures spatial patterns.

# Technologies Used
- Python (NumPy, Pandas for data handling)
- TensorFlow/Keras (for CNN building, training, and evaluation)
- Matplotlib (for visualizing training curves, misclassified images, and feature maps)
- Scikit-learn (for data splitting and performance metrics)

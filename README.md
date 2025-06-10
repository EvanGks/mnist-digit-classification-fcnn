# 🧠 MNIST Classification with Fully Connected Neural Network (FCNN)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📈 Live Results

You can view the notebook with all outputs and results on Kaggle:
[https://www.kaggle.com/code/evangelosgakias/fcnn-image-classification-tensorflow](https://www.kaggle.com/code/evangelosgakias/fcnn-image-classification-tensorflow)

---

## 📑 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [Sample Visualizations](#sample-visualizations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📋 Overview
This project demonstrates how to build, train, and evaluate a fully connected neural network (Multi-Layer Perceptron, MLP) for image classification using the MNIST dataset. The implementation leverages TensorFlow and Keras to construct a deep learning model that learns to recognize handwritten digits (0–9). The notebook also includes systematic hyperparameter tuning using Keras Tuner to optimize model performance.

- **Dataset:** MNIST (60,000 training images, 10,000 test images, 28x28 grayscale, 10 classes)
- **Goal:** Classify handwritten digits with high accuracy using a robust, regularized MLP
- **Skills Showcased:** Data preprocessing, model design, training, evaluation, hyperparameter tuning, visualization, and analysis

---

## 🏗️ Project Structure
```
.
├── FNN.ipynb         # Jupyter notebook with the complete implementation
├── requirements.txt  # Python dependencies
├── LICENSE           # MIT License
├── README.md         # Project documentation (this file)
└── hyperparameter_tuning/
    └── mnist_tuning/ # Keras Tuner search results
```

---

## 🚀 Features
- **Data Preparation:**
  - Automatic download and loading of the MNIST dataset
  - Normalization of pixel values to [0, 1]
  - Train/validation/test split (80%/20%/test)
  - Visualization of sample images
- **Model Architecture:**
  - Input flattening (28x28 → 784)
  - Two dense layers with ReLU activation and L2 regularization
  - Dropout layers for regularization
  - Output layer with softmax activation for multi-class classification
- **Training Process:**
  - Adam optimizer, sparse categorical cross-entropy loss
  - Training with validation monitoring
  - Visualization of accuracy and loss curves
- **Evaluation & Visualization:**
  - Metrics: Accuracy, Loss
  - Confusion matrix
  - Sample predictions with true vs. predicted labels
  - Visualization of misclassified examples
- **Hyperparameter Tuning:**
  - Keras Tuner for dense units, dropout rates, L2 values, and optimizer
  - Comparison of original and fine-tuned models
  - Visual and tabular performance comparison

---

## ⚡ Usage

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/EvanGks/mnist-digit-classification-fcnn.git
   cd mnist-digit-classification-fcnn
   ```
2. **Create and activate a virtual environment:**
   - **On Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook FNN.ipynb
   ```
5. **Run the notebook cells in order:**
   - The notebook is organized sequentially with explanations and visualizations.
   - All outputs and plots will be displayed inline.

---

## 📊 Results

The model achieves the following performance (see [Kaggle notebook](https://www.kaggle.com/code/evangelosgakias/fcnn-image-classification-tensorflow) for full details):

| Model              | Test Accuracy | Test Loss |
|--------------------|--------------|-----------|
| Original Model     | ~98%         | ~0.07     |
| Fine-Tuned Model   | ~98.2%       | ~0.06     |

- **Validation accuracy and loss closely track training metrics, indicating good generalization.**
- **Confusion matrices show most predictions are correct, with errors concentrated in visually ambiguous digits.**
- **Fine-tuned model offers a slight but consistent improvement over the baseline.**

---

## 🖼️ Sample Visualizations

- **Training and Validation Accuracy/Loss Curves:**
  - Show convergence and help detect overfitting.
- **Confusion Matrix:**
  - Visualizes correct vs. incorrect predictions for each digit.
- **Sample Predictions:**
  - Grid of test images with true and predicted labels, color-coded for correctness.
- **Misclassified Examples:**
  - Highlights challenging cases for the model, useful for error analysis.

_See the notebook and [Kaggle live version](https://www.kaggle.com/code/evangelosgakias/fcnn-image-classification-tensorflow) for all plots and outputs._

---

## 🛠️ Future Improvements
- Experiment with deeper or alternative architectures (e.g., CNNs)
- Apply data augmentation to increase robustness
- Explore additional regularization (early stopping, L1/L2 variants)
- Try different optimizers and learning rate schedules
- Integrate TensorBoard for richer training visualization
- Expand to other datasets or multi-task learning

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you would like to change.

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact
For questions or feedback, please reach out via:

- **GitHub:** [EvanGks](https://github.com/EvanGks)
- **X (Twitter):** [@Evan6471133782](https://x.com/Evan6471133782)
- **LinkedIn:** [Evangelos Gakias](https://www.linkedin.com/in/evangelos-gakias-346a9072)
- **Kaggle:** [evangelosgakias](https://www.kaggle.com/evangelosgakias)
- **Email:** [evangks88@gmail.com](mailto:evangks88@gmail.com)

---

Happy Coding! 🚀

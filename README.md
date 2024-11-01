# Music Genre Classification

<p align="justify">
Experts have been trying for a long time to understand music and what differentiates one from the other. Computational techniques that can achieve human level accuracy on classifying a music genre that provides quick and cost-effective solution to various music application companies such as Apple Music, Spotify or Wynk ets, will play a major role in increasing customer satisfaction and building a robust recommender system. Throughout the course of the project, we will explore how various Neural Networks and DL paradigms like ANN and CNNs can be applied to the task of music genre classification.  
</p>

> 📖 **Related Article**: [Check out the companion article on Medium for an in-depth walkthrough](https://medium.com/@teendifferent/my-exploration-into-genre-classification-with-deep-learning-a-project-revisited-34d8cb631b1b).

---

## 🎯 Project Overview

The goal of this project is to classify music genres using both conventional and deep learning techniques. We perform the following steps:
1. **Feature Extraction**: Extract musical features from audio files (e.g., MFCC, spectral contrast).
2. **Modeling**: Train machine learning and deep learning models, including ANN, CNN, and Transfer Learning.
3. **Evaluation**: Assess each model's performance, comparing architectures and their accuracies.

By applying deep learning to music information retrieval, this project also aims to explore how different architectures capture the nuances of audio data and genre characteristics.

## 📁 Dataset Description

This project uses the **GTZAN Genre Collection** dataset, containing:
1. **Audio Files**: 10 genres (e.g., rock, pop, jazz), with 100 audio files per genre, each 30 seconds in length.
2. **Spectrogram Images**: Generated images representing the frequency content of each audio file.
3. **Feature CSV Files**: Two CSV files provide extracted audio features (mean and variance for each 3-second segment).

### Genres and Sample Counts
| Genre     | Count |
|-----------|-------|
| Blues     | 1000  |
| Classical | 1000  |
| Country   | 1000  |
| Disco     | 999   |
| Hip Hop   | 998   |
| Jazz      | 1000  |
| Metal     | 1000  |
| Pop       | 1000  |
| Reggae    | 1000  |
| Rock      | 998   |

---

## 🎛 Audio Feature Extraction

Using the `librosa` library, we perform feature extraction on each audio file to obtain the following attributes:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**: Represent audio timbre, allowing for genre-specific identification.
2. **Spectral Centroid**: Indicates the "center of mass" of the spectrum, providing insight into brightness.
3. **Spectral Bandwidth**: Describes the width of the frequency spectrum.
4. **Zero-Crossing Rate**: Measures the rate of signal sign changes, useful in distinguishing percussion-heavy genres.

The **Mel-spectrogram** images serve as input for Convolutional Neural Networks (CNNs), which excel at capturing spatial data patterns in spectrograms.

---

## 🧠 Models Implemented

### 1. Artificial Neural Network (ANN)
The first model is a simple ANN trained on extracted features. This initial network achieved a **validation accuracy of 72%**. 

*Model Architecture*:
- **Layers**: Fully connected dense layers with dropout for regularization
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam

### 2. Convolutional Neural Network (CNN)
For the second model, we implement a custom CNN that classifies Mel-spectrogram images of each audio file. This CNN learns spatial relationships within spectrograms, aiming for a more granular genre classification.

*Model Architecture*:
- **Layers**: Convolutional layers with MaxPooling
- **Dense Layers**: Fully connected dense layers following feature extraction
- **Accuracy**: **63% validation accuracy**

### 3. Transfer Learning Models
We experiment with several pre-trained models, adapting their architectures for genre classification.

| Model           | Description                                          | Validation Accuracy |
|-----------------|------------------------------------------------------|---------------------|
| **VGG19**       | Known for its depth and high accuracy in image tasks | 58%                |
| **EfficientNetB0** | Balances accuracy with computational efficiency      | 62%                |
| **InceptionV3** | Handles complex features and performs well on images | 55%                |
| **MobileNetV2** | Lightweight and mobile-friendly                      | 51%                |

These models showcase how transfer learning can offer a strong baseline and potential for further tuning.

---

## 📊 Results and Analysis

| Model           | Training Accuracy | Validation Accuracy | Test Accuracy |
|-----------------|-------------------|---------------------|---------------|
| ANN             | 99%               | 72%                | 72%           |
| Custom CNN      | 82%               | 63%                | 62%           |
| VGG19           | 80%               | 58%                | 58%           |
| EfficientNetB0  | 85%               | 62%                | 62%           |
| MobileNetV2     | 83%               | 51%                | 51%           |
| InceptionV3     | 82%               | 55%                | 55%           |

The results show that CNNs perform better with spectrograms than traditional ANNs. However, Transfer Learning models need further tuning to fully utilize the potential of complex architectures. **EfficientNetB0** performed well, offering a balance between efficiency and accuracy, ideal for mobile applications.

---

## 🚀 Further Improvements

1. **Hyperparameter Tuning**: Fine-tune the hyperparameters of transfer learning models for improved accuracy.
2. **Additional Features**: Explore spectral contrast, tonal centroid features, and harmonic/percussive source separation for enhanced feature representation.
3. **Data Augmentation**: Experiment with audio data augmentation (e.g., time-stretching, pitch-shifting) to increase model robustness.
4. **Model Ensembles**: Combine predictions from multiple models for improved generalization.

---

## 📚 References and Acknowledgements

- **Librosa**: https://librosa.org/doc/latest/index.html
- **GTZAN Dataset**: https://marsyas.info/downloads/datasets.html
- **TensorFlow and Keras Documentation**: https://www.tensorflow.org/api_docs

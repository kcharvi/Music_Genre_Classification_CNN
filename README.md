# Music Genre Classification

<p align="justify">
Experts have been trying for a long time to understand music and what differentiates one from the other. Computational techniques that can achieve human level accuracy on classifying a music genre that provides quick and cost-effective solution to various music application companies such as Apple Music, Spotify or Wynk ets, will play a major role in increasing customer satisfaction and building a robust recommender system. Throughout the course of the project, we will explore how various Neural Networks and DL paradigms like ANN and CNNs can be applied to the task of music genre classification.  
</p>

Modelling Stage: 

### 1)	Artificial Neural Networks

We created a Sequential model and added layers. The first thing to get right is to ensure the input layer has the right number of input features.

>  This can be specified when creating the first layer with the <font color="red">input_dim </font> argument. Generally, we need a network large enough to capture the structure of the problem. While compiling we must specify the loss function to use to evaluate a set of weights, the optimizer is used to search through different weights for the network.

>  We have used several Dense and Dropout layers with activation function as 'relu', and Dropout of 0.2 input_shape=(X_train.shape [1],)), and the output layer uses ‘softmax’.

>  We train or fit our model on our loaded data by calling the fit() function on the model.
Training occurs over epochs and each epoch is split into batches.


-	**Epoch: One pass through all of the rows in the training dataset.**

-	**Batch: One or more samples considered by the model within an epoch before weights are updated.**


>  We have given 600 epochs with batch size as 128, further more these parameters could be changed to compare and obtain a better accuracy as well.
The evaluate() function will return a list with two values. The first will be the loss of the model on the dataset and the second will be the accuracy of the model on the dataset. 
```sh
loss='sparse_categorical_crossentropy', metrics='accuracy'
```

>  We could change the layers and improve the accuracy further by expanding the network until it identifies the pattern of the dataset.

>  We have done two different models for these parameters in layers: Adam optimizer, RELU activation function, softmax function, sparse_categorical_crossentropy function, Dropout.

### 2)	Convolution Neural Networks

A CNN is a Deep Learning algorithm which can take in an input image, assign importance, learnable weights and biases to various aspects or objects in the image and be able to differentiate one from the other. 

>  The TARGET_SIZE =224 is set and the input of each image is scaled down using 
```sh
preprocessing.Rescaling(1./255, input_shape=(TARGET_SIZE, TARGET_SIZE, 3) 
```

>  The different genre images are similar to the following, with the variations in the Mel spectrograms.

2D convolution layer: this layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.

-	filters: the dimensionality of the output space (i.e. the number of output filters in the convolution), used= 16,32,64 

-	kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions, used= 3.

-	Strides and padding: are set to default.

MaxPooling layer: is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map. 
 
### 3)	Transfer Learning

In a CNN, as the number of layers increase, so does the ability of the model to fit more complex functions. Transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem. It is usually done for tasks where our dataset has too little data to train a full-scale model from scratch. It has the advantage of decreasing the training time for a learning model and can result in lower generalization error. We are using the pre-trained weights of several famous architectures, change the output layer and solve the classification problem on our music dataset. 

We have performed our classification on different pre-trained weights of ImageNet, that includes, VGG19(Visual Geometry Group19) – 19 layers deep , EfficientNetB0 - 237 layers deep,  MobileNetV2,- 53 layers deep, InceptionV3.  48 layers deep.

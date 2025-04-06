# CNN-Image-Classifier
This is a Convolutional Neural Network implementation that classifies card images into 53 classes(4 suits * 13 cards + joker). The dataset consists of ~120 training examples for each card.
# Python libraries used
Pandas library is used to parse the csv data. Matplotlib is used to make graphs and visualizations about the data and results. PyTorch is used for the creation of the neural network.
# About the Network architecture
## Data augmentation
Firstly, data augmentation is used to expand the training data. The transformations aren't very heavy - slight rotation of the train images is applied at random, color jitter is introduced to show the model different lightning and contrast. Input image data is normalized - subtracted mean and divided by standard deviation to ensure faster convergence and even weight updates during training. The input data is then converted to a pytorch tensor.
## DataSet and DataLoader
To use the Pandas dataframe, which is the parsed csv data, MyDataset inherits Dataset, such that it can be used by the DataLoader to load the training data into training batches.
## Loss and Optimizer
Since this is a classification problem, cross entropy loss is used. Adam optimizer is used for the stochastic gradient descend algorithm.
## Neural Network Architecture
The architecture consists of 5 convolutional layers, as well as Batch Norm(to normalize data after convolution) and Max Pooling(to reduce the image size) layers between them. The input images are of shape 3x224x224 (3 channels, 244x244 W/H). During these layers, channels are extended and width and height are reduced. The output of these layers (shape is 256x7x7) is then flattened and passed to a regular fully connected layer. This layer's output features are 512, and then it is passed to another fully connected layer, which then resembles the probability distribution of each class. The total parameters of the model are around 6.8 million.
### NOTE: After every convolutional and linear layer, ReLU activation function is used to introduce non-linearity.
## Possible tweaks to the model
As of now, the model recognises the card images with 95% accuracy, which is pretty good, but this could potentially be increased.
### Dropout
Dropout with probability 0.3 is currently used after each fully connected layer. Maybe higher dropout could result in better results?
### Learning rate
Maybe dynamic learning rate which goes down during training could be implemented? Currently it is fixed 0.005.
### Weight decay
Maybe AdamW could be used to introduce regularization and improve model performance? Try different optimizers as well.
### Batch size
Currently the batch size is 15 samples. Maybe the sample size could be experimented with, increasing it could help.
### Model Architecture
The 5 convolutional and 2 linear layers works good, but maybe this could be experimented with.
### Residual/skip connections
Experiment with adding residual/skip connections to the model.

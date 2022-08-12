# Neural Network from scratch, applied to novel dataset
This learning algorithm was created with no references to any existing Machine Learning libraries. The neural network framework
was adapted from The One on youtube - https://youtu.be/L_PByyJ9g-I.

A TanH activation function was used with a adjustable learning rate and number of feed forward / back propagation iterations.

The number of hidden layers along with neurons in each layer can be set when constructing the NeuralNetwork class. 
The ideal font colour was chosen for 100 different background colours, with effort to reduce biases across the colour 
spectrum. 

## Application - Font colour prediction
Console application in C# using neural network with to predict the ideal font colour (black or white) to display over 
any given background colour (RGB representation).

100 training samples were generated manually to train the network with. The background was defined by red, green, blue values of the range 0-255 and ideal font colour 0 for black, 1 for white.

This list of 100 entries was shuffled in excel then split into 75 lines for training data and 
25 for test data. Red, green and blue values (0-255) were normalized (0-1).

Some large errors exist due to the at times subjective nature of background / font aesthetics.

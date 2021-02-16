import os
import numpy as np
import tensorflow as tf
import random

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from math import e, floor
from statistics import mean

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"




class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        # self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        # self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.NN = NeuralNetwork_NLayer(self.inputSize,self.outputSize,[self.neuronsPerLayer],self.lr)



    # Activation function.
    def __sigmoid(self, x):
        return self.NN.__sigmoid(x)

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return self.NN.__sigmoidDerivative(x)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        self.NN.train(xVals,yVals,epochs,minibatches,mbs,flatten=True,shuffle=True)

    # Forward pass.
    def __forward(self, input):
        # layer1 = self.__sigmoid(np.dot(input, self.W1))
        # layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        # return layer1, layer2
        return

    # Predict.
    def predict(self, xVals):
        return self.NN.predict(xVals)


class NeuralNetwork_NLayer():
    def __init__(self, inputSize, outputSize,hiddenLayerNeurons,learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.nLayers = len(hiddenLayerNeurons) + 1
        self.lr = learningRate
        self.weights = []
        self.biases = []
        for i,num in enumerate(hiddenLayerNeurons):
            if (i == 0):
                self.weights.append(np.random.randn(self.inputSize, num) / sqrt(num))
            else:
                self.weights.append(np.random.randn(hiddenLayerNeurons[i-1], num) / sqrt(num))

            self.biases.append(np.random.randn(num) / sqrt(num))

        self.weights.append(np.random.randn(hiddenLayerNeurons[len(hiddenLayerNeurons) - 1], self.outputSize) / sqrt(self.outputSize))
        self.biases.append(np.random.randn(self.outputSize) / sqrt(self.outputSize))

        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)

    # Activation function.
    def __sigmoid(self, x):
        return (e**x)/(e**x + 1)

    # Activation prime function.
    def __sigmoid_derivative(self, x,sig_calc):
        return self.__sigmoid(x)*(1 - self.__sigmoid(x)) if not sig_calc else x * (1-x)

    def __mse(self,y,yhat):
        return mean([0.5 * (y-yhat)**2 for y,yhat in zip(y,yhat)])

    def __mse_derivative(self,y,yhat):
        return -1 * (y - yhat)

    # Batch generator for mini-batches. Not randomized.
    def __batch_generator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100,flatten=True,shuffle=True):
        if shuffle:
            zipped = list(zip(xVals, yVals))
            random.shuffle(zipped)
            xVals, yVals = zip(*zipped)    
    
        # batches_x = self.__batch_generator(xVals,mbs)
        # batches_y = self.__batch_generator(yVals,mbs)
        split_to_batches = lambda A, n=mbs: [A[i:i+n] for i in range(0, len(A), n)]
        batches_x = split_to_batches(xVals)
        batches_y = split_to_batches(yVals)
        # print(len(list(batches_x)))

        # swch = True

        for epoch in range(epochs):
            batch_ctr = 0
            print("Epoch ",epoch+1)
            for batch_x,batch_y in zip(batches_x,batches_y):
                averaged_deltas = []
                for x,y in zip(batch_x,batch_y):
                    if flatten:
                        x = x.flatten()
                        y = y.flatten()
                    layer1, layer2 = self.__forward(x)
                    layers = [layer1,layer2]
                    deltas = self.__back(x,layers,[self.W1,self.W2],y)
                    if (len(averaged_deltas) == 0):
                        averaged_deltas = np.array(deltas)
                    else:
                        averaged_deltas += np.array(deltas)
                averaged_deltas /= mbs
                self.W1 -= self.lr * averaged_deltas[1]
                self.W2 -= self.lr * averaged_deltas[0].reshape((self.neuronsPerLayer,self.outputSize))
                
                batch_ctr+=1
                printout = "batch "+str(batch_ctr)+"/"+str(len(batches_x))
                print(printout, end='\r')
            print()
        # return
    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2
    
    def __back(self,input,layers,weights,truths):
        layers_deltas = [0 for i in range(self.nLayers)]
        out_layer = layers[len(layers)-1]
        output_layer_calc = True

        error_deltas = []
        for pred,truth in zip(out_layer,truths):
            error_deltas.append(self.__mse_derivative(truth,pred) * self.__sigmoid_derivative(pred,True))
        error_delta = mean(error_deltas)

        for i,layer in reversed(tuple(enumerate(layers))):
            deltas = []
            if output_layer_calc:
                for neuron in layers[i-1]:
                    deltas.append(error_delta * neuron)
                output_layer_calc = False
                layers_deltas[i] = np.array(deltas)
            else:
                for j in range(len(layer)):
                    weights_flattened = (weights[i+1]).flatten()
                    derivative_respect_zhidden = error_delta * weights_flattened[j]
                    hidden_derivative = derivative_respect_zhidden * self.__sigmoid_derivative(layer[j],sig_calc=True)
                    prev_layer = input if (i-1 < 0) else layers[i-1]
                    sub_deltas = []
                    for k in range(len(layers[i-1])):
                        sub_deltas.append(hidden_derivative * layers[i-1][k])
                    deltas.append(np.array(sub_deltas))
                layers_deltas[i] = np.transpose(deltas)
        return layers_deltas

    # Predict.
    def predict(self, xVals,flatten=True):
        if flatten:
            xVals = xVals.flatten()
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = np.array([tr/255 for tr in xTrain])
    xTest = np.array([ts/255 for ts in xTest])
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data

    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        # xTrain = xTrain.flatten()
        # yTrain = yTrain.flatten()
        custom_nn = NeuralNetwork_2Layer(784, 10, 512, learningRate = 0.01)
        custom_nn.train(xTrain,yTrain,epochs=10,mbs=500)
        print("Trained")                   #TODO: Write code to build and train your custon neural net.
        return custom_nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")


def binarize_output(pred):
    idx = np.argmax(pred)
    pred[idx] = 1.0
    floor_v = np.vectorize(floor)
    return floor_v(pred)

def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        # print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        preds = []
        for d in data:
            pred = binarize_output(model.predict(d))
            # print(pred)
            preds.append(pred)
        return np.array(preds)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    # print(preds[0])
    # print(data[1][1][0])
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
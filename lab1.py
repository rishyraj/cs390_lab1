import os
import numpy as np
import tensorflow as tf
import random
import sys
import argparse

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from math import e, floor, sqrt
from statistics import mean

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_net"
# ALGORITHM = "custom_net_3layer"
# ALGORITHM = "custom_net_nlayer"
# ALGORITHM = "tf_net"

ALGORITHM = ""



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
        # for bias in self.biases:
        #     print(bias)

    def progress_bar(self,progress,goal,subdivisions=32):
        modulo_factor = goal / subdivisions
        # if (goal % subdivisions != 0):
            # subdivisions += 1
        
        bar_graphic = "█"
        p_bar = "|"
        # if (progress % modulo_factor == 0):
        bars = int(progress/modulo_factor)
        spaces = subdivisions - bars
        for i in range(subdivisions):
            if (i < bars and bars != 0):
                p_bar+=bar_graphic
            else:
                p_bar+=" "
        
        if (goal % subdivisions != 0):
            if (progress != goal):
                p_bar += " "
            else:
                p_bar += bar_graphic

        p_bar += "|"
        return p_bar

    # Activation function.
    def __sigmoid(self, x):
        return (e**x)/(e**x + 1)

    # Activation prime function.
    def __sigmoid_derivative(self, x,sig_calc=True):
        return self.__sigmoid(x)*(1 - self.__sigmoid(x)) if not sig_calc else x * (1-x)

    def __mse(self,y,yhat):
        # return mean([0.5 * (y-yhat)**2 for y,yhat in zip(y,yhat)])
        return np.mean(np.square(y-yhat))

    def __mse_derivative(self,y,yhat):
        return -1 * (y - yhat)

    # Batch generator for mini-batches. Not randomized.
    def __batch_generator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100,flatten=True,shuffle=True,testData=None):

        if flatten:
            x_flattened = []
            y_flattened = []
            for x,y in zip(xVals,yVals):
                x_flattened.append(x.flatten())
                y_flattened.append(y.flatten())
            xVals = np.array(x_flattened)
            yVals = np.array(y_flattened)
        
        
        if shuffle:
            zipped = [(x,y) for x,y in zip(xVals,yVals)]
            random.shuffle(zipped)
            xVals = np.array([x for x,y in zipped])
            yVals = np.array([y for x,y in zipped])
    
        batches_x = np.array([x for x in self.__batch_generator(xVals,mbs)])
        batches_y = np.array([y for y in self.__batch_generator(yVals,mbs)])

        print("\n")
        loss = 0
        for epoch in range(epochs + 1):
            progress = 0
            p_bar_1 = self.progress_bar(epoch,epochs,subdivisions=(20 if (epochs / 2 > 32) else int(epochs / 2)))
            p_bar_2 = self.progress_bar(progress,len(xVals),subdivisions=(20 if (len(xVals) / 2 > 32) else len(xVals) / 2))
            # print()
            if (epoch == epochs):
                progress = len(xVals)
                p_bar_2 = self.progress_bar(progress,len(xVals),subdivisions=(20 if (len(xVals) / 2 > 32) else len(xVals) / 2))
                # print("Epochs: ",p_bar_1," ",epoch,"/",epochs,"\nRecords Trained: ",p_bar_2," ",progress,"/",len(xVals),end="\r\n")
                print("\033[FEpochs:          ",p_bar_1," ",epoch,"/",epochs," loss: ",loss,"\nRecords Trained: ",p_bar_2,progress,"/",len(xVals),end="",sep="",flush=True)
                break
            for batch_x,batch_y in zip(batches_x,batches_y):
                averaged_deltas = []

                before_activation,after_activation = self.__forward(batch_x)
                # deltas_weights = self.__back(batch_x,before_activation,after_activation,batch_y,mbs)
                deltas_weights,deltas_biases = self.__back(batch_x,before_activation,after_activation,batch_y,mbs)
                
                # print(np.array(deltas).shape)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * deltas_weights[i]
                    # print(self.biases[i].shape,deltas_biases[i])
                    self.biases[i] -= self.lr * deltas_biases[i]

                progress += len(batch_x) 
                p_bar_2 = self.progress_bar(progress,len(xVals),subdivisions=(20 if (len(xVals) / 2 > 32) else len(xVals) / 2))
                loss = self.__mse(after_activation[-1],batch_y)
                print("\033[FEpochs:          ",p_bar_1," ",epoch,"/",epochs," loss: ",loss,"\nRecords Trained: ",p_bar_2,progress,"/",len(xVals),end="",sep="",flush=True)

        print()
        return

    # Forward pass.
    def __forward(self, input,activation='sigmoid'):
        if (activation == 'sigmoid'):
            activation = self.__sigmoid
        before_activation = []
        after_activation = []

        a = input
        
        for i in range(len(self.weights)):
            weight = self.weights[i]
            bias = self.biases[i]

            # z = np.dot(a,weight)
            z = np.dot(a,weight) + bias
            a = activation(z)

            before_activation.append(z)
            after_activation.append(a)

            # print(z.shape,a.shape)
        
        return before_activation,after_activation
            
    def __back(self,X,x_b,x_a,y,batchSize):
        dLoss = 0
        dA = 0
        dZ = 0

        deltas_weights = [0 for i in range(len(x_b))]
        deltas_biases = [0 for i in range(len(x_b))]

        x_a.insert(0,X)
        x_b.insert(0,X)

        # print(x_a)
        for i in range(len(x_a) - 1,0,-1):
            # print(len(self.weights))
            a,a_prev,z = x_a[i],x_a[i-1],x_b[i]
            weights = self.weights[i - 1]

            if (i == len(x_a) - 1):
                dLoss = self.__mse_derivative(y,a)
                # print("dLoss:",dLoss.shape)
                dZ = np.multiply(dLoss,self.__sigmoid_derivative(a))
                # print(dZ.shape)
            else:
                dZ = np.multiply(dA, self.__sigmoid_derivative(a))
            
            # print("dZ",dZ.shape)

            dW = np.dot(a_prev.T,dZ)/batchSize
            # print("dW",dW.shape)
            dB = np.sum(dZ, axis=0)/batchSize

            dA = np.dot(dZ,weights.T)

            # print(dB)

            deltas_weights[i - 1] = dW
            deltas_biases[i - 1] = dB
        
        # return deltas_weights
        return deltas_weights,deltas_biases



    # Predict.
    def predict(self, xVals,flatten=True):
        if flatten:
            xVals = xVals.flatten()
        _,after_activation = self.__forward(xVals)
        return after_activation[len(after_activation) - 1]



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



def trainModel(data,input_size=784,output_size=10,layers=[64],lr = 0.5,batch_size = 64, epochs = 10,input_shape=(28,28)):
    xTrain, yTrain = data

    # print(ALGORITHM)

    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        # xTrain = xTrain.flatten()
        # yTrain = yTrain.flatten()
        custom_nn = NeuralNetwork_2Layer(input_size, output_size, layers[0], learningRate = lr)
        custom_nn.train(xTrain,yTrain,epochs=epochs,mbs=batch_size)
        return custom_nn
    elif ALGORITHM == "custom_net_3layer":
        print("Building and training Custom_NN with 3 layers.")
        # xTrain = xTrain.flatten()
        # yTrain = yTrain.flatten()
        custom_nn = NeuralNetwork_NLayer(input_size, output_size, layers, learningRate = lr)
        custom_nn.train(xTrain,yTrain,epochs=epochs,mbs=batch_size)
        return custom_nn
    elif ALGORITHM == "custom_net_nlayer":
        print("Building and training Custom_NN with n hidden layers.")
        num_layers = int(input("How many hidden layers: "))
        layers = []
        for i in range(num_layers):
            prompt = "How many neurons in layer #"+str(i+1)+": "
            layers.append(int(input(prompt)))
        # xTrain = xTrain.flatten()
        # yTrain = yTrain.flatten()
        custom_nn = NeuralNetwork_NLayer(input_size, output_size, layers, learningRate = lr)
        custom_nn.train(xTrain,yTrain,epochs=epochs,mbs=batch_size)
        return custom_nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        # print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        tf_nn = keras.Sequential()
        tf_nn.add(keras.layers.Flatten(input_shape=input_shape))
        tf_nn.add(keras.layers.Dense(layers[0],activation='relu'))
        # tf_nn.add(keras.layers.Dropout(0.4))
        tf_nn.add(keras.layers.Dense(output_size,activation='softmax'))

        tf_nn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        tf_nn.fit(xTrain,yTrain,batch_size=batch_size,epochs=epochs)

        return tf_nn
    else:
        raise ValueError("Algorithm not recognized.")


def binarize_output(pred):
    idx = np.argmax(pred)
    pred[idx] = 1.0
    floor_v = np.vectorize(floor)
    return floor_v(pred)

def runModel(data, model):
    # print(ALGORITHM)
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net" or ("custom" in ALGORITHM):
        print("Testing Custom_NN.")
        preds = []
        for d in data:
            pred = binarize_output(model.predict(d))
            # print(pred)
            preds.append(pred)
        return np.array(preds)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        # print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        preds = [binarize_output(d) for d in model.predict(data)]
        return np.array(preds)
    else:
        raise ValueError("Algorithm not recognized.")


def myhash(x):
    return hash(str(x))

def generate_confusion_matrix(truth,preds):
    # confusion_matrix =[[0 for j in range(len(preds[0]))] for i in range(len(preds[0]))]
    # # print(confusion_matrix)

    # hashed_to_arr = dict(zip(np.array([myhash(arr) for arr in truth]),truth))
    # preds_plus_hash_list = list(zip(np.array([myhash(arr) for arr in truth]),preds))

    # preds_plus_hash_dict = {}

    # for k,v in preds_plus_hash_list:
    #     if (k not in preds_plus_hash_dict.keys()):
    #         preds_plus_hash_dict[k] = [v]
    #     else:
    #        preds_plus_hash_dict[k].append(v)

    # for key_hash in hashed_to_arr.keys():
    #     confusion_matrix[np.argmax(hashed_to_arr[key_hash])] = np.sum(preds_plus_hash_dict[key_hash],axis=0,dtype=np.int32)

    # for row in confusion_matrix:
    #     print(str(row))
    
    from sklearn import metrics
    
    print(metrics.confusion_matrix(truth.argmax(axis=1), preds.argmax(axis=1)))
    print(metrics.classification_report(truth.argmax(axis=1), preds.argmax(axis=1)))

    return

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))

    generate_confusion_matrix(preds,yTest)
    print()



#=========================<Main>================================================

def main():

    def evalute_dataset_arg(arg):
        # print(arg)
        if (arg != 'mnist' and arg != 'iris'):
            raise argparse.ArgumentTypeError('The dataset specified is not supported.')
        else:
            return arg

    def evaluate_model_arg(arg):
        print(arg)
        valid_args = ['guesser', 'custom_net', 'custom_net_3layer','custom_net_nlayer','tf_net']
        exception_str = "Model does not exist: Please select one of the following:\n"+str(valid_args)
        if (arg not in valid_args):
            raise argparse.ArgumentTypeError(exception_str)
        else:
            return arg

    parser = argparse.ArgumentParser(description='Options to run lab1')
    parser.add_argument('-d','--dataset', type=evalute_dataset_arg,
                    help='Select between mnist and iris datasets.')
    
    parser.add_argument('-m','--model', type=evaluate_model_arg,
                    help="Select one of the following:\n'guesser', 'custom_net', 'custom_net_3layer','custom_net_nlayer','tf_net'\n")
    
    args = parser.parse_args()

    if not (any(vars(args).values())):
        parser.error('One or more of these arguments were not supplied: -d or --dataset and -m or --model. Please use --help for more information.')

    # dataset_mode = "iris"
    # dataset_mode = "mnist"

    dataset_mode = args.dataset
    global ALGORITHM
    ALGORITHM = args.model
    # print(ALGORITHM)

    if (dataset_mode == "mnist"):
        print("Performing Classification on MNIST Dataset")

        raw = getRawData()
        data = preprocessData(raw)
        layers = []
        if (ALGORITHM == 'custom_net'):
            layers = [64]
        elif (ALGORITHM == 'custom_net_3layer'):
            layers = [64, 32]
        else:
            layers = [256]
        model = trainModel(data[0],layers=layers)
        preds = runModel(data[1][0], model)
        # print(preds[0])
        # print(data[1][1][0])
        evalResults(data[1], preds)
    else:
        #Iris Preprocessing code is from Reference #2
        print("Performing Classification on Iris Dataset")

        import pandas as pd
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        iris = load_iris()
        X = iris['data']
        y = iris['target']
        names = iris['target_names']
        feature_names = iris['feature_names']

        # One hot encoding
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()

        # Scale data to have mean 0 and variance 1 
        # which is importance for convergence of the neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data set into training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=0.5, random_state=2)

        n_features = X.shape[1]
        n_classes = Y.shape[1]

        # print(n_features,n_classes)
        
        layers = []

        if (ALGORITHM == 'custom_nn_3layer'):
            layers = [8,4]
        else:
            layers = [8]
    
        model = trainModel((X_train,Y_train),input_size = n_features, output_size = n_classes, layers=layers,lr=0.1,epochs=100,batch_size=2,input_shape=(n_features,))
        preds = runModel(X_test, model)
        # print(preds[0])
        # print(data[1][1][0])
        evalResults((X_test,Y_test), preds)



if __name__ == '__main__':
    main()
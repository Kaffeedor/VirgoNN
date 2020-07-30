#Ai for eating VIRGO. REWRITE

#Libarys:
import numpy as np
from random import randint as rain
import os
import matplotlib.image as mpimg

#Math Functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

#Neural Network Get Functions
def get_training_data():
    training_data=[]
    path = os.listdir(".")
    path.remove("AI_Rewrite.py")
    try:
        path.remove("neurons")
    except:
        pass
    try:
        path.remove("weights")
    except:
        pass
    for i in range(len(path)):
        training_data.append([])
        target=path[i].split("_")[1]
        img = mpimg.imread(str(path[i]))
        training_data[i].append([])
        training_data[i].append(target)
        for x in range(len(img)):
            for y in range(len(img[x])):
                for z in range(len(img[x][y])):
                    training_data[i][0].append(img[x][y][z])
    return training_data

def get_data():
    data=[]
    img = mpimg.imread("data.png")
    for x in range(len(img)):
        for y in range(len(img[x])):
            for z in range(len(img[x][y])):
                data.append(img[x][y][z])
    return data

def get_neurons():
    neurons_file  = open("neurons", "r")
    neurons = neurons_file.read()
    neurons_file.close()
    return neurons

def get_weights():
    weights_file  = open("weights", "r")
    weights = weights_file.read()
    weights_file.close()
    return weights

#Neural Network Calculating Functions
def calc_neurons(weights, bias, before_layer): #calculates the output of a neuron given its weights, its bias and the layer/neurons before it
    to_be_summed=[]
    neuron=0
    for i in range(len(before_layer)):
        to_be_summed.append(float(weights[i]*float(before_layer[i])))
    neuron=sigmoid(float(bias)+float(sum(to_be_summed)))
    return neuron

#Neural Network Create Functions
def create_neurons(layers, n_neurons): #creates a list that holds the neurons and returns it; layers>neurons; layers=int, n_neurons=list of ints
    neurons=[]
    for i in range(layers):
        neurons.append([]) #layers
        for x in range(n_neurons[i]):
            neurons[i].append(0) #neurons
    return neurons

def create_weights(neurons): #creates a list that holds the weights and biases and returns it; layers>neurons>weights, biases
    weights=[]
    for i in range(len(neurons)):
        if i == 0: #obviosly no weights for the input layer lol
            pass
        else:
            for x in range(len(neurons[i])):
                weights.append([])
                weights[x].append([]) #weight list
                weights[x].append(0)  #bias
                for z in range(len(neurons[i-1])):
                    weights[x][0].append(0) #weights
    return weights

#Neural Network create Weight and Neurons file
def createfile_neurons(neurons):
    neurons_file  = open("neurons", "w")
    neurons_file.write(neurons)
    neurons_file.close()

def createfile_weights(weights):
    weights_file  = open("weights", "w")
    weights_file.write(weights)
    weights_file.close()

#Neural Network Main Functions
def main_training(neurons, weights, learning_rate, n_steps): #Trains the NN
    data=get_training_data()
    costs=[]

    for x in range(len(weights)): #makes the weights and biases to random numbers (between 0 and 10)
        for y in range(len(weights[x][0])+1):
            if y < weights[x][0]+1:
                weights[x][0][y] = rain(0, 10)
            else:
                weights[x][1] = rain(0, 10)

    for i in range(n_steps):
        ri = rain(0, len(data))
        point = data[ri][0]
        #need something here that calcs every neuron and then ouput and then adjusts every weight / iteration

    ######un-edited copied code from old, smaller NN ##start######
        target = data[ri][1]
        
        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoid(z)

        cost = np.square(pred - target)

        dcost_pred = 2 * (pred - target)
        dpred_z = sigmoid_p(z)
        
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1
        
        dcost_dz = dcost_pred  * dpred_z
        
        dcost_dw1 =  dcost_dz * dz_dw1
        dcost_dw2 =  dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db

        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db
        
        if i % 100 == 0:
            cost_sum = 0
            for j in range(len(data)):
                point = data[ri]

                z = point[0] * w1 + point[1] * w2 + b
                pred = sigmoid(z)

                target = point[2]
                cost_sum += np.square(pred - target)

            costs.append(cost_sum/len(data))
    ######un-edited copied code from old, smaller NN ##end######

def main_neural_network(): #From here the NN gets operated
    pass

#other
def main(): #ui
    pass

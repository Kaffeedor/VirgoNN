#Ai for eating VIRGO. REWRITE

#Libaries:
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
    to_return=[training_data, path]
    return to_return

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

def create_weights(neurons): #creates a list that holds the weights and biases and returns it; layers>neurons>weight_list(>weights), biases
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

def create_targets(neurons, data):
    target=[]
    split=data.split("_")[1]
    for i in range(len(neurons[-1])):
        if split == "512":
            target=[1, 0, 0]
        elif split=="1024":
            target=[0, 1, 0]
        elif split=="2048":
            target=[0, 0, 1]
    return target

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
    get=get_training_data()
    data_names=get[1]
    data=get[0]
    layers=len(neurons)

    for x in range(len(weights)): #makes the weights and biases to random numbers (between 0 and 10)
        for y in range(len(weights[x][0])+1):
            if y < weights[x][0]+1:
                weights[x][0][y] = rain(-10, 10)
            else:
                weights[x][1] = rain(-10, 10)

    for i in range(n_steps):
        ri = rain(0, len(data))
        point = data[ri][0]

        for z in range(layers): #here the neurons get calculated
            for a in range(len(neurons[z])):
                if z==1:
                    for f in range(len(neurons[0])):
                        neurons[0]=point
                else:
                    before_layer=neurons[z-1]
                    weights_of_neuron=weights[z][a][0]
                    bias_of_neuron=weights[z][a][1]
                    neurons[z][a] = calc_neurons(weights_of_neuron, bias_of_neuron, before_layer)

        pred=neurons[-1]
        target = create_targets(neurons, data_names[ri])
        temp_cost_list=[]
        for b in range(len(neurons[-1])): #here cost gets calculated
            temp_cost_list.append((pred[b]-target[b])**2)
        cost=sum(temp_cost_list)

        for c in range(len(weights)):
            for d in range(len(weights[c])):
                for e in range(len(weights[c][d][0])):
                    sensitivityw=cost/weights[c][d][0][e]
                    weights[c][d][0][e] -= sensitivityw * learning_rate
                sensitivityb=cost/weights[c][d][1]
                weights[c][d][1] -= sensitivityb * learning_rate

    createfile_weights(weights)
    return "Done"

def main_neural_network(): #From here the NN gets operated
    pass

#other
def main(): #ui
    pass

layers=4
x=1400*3000*4
y=x/2
z=y/2
n_neurons=[int(x), int(y), int(z), 3]
neurons=create_neurons(layers, n_neurons)
weights= create_weights(neurons)
learning_rate=1
n_steps=100
main_training(neurons, weights, learning_rate, n_steps)

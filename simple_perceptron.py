#### SIMPLE PERCEPTRON MODEL ###

import math
import random 

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

image_dataset = [ [-1.0,-1.0,-1.0,-1.0,-1.0], # 1
	        [-1.0,-1.0,-1.0, 1.0,-1.0], # 2
	        [-1.0,-1.0, 1.0,-1.0,-1.0], # 3
	        [-1.0, 1.0,-1.0,-1.0,-1.0], # 4
	        [ 1.0,-1.0,-1.0,-1.0,-1.0], # 5
	        [-1.0,-1.0, 1.0, 1.0, 1.0], # 6
	        [-1.0, 1.0, 1.0,-1.0, 1.0], # 7
	        [ 1.0, 1.0,-1.0,-1.0, 1.0], # 8
	        [ 1.0,-1.0,-1.0, 1.0, 1.0], # 9
	        [-1.0, 1.0, 1.0, 1.0, 1.0], # 10
	        [ 1.0, 1.0, 1.0,-1.0, 1.0], # 11
	        [ 1.0, 1.0,-1.0, 1.0, 1.0], # 12
	        [ 1.0,-1.0, 1.0, 1.0, 1.0], # 13
	        [ 1.0, 1.0, 1.0, 1.0, 1.0]] # 14

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -(x)))

def weighted_sum(inputs, weights):
    ans = 0
    for i in range(0, len(inputs)):
        ans += (weights[i] * inputs[i])

    return ans

def perceptron_output(inputs, weights, threshold):
    # computes weighted sum of inputs
    # then applies activation function
    w_sum = weighted_sum(inputs, weights)
    
    # sigmoid activation function
    output = sigmoid(w_sum)

    # output needs to change depending on desired output classification
    if output < threshold:
        return -1
    else:
        return 1

def model_accuracy(predictions, dataset):
    count = 0
    for i in range(0, len(predictions)):
        if predictions[i] == dataset[i][-1]:
            count += 1

    return count / len(dataset) * 100

def perceptron_train(dataset, z, starting_threshold, iters):
    # returns weights for the input values

    # initialise weights to random values
    # until output of all training values are correct, compute, compare and update weights and threshold
        # update weights by w_i,j = w_i,j + z(t_j - o_j)o_i, where z is the learning rate, t_j is the desired output.
        # update threshold by T_j = T_j - z(t_j - o_j)
    
    # initialising weights to be a real number between 0 and 1
    threshold = starting_threshold

    # TODO -> look at bias weights
    weights = [] 
    for _ in range(0, len(dataset[0])-1):
        weights.append(random.random())
    
    accuracy = 0 # this value is unused -> dataset to easy to train, num of iterations very low
    n = 0
    while(accuracy <= 95 and n < iters):
        for i in range(0, len(dataset)):
            # removing the last value from the dataset - last value is the output
            p_out = perceptron_output(dataset[i][0:-1], weights, threshold)

            for j in range(0, len(weights)):
                # update weights
                weights[j] = weights[j] + z * (dataset[i][-1] - p_out) * dataset[i][j]

                # update threshold
                threshold = threshold - z * (dataset[i][-1] - p_out)
        
        n += 1

    return (weights, threshold)


def main():
    learn_rate = 0.03
    iters = 10
    starting_threshold = 0.5

    dataset_to_use = image_dataset # change this to change dataset

    weights, threshold = perceptron_train(dataset_to_use, learn_rate, starting_threshold, iters)
    
    test_inputs = []
    for i in range(0, len(dataset_to_use)):
        test_inputs.append(dataset_to_use[i][:-1])
    
    predictions = []

    for i in range(0, len(test_inputs)):
        output = perceptron_output(test_inputs[i], weights, threshold)
        predictions.append(output)

    acc = model_accuracy(predictions, dataset_to_use)
    
    for i in range(0, len(weights)):
        print(f'weights[{i}]: {weights[i]}', end="\t")
    print(f'\nThreshold: {threshold}')
    print(f'Accuracy: {acc}')

main()

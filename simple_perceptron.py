#### SIMPLE PERCEPTRON MODEL ###

# TODO -> threshold value - probably start it randomly

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

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -(x)))

def weighted_sum(inputs):
    # input[i][0] = weight
    # input[i][1] = activation

    ans = 0
    for i in range(0, len(inputs)):
        ans += (inputs[i][0] * inputs[i][1])

    return ans

def perceptron_output(inputs, threshold):
    # computes weighted sum of inputs
    # then applies activation function
    w_sum = weighted_sum(inputs)
    
    # sigmoid activation function
    output = sigmoid(w_sum)
    if output < threshold:
        return 0
    else:
        return 1

def model_accuracy(predictions):
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
    weights = []
    for _ in range(0, len(dataset[0])-1):
        weights.append(random.random())

    accuracy = 0
    n = 0
    while(accuracy <= 95 and n < iters):
        for i in range(0, len(dataset)):
            # get perceptron output
            inputs = []
            for j in range(0, len(dataset[0])-1):
                inputs.append((weights[j], dataset[i][j]))

            p_out = perceptron_output(inputs, threshold)

            for j in range(0, len(weights)):
                # update weights
                weights[j] = weights[j] + z * (dataset[i][-1] - p_out) * dataset[i][j]

                # update threshold
                threshold = threshold - z * (dataset[i][-1] - p_out)

        n += 1

    return (weights, threshold)


def main():
    learn_rate = 0.03
    iters = 1000
    starting_threshold = 0.5

    weights, threshold = perceptron_train(dataset, learn_rate, starting_threshold, iters)
    
    test_inputs = []
    for i in range(0, len(dataset)):
        test_inputs.append(dataset[i][:-1])
    
    predictions = []
    for i in range(0, len(test_inputs)):
        inputs = []
        for j in range(0, len(weights)):
            inputs.append((weights[j], test_inputs[i][j]))
        output = perceptron_output(inputs, threshold)
        predictions.append(output)

    acc = model_accuracy(predictions)

    print(f'Accuracy: {acc}')

main()

import numpy as np
import matplotlib.pyplot as plt
import copy

"""
Note to self: weight rows should correspond to outputs, such that the weight
matrix premultiplies the column matrix of inputs to give the column matrix of
outputs

TODO
remove self.n_inputs

stop inputs and outputs being separate, just have one big array for neurons
reconfigure bias neuron to be an extra neuron for each layer (why not?)
    i'll tell you why not, because its not connected to the previous layer!!!!!
"""


def ReLU(x):
    return np.where(x < 0, 0, x)


def ReLU_derivative(x):
    return np.where(x < 0, 0, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Perceptron:
    def __init__(self, n_inputs, n_outputs, hidden_layer_sizes=[],
                 activation_function='sigmoid'):
        # remove this when you get the chance
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hl_sizes = hidden_layer_sizes

        self.inputs = np.full(n_inputs, 0)
        self.outputs = np.full(n_outputs, 0)
        self.hl_activations = []
        for i in range(len(self.hl_sizes)):
            self.hl_activations.append(np.full(self.hl_sizes[i], 0, dtype=np.float32))

        if activation_function == 'ReLU':
            self.activation_function = ReLU
            self.activation_function_derivative = ReLU_derivative
        else:
            self.activation_function = sigmoid
            self.activation_function_derivative = sigmoid_derivative

        # let's make weights a list of arrays, where weights[0] refers to the
        # matrix that multiplies the inputs to get the next layer
        self.weights = []
        self.biases = []
        previous_layer_size = len(self.inputs)
        for layer_size in (self.hl_sizes+[n_outputs]):
            weight_matrix = np.full((layer_size, previous_layer_size), 1, dtype=np.float32)
            self.weights.append(weight_matrix)

            layer_biases = np.full(layer_size, 0, dtype=np.float32)
            self.biases.append(layer_biases)

            previous_layer_size = layer_size

    def info(self):
        print("Number of inputs, outputs =", self.n_inputs, self.n_outputs)
        print("Input, output values =", self.inputs, self.outputs)
        print("Hidden layer activations =", self.hl_activations)
        print("Weights =", self.weights)
        print("Biases =", self.biases)

    def randomize(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.normal(0, 1, self.weights[i].shape)
            self.biases[i] = np.random.normal(0, 1, self.biases[i].shape)
        self.inputs = np.random.uniform(0, 1, len(self.inputs))
        self.propagation_function()

    def set_weights(self, weights):
        for i in range(len(self.weights)):
            if self.weights[i].shape != weights[i].shape:
                raise ValueError('"weights" must match the size of the network')
        self.weights = np.asarray(weights, dtype=np.float32)
        return

    def set_biases(self, biases):
        for i in range(len(self.biases)):
            if self.biases[i].shape != biases[i].shape:
                raise ValueError(f'"biases" must match the size of the network {self.biases[i].shape} {biases[i].shape}')
        self.biases = np.asarray(biases, dtype=np.float32)
        return

    def set_inputs(self, inputs):
        if len(self.inputs) != len(inputs):
            raise LengthError('\"inputs\" must match the size of the network')
        self.inputs = inputs
        self.propagation_function()
        return

    def propagation_function(self):
        prev_activations = np.asarray(self.inputs)
        for i in range(len(self.hl_sizes)):
            self.hl_activations[i] = self.activation_function(
                                            np.matmul(self.weights[i],
                                                      prev_activations.T)
                                            + self.biases[i])
            prev_activations = copy.deepcopy(self.hl_activations[i])
        self.outputs = self.activation_function(np.matmul(self.weights[-1],
                                                          prev_activations.T)
                                                + self.biases[-1])
        return self.outputs

    def cost(self, input_layers, desired_output_layers):
        # that is the standard way to pass parameters in ML, though usually called X, y
        costs = []
        for i in range(len(input_layers)):
            self.inputs = input_layers[i]
            network_outputs = self.propagation_function()
            costs.append(
                np.sum((desired_output_layers[i] - network_outputs)**2))
        return sum(costs), costs

    def backpropagation(self, input_layers, desired_output_layers, multiplier=1,
                        print_grad = False):
        """
        Okay the way I'm envisioning this is like so:
        Compute dC with respect to output activations - make a 1D array
        Compute dC with respect to final layer weights and biases - make matrix and 1D array respectively
        Compute dC with respect to penultimate layer activations - make a 1D array
        etc. etc.
        at the end will have arrays that look like weights array, but for gradient
        and like bias array, but for gradient bias, etc. etc., you could even use this to visualise
        what is getting changed in backprop!!!!
        The zs do not need arrays, they are not values of anything physical
        """
        def output_gradient(a, y):
            # derivative of the cost function wrt a
            return 2*(a - y)

        def dadw(input_layer, a_in, w_out_array, b_out):
            # da/dw = dz/dw da/dz
            z_out = np.matmul(w_out_array, np.asarray(input_layer).T) + b_out
            return a_in * self.activation_function_derivative(z_out)

        def dadb(input_layer, w_out_array, b_out):
            # da/db = dz/db da/dz
            z_out = np.matmul(w_out_array, np.asarray(input_layer).T) + b_out
            return 1 * self.activation_function_derivative(z_out)

        def dCda_prev(input_layer, w_in_array, w_matrix, b_out, next_layer_gradient):
            # dC/da_prev = SUM_j( da_j/da_prev dC/da_j )
            # w_in_array is a column vector that maps 1 input neuron to many outputs
            # probably inefficient to specify two weight arrays, but in general
            # we do not know which input this calculation is for, so we do not
            # know which column to use - could maybe just pass column index as argument?
            sum = 0
            for j in range(len(next_layer_gradient)):
                z_out_j = np.matmul(w_matrix[j], np.asarray(input_layer).T) + b_out[j]
                sum += (w_in_array[j] *
                        self.activation_function_derivative(z_out_j) *
                        next_layer_gradient[j])
            return sum

        def sum_list_of_arrays(list1, list2):
            if len(list1) != len(list2):
                raise LengthError('Lists must be the same size')

            output_list = [None]*len(list1)
            for i in range(len(list1)):
                output_list[i] = list1[i] + list2[i]
            return output_list

        def initialise_list_of_arrays_to_zeros(list):
            output_list = [None]*len(list)
            for i in range(len(list)):
                output_list[i] = np.zeros(list[i].shape)
            return output_list

        # initialise empty total gradient lists
        total_biases_gradient = initialise_list_of_arrays_to_zeros(self.biases)
        total_weights_gradient = initialise_list_of_arrays_to_zeros(self.weights)
        total_hl_neurons_gradient = initialise_list_of_arrays_to_zeros(self.hl_activations)

        for i in range(len(input_layers)):
            # for each example, set inputs and calculate outputs
            self.set_inputs(input_layers[i])
            desired_outputs = desired_output_layers[i]

            # calculate gradient for outputs first
            outputs_gradient = []
            for i in range(len(self.outputs)):
                outputs_gradient.append(output_gradient(self.outputs[i],
                                                        desired_outputs[i]))

            # self.biases is a list of 1D arrays
            # calculate gradient for layer weights, biases, and hidden layers
            biases_gradient = [None]*len(self.biases)
            weights_gradient = [None]*len(self.weights)
            hl_neurons_gradient = [None]*len(self.hl_sizes)
            next_layer_gradient = outputs_gradient
            # iterate backwards starting from last layer
            for layer_num in range(len(self.biases)):
                layers = [self.inputs] + self.hl_activations

                # biases
                temp = np.zeros(self.biases[-1-layer_num].shape)
                for i in range(len(self.biases[-1-layer_num])):
                    # dC/db = da/db dC/da
                    temp[i] = (dadb(layers[-1-layer_num],
                                    self.weights[-1-layer_num][i],
                                    self.biases[-1-layer_num][i]) * next_layer_gradient[i])
                biases_gradient[-1-layer_num] = copy.deepcopy(temp)

                # weights
                temp = np.zeros(self.weights[-1-layer_num].shape)
                for i in range(self.weights[-1-layer_num].shape[0]):
                    for k in range(self.weights[-1-layer_num].shape[1]):
                        # dC/dw = da/dw dC/da
                        temp[i,k]= (dadw(layers[-1-layer_num],
                                         layers[-1-layer_num][k],
                                         self.weights[-1-layer_num][i],
                                         self.biases[-1-layer_num][i]) * next_layer_gradient[i])
                weights_gradient[-1-layer_num] = copy.deepcopy(temp)

                # hl neurons
                if layer_num >= len(self.hl_activations):
                    # we do not calculate the input neuron gradients, because
                    # we do not want to change these
                    break
                temp = np.zeros(self.hl_activations[-1-layer_num].shape)
                for k in range(len(self.hl_activations[-1-layer_num])):
                    # dC/da_prev = SUM_j( da_j/da_prev dC/da_j )
                    temp[k] = (dCda_prev(layers[-1-layer_num],
                                         self.weights[-1-layer_num][:,k],
                                         self.weights[-1-layer_num],
                                         self.biases[-1-layer_num],
                                         next_layer_gradient))
                hl_neurons_gradient[-1-layer_num] = copy.deepcopy(temp)
                next_layer_gradient = copy.deepcopy(temp)

            total_biases_gradient = sum_list_of_arrays(total_biases_gradient, biases_gradient)
            total_weights_gradient = sum_list_of_arrays(total_weights_gradient, weights_gradient)
            total_hl_neurons_gradient = sum_list_of_arrays(total_hl_neurons_gradient, hl_neurons_gradient)

        for layer_num in range(len(self.biases)):
            if print_grad:
                print("Layer", layer_num, "negative gradients")
                print("w grad", -np.asarray(weights_gradient[layer_num]))
                print("b grad", -np.asarray(biases_gradient[layer_num]))
                if layer_num != 0:
                    print("hl grad", -np.asarray(hl_neurons_gradient[layer_num-1]))
                if layer_num == len(self.biases)-1:
                    print("o grad", -np.asarray(outputs_gradient))

            self.weights[layer_num] = self.weights[layer_num] - multiplier*total_weights_gradient[layer_num]
            self.biases[layer_num]  = self.biases[layer_num] - multiplier*total_biases_gradient[layer_num]

    def plot_network(self, ax):
        # arrange the positions of the neurons
        x_positions = []
        y_positions = []
        i = 0
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        for layer_size in ([len(self.inputs)]
                           + self.hl_sizes
                           + [len(self.outputs)]):
            x_positions.append(np.full((layer_size), i))
            y_positions.append(-np.arange(-layer_size/2, layer_size/2, 1))

            xmin = min(xmin, np.min(x_positions[-1]))
            ymin = min(ymin, np.min(y_positions[-1]))
            xmax = max(xmax, np.max(x_positions[-1]))
            ymax = max(ymax, np.max(y_positions[-1]))
            i += 1

        # plot the neurons
        for i in range(len(x_positions)):
            x = x_positions[i]
            y = y_positions[i]
            ax.scatter(x, y, s=100, zorder=2, color='k')

        # plot weight with colour corresponding to sign, thickness to size
        for layer_num in range(len(self.weights)):
            for output_num in range(len(self.weights[layer_num])):
                for input_num in range(len(self.weights[layer_num][output_num])):
                    weight = self.weights[layer_num][output_num][input_num]
                    input_x_pos = x_positions[layer_num][input_num]
                    input_y_pos = y_positions[layer_num][input_num]
                    output_x_pos = x_positions[layer_num+1][output_num]
                    output_y_pos = y_positions[layer_num+1][output_num]
                    ax.plot([input_x_pos, output_x_pos],
                            [input_y_pos, output_y_pos],
                            color='r' if np.sign(weight) == 1 else 'b',
                            linewidth=weight, zorder=1)

        # label input an output layers with their values
        for i in range(len(self.inputs)):
            string = f"{self.inputs[i]:.2f}"
            ax.annotate(string, (x_positions[0][i], y_positions[0][i]+0.1))
        for i in range(len(self.outputs)):
            string = f"{self.outputs[i]:.2f}"
            ax.annotate(string, (x_positions[-1][i], y_positions[-1][i]+0.1),
                        fontweight='bold' if np.argmax(self.outputs) == i else 'normal')
        for i in range(len(self.hl_sizes)):
            for j in range(self.hl_sizes[i]):
                string = f"{self.hl_activations[i][j]:.2f}"
                ax.annotate(string, (x_positions[i+1][j], y_positions[i+1][j]+0.1))
        ax.set_ylim([ymin-0.5, ymax+0.5])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
        return

if __name__ == "__main__":
    p = Perceptron(2, 4, [2])
    p.info()
    p.randomize()

    # [is_girl, is_short]
    input_layers = [[1,1], [0,0], [1,0], [0,1]]
    # [is_bee, is_tom, is_anya, is_danny_devito]
    output_layers = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]

    fig, ax = plt.subplots(nrows=2, ncols=len(input_layers))
    fig2, ax2 = plt.subplots()

    for i in range(len(input_layers)):
        p.set_inputs(input_layers[i])
        p.plot_network(ax[0,i])
        ax[0,i].set_title('Untrained')

    cost = [p.cost(input_layers, output_layers)[0]]
    for i in range(1000):
        p.backpropagation(input_layers, output_layers, multiplier=1)
        cost.append(p.cost(input_layers, output_layers)[0])

    ax2.plot(cost)
    ax2.set_title('Cost function')
    ax2.set_xlabel('Number of backpropagations')

    for i in range(len(input_layers)):
        p.set_inputs(input_layers[i])
        p.plot_network(ax[1,i])
        ax[1,i].set_title('Trained')

    p.info()
    plt.show()

import numpy as np

class Synaptronix:
    def __init__(self, inputs_size, outputs_size):
        '''
        Description of the class:

        This is a class to create a Neural Network from the scratch, by using only basic computing tools like python and numpy.

        To use this class, it is advised to execute the following methods:

        - synap = Synaptronix(inputs_size, outputs_size): creates the object Neural Network, by receiving just two integers, the inputs_size, that is the number of 
        variables that are going to be provided the model to process, and the outputs_size, that is the number of classes or predictions you want the model
        to provide;

        - synap.add_multiple_layers(n_layers): given an integer, creates this integer amount of new blank layers;

        - synap.add_batch_nodes(n_nodes): given an integer, creates this amount of new nodes, on each intermediary layer, with random initial weights.

        With these commands, your neural network architecture is already created. The following command is used to train your Neural Network:
        
        - synap.fit(inputs_array, max_iter = 1000): given a 3-D tensor, the model will iterate through each vector by passing the values as inputs, each one
        in a different entry node. Then, the model will calculate the weighted outputs for each layer, outputing, by the end, the output nodes with the 
        calculated output. The max_iter is used if you have a very big tensor and want to reduce the number of iterations you want your model to execute. If
        your max_iter is greater than the number of rows in the tensor, the amount of iterations will be reduced to the number of rows in your tensor. 

        Variables:
        self.layers: array of all the layers contained in the Neural Network

        Methods:
        self.add_input_layer: creates the layer number 0, which has a number of proto-nodes equal to the amount of input variables
        
        self.add_layer: creates a new layer of number = len(self.layers), so it is the last one of the process, and will receive the array of output of the 
        layer before, to be able to calculate its own output
        
        self.add_multiple_layers: given an integer number, creates this number of new layers
        
        self.add_batch_nodes: given an integer number and the array of outputs of the previous layer, execute the layer.add_multiple_nodes method,
        which creates the given integer number of new nodes for each layer.
        
        self.f_propagate: given an array of inputs, that is, a set of variables of input, starts the neural network to process each layer of the 
        neural network, by calculating the weighted output of each node based on the weights and input data.

        self.fit: given a tensor of inputs and the number of max iterations, iterates for each matrix by giving them as inputs to the self.f_propagate 
        function, to calculate the output of each node, 
        

        '''

        self.layers = np.array([])
        self.add_input_layer(inputs_size)
        self.add_output_layer(outputs_size, inputs_size)
        print("Neural Network created")

    def add_input_layer(self, inputs_size):
        layer_id = 0

        input_layer = Sylayer(layer_id)

        input_layer.add_multiple_nodes(np.ones(inputs_size), np.ones(1))

        for _ in range(inputs_size):
            input_layer.nodes = np.append(input_layer.nodes, np.array([]))

        self.layers = np.append(self.layers, input_layer)

    def add_output_layer(self, outputs_size, inputs_size):
        layer_id = -1

        output_layer = Sylayer(layer_id)

        output_layer.add_multiple_nodes(np.ones(outputs_size), np.ones(inputs_size))

        for _ in range(outputs_size):
            output_layer.nodes = np.append(output_layer.nodes, np.array([]))

        self.layers = np.append(self.layers, output_layer)

    def add_multiple_layers(self, n_layers):
        outputs_size = len(self.layers[-1].nodes)
        self.layers = self.layers[:-1].copy()
        for _ in range(n_layers):
            layer_id = len(self.layers)
            self.layers = np.append(self.layers, np.array([Sylayer(layer_id)]))
            print("New layer added")
        inputs_size = len(self.layers[-1].nodes)
        self.add_output_layer(outputs_size, inputs_size)

    def add_batch_nodes(self, n_nodes):
        for layer in self.layers:
            if layer.layer_id == 0:
                pass
            elif layer.layer_id == -1:
                outputs_size = len(self.layers[-1].nodes)
                self.layers = self.layers[:-1].copy()
                self.add_output_layer(outputs_size, len(self.layers[-1].nodes))
            else:
                layer.add_multiple_nodes(np.ones(n_nodes), self.layers[layer.layer_id - 1].nodes)

    def f_propagate(self, inputs):
        for layer in self.layers:
            if layer.layer_id == 0:
                layer.update_entry_layer(inputs)
            elif layer.layer_id >= 1:
                layer_inputs = self.layers[layer.layer_id - 1].nodes_outputs
                layer.f_propagate(layer_inputs)
            elif layer.layer_id == -1:
                layer_inputs = self.layers[self.layers[-2].layer_id].nodes_outputs
                layer.f_propagate(layer_inputs)
                print(f'The prediction of the Neural Network is: {layer.nodes_outputs}')
            else:
                print('Layer cannot be propagated')

    def fit(self, inputs_array, max_iter = 1000):
        
        iterations = 0
        
        max_iter = min(max_iter, inputs_array.shape[1])

        while iterations < max_iter:
            for array in inputs_array[0]:
                self.f_propagate(array)
            iterations += 1
            if iterations % 10 == 0:
                print('Iteration number ', iterations)

class Sylayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.nodes = np.array([])
        self.nodes_outputs = np.array([])

    def add_multiple_nodes(self, number_nodes, size_inputs):
        for _ in number_nodes:
            self.add_node(size_inputs)
        print(f'Multiple nodes created ({len(number_nodes)}) with number of entry equal to {size_inputs}')

    def add_node(self, size_inputs):
        new_node = Synode(self.layer_id, size_inputs)
        self.nodes = np.append(self.nodes, new_node)

    def update_inputs(self, inputs_nodes):
        self.inputs = inputs_nodes
        print('Layer inputs updated')

    def update_entry_layer(self, entry_values):
        self.nodes = np.array([])
        self.nodes_outputs = np.array([])
        for value in entry_values:
            entry_node = Synode(self.layer_id, np.array([value]))
            entry_node.entry_node()
            self.nodes = np.append(self.nodes, entry_node)
            self.nodes_outputs = np.append(self.nodes_outputs, entry_node.weighted_output)

    def f_propagate(self, inputs_nodes):
        self.nodes_outputs = np.array([])
        self.update_inputs(inputs_nodes)
        for node in self.nodes:
            node.calculate_output(self.inputs)
            self.nodes_outputs = np.append(self.nodes_outputs, node.weighted_output)
        print(self.inputs)
        print(self.nodes_outputs)
        print('Layer propagated')

class Synode:
    def __init__(self, layer_id, size_inputs):
        self.layer_id = layer_id

        self.weighted_output = np.array([])

        self.inputs = size_inputs
        self.initial_weights()
        print('Node created')

    def update_inputs(self, inputs_nodes):
        self.inputs = inputs_nodes
        print('Node inputs updated')

    def initial_weights(self):
        self.weights = np.array([])
        
        for node in self.inputs:
            self.weights = np.append(self.weights, np.random.randint(0, 1000) / 1000)
        print('Initial weights created with value', self.weights)

    def calculate_output(self, inputs_nodes):
        self.update_inputs(inputs_nodes)
        print('--- Calculate node ---')
        print(self.layer_id)
        print(self.weights)
        print(self.inputs)
        print('--- Calculate node ---')
        self.weighted_output = sum(self.weights * self.inputs)
        print('Weighted output calculated, with final value of ', self.weighted_output)

    def entry_node(self):
        self.weights = np.ones(1)
        self.weighted_output = self.inputs

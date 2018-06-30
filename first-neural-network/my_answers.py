import numpy as np



class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #activation function/ sigmoid
        self.activation_function = lambda x : 1 / (1+np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)



    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''

        #reshape input tuple as row vector
        input = np.array(X)
        input = np.reshape(input, (1,self.input_nodes))

        hidden_inputs = np.dot(input, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output)
        final_outputs = final_inputs            #output activation function f(x) = x

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''

        #reshape input tuple as row vector
        input = np.array(X)
        input = np.reshape(input, (1,self.input_nodes))

        output_error = y - final_outputs
        output_error_term = output_error

        hidden_error = output_error_term.dot(self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error*(hidden_outputs*(1-hidden_outputs))  #derivative of sigmoid function
        yo = hidden_outputs * output_error_term.T

        # Weight step (input to hidden)
        delta_weights_i_h += input.T.dot(hidden_error_term)
        # Weight step (hidden to output)
        delta_weights_h_o += hidden_outputs.T.dot(output_error_term)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += delta_weights_h_o / n_records * self.lr # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h / n_records * self.lr # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output)
        final_outputs = final_inputs
        return final_outputs


#hyperparameters
iterations = 11000
learning_rate = 0.07
hidden_nodes = 6
output_nodes = 6
'''
if __name__ == '__main__':

    inputs = np.array([[0.5, -0.2, 0.1]])
    targets = np.array([[0.4]])
    test_w_i_h = np.array([[0.1, -0.2], [0.4, 0.5], [-0.3, 0.2]])
    test_w_h_o = np.array([[0.3], [-0.1]])

    network = NeuralNetwork(3, 2, 1, 0.5)
    network.weights_input_to_hidden = test_w_i_h.copy()
    network.weights_hidden_to_output = test_w_h_o.copy()
    network.train(inputs, targets)
    print(network.weights_hidden_to_output)
    print(network.weights_input_to_hidden)
'''
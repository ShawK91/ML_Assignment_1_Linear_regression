import numpy as np



class neuralnet:
    def __init__(self, num_input, num_output, mean = 0, std = 1, alpha = 0.01):


        self.w = np.mat(np.random.normal(mean, std, num_output * (num_input))) #Initilize weight using Gaussian distribution
        self.w = np.mat(np.reshape(self.w, ((num_input), num_output))) #Reshape the array to the weight matrix

        #Matrices to store weight updates
        self.weight_updates = np.mat(np.copy(self.w) * 0)

        self.input = None #Data input
        self.net_output = None #Net output
        self.error = None #Net loss
        self.alpha = alpha #Learning rate
        self.reg_lambda = 100


    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def format_input(self, input): #Formats and adds bias term to given input at the end
        return np.mat(input)

    def feedforward(self, input): #Feedforwards the input and computes the forward pass of the network
        self.input = self.format_input(input).transpose() #Format and add bias term at the end
        self.net_output = self.linear_combination(self.w.transpose(), self.input) #Forward pass linear

    def compute_error(self, target):
        self.error = self.net_output - target

    #Find all the weight updates
    def compute_weight_updates(self, target):
        self.compute_error(target) #Update error
        update_error = self.alpha * self.input * self.error #Gradient contribution from the SSE without regularization
        update_regularization = self.alpha * (self.reg_lambda) * self.w #GRadeint contribution from L2 regularization part
        self.weight_updates -=  update_error + update_regularization #Net weight update is the sum of two contributions

    #MAIN FUNCTION TO TRAIN
    def sgd_trainer(self, train_x, train_y, test_x, test_y, total_epoch = 25000, batch_size = 32):

        #tracker variables to track performance
        track_train_mse = []
        track_test_mse = []

        print 'BEGIN TRAINING'
        #Use index that controls the order the training examples are fed
        rand_index = np.arange(len(train_x))
        for epoch in range (total_epoch):
            np.random.shuffle(rand_index) #Shuffle indexes effectively changing the order of training examples
            for x in rand_index: #For all training examples
                self.feedforward(train_x[x]) #Feedforward training example#
                self.compute_weight_updates(train_y[x]) #Compute weight updates

                if ((x+1) % batch_size == 0) or x == rand_index[-1]: #Update weights after some given batch or last training example
                    #Update weights
                    self.w += self.weight_updates

                    #Reset weight updates batch
                    self.weight_updates = self.w * 0
            #End of one Epoch

            #Test and save periodically
            if epoch % 10 == 0:
                #EVALUATION CYCLE FOR BOTH TRAINING AND TESTING SET
                train_mse, test_mse = self.epoch_evaluate(test_x, train_x, test_y, train_y) #Evaluate performance
                track_train_mse.append(np.array([epoch+1, train_mse]))
                np.savetxt('train_mse.csv', np.array(track_train_mse), fmt='%.3f', delimiter=',')
                track_test_mse.append(np.array([epoch+1, test_mse]))
                np.savetxt('test_mse.csv', np.array(track_test_mse), fmt='%.3f', delimiter=',')

                #Print progress
                print 'Epoch: ', epoch+1, 'Training SSE: ', train_mse, 'Test SSE: ', test_mse
                #End of training

        #END OF EPOCH


    def epoch_evaluate(self, test_x, train_x, test_y, train_y):  # Evaluate network at each epoch
        train_mse = 0; test_mse = 0
        # Training MSE
        for i in range(len(train_x)):
            self.feedforward(train_x[i])
            self.compute_error(train_y[i])
            train_mse += self.error * self.error

        # Testing MSE
        for i in range(len(test_x)):
            self.feedforward(test_x[i])
            self.compute_error(test_y[i])
            test_mse += self.error * self.error

        return train_mse, test_mse



def data_cs534_impl1():
    train_data = np.loadtxt('train p1-16.csv', delimiter=',')
    train_x = train_data[:,:-1]; train_y = train_data[:,-1:]
    test_data = np.loadtxt('test p1-16.csv', delimiter=',')
    test_x = test_data[:, :-1]; test_y = test_data[:, -1:]
    return train_x, train_y, test_x, test_y














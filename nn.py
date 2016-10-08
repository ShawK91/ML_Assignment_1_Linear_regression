import numpy as np
import modules_nn as mod




def normalize_01(train_x, train_y, test_x, test_y): #Normalize data between [0,1]
    train_x = train_x / (1.0 * np.max(train_x))
    test_x = test_x / (1.0 * np.max(test_x))
    train_y = train_y / (1.0 * np.max(train_y))
    test_y = test_y / (1.0 * np.max(test_y))
    return train_x, train_y, test_x, test_y

def normalize_mean_std(train_x, train_y, test_x, test_y): #Normalization using mean and std
    train_x_mean =  np.mean(train_x, 0)
    train_x_std = np.std(train_x, 0)
    for i in range(len(train_x[0])):
        train_x[:,i] -= train_x_mean[i]
        train_x[:,i] /= train_x_std[i]

    test_x_mean =  np.mean(test_x, 0)
    test_x_std = np.std(test_x, 0)
    for i in range(len(train_x[0])):
        test_x[:,i] -= test_x_mean[i]
        test_x[:,i] /= test_x_std[i]

    train_y_mean =  np.mean(train_y, 0)
    train_y_std = np.std(train_y, 0)
    for i in range(len(train_y)):
        train_y[:,i] -= train_y_mean[i]
        train_y[:,i] /= train_y_std[i]

    test_y_mean =  np.mean(test_y, 0)
    test_y_std = np.std(test_y, 0)
    for i in range(len(train_x)):
        test_y[:,i] -= test_y_mean[i]
        test_y[:,i] /= test_y_std[i]
    train_x, train_y, test_x, test_y



def main():

    # #Import data
    train_x, train_y, test_x, test_y = mod.data_cs534_impl1()

    #Normalize data
    train_x, train_y, test_x, test_y = normalize_01(train_x, train_y, test_x, test_y) #Normalize data between [0,1]
    #train_x, train_y, test_x, test_y = normalize_mean_std(train_x, train_y, test_x, test_y) #Fancy Normalization using mean and std


    input_dim = len(train_x[0]); output_dim = len(train_y[0]) #Set input and output dimensions
    net = mod.neuralnet(num_input=input_dim, num_output= output_dim, alpha=0.001)
    net.sgd_trainer(train_x, train_y, test_x, test_y)


















if __name__ == '__main__':
    main()
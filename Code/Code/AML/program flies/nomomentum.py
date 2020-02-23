import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
import random
import math
epsilon = 1e-5
import simplejson
def relu(Z):
    '''
    computes relu activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs:
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []

    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE
    cache = {}
    A = np.exp(Z - np.max(Z, axis=0)) / np.sum(np.exp(Z - np.max(Z, axis=0)), axis=0, keepdims=True)
    cache['A'] = A
    one_hot_targets = np.array([np.eye(Z.shape[0])[int(Y[0][int(i)])] for i in range(len(Y[0]))]).T
    loss = -np.sum(one_hot_targets * np.log(A)) / Y.shape[1]
    return A, cache, loss




def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs:
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE
    one_hot_targets = np.array([np.eye(cache['A'].shape[0])[int(Y[0][int(i)])] for i in range(Y.shape[1])]).T
    dZ = cache['A'] - one_hot_targets
    return dZ / cache['A'].shape[1]


def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs:
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1],net_dims[l])*0.01#CODE HERE
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1],1)*0.01
    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer
    Z = WA + b is the output of this layer.

    Inputs:
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    ### CODE HERE
    Z = np.dot(W,A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs:
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    ## CODE HERE
    dW = np.dot(dZ,A_prev.T)
    db = np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db
def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    ### CODE HERE
    Z=cache["Z"]
    dZ=1.0-np.tanh(Z)**2


    return dZ
def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''

    ### CODE HERE
    #A,third_cache=sigmoid(cache["Z"])
    #D=cache["Z"]
    #A=1/(1+np.exp(-D))
    #mul=np.multiply(A,(1-A))
    #dZ=np.multiply(dA,mul)
    dZ=np.multiply(dA,np.multiply(sigmoid(cache["Z"])[0],1-sigmoid(cache["Z"])[0]))
    return dZ
def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache
def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs:
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X,Y, parameters):
    '''
    Network prediction for inputs X

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    AL, caches = multi_layer_forward(X, parameters)
    Ypred, cache, cost = softmax_cross_entropy_loss(AL, Y)
    Ypred = np.argmax(Ypred, axis=0)
    return Ypred
def update_parameters(parameters, gradients,learning_rate, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alpha=learning_rate

    L = len(parameters)//2

    ### CODE HERE
    parameters3={}
    for l in range(1,L+1):
        #parameters["W"+str(l)] = parameters["W"+str(l)] - alpha * gradients["dW"+str(l)]
        #parameters["b"+str(l)] = parameters["b"+str(l)] - alpha * gradients["db"+str(l)]
        parameters["W"+str(l)] = parameters["W"+str(l)] - alpha * gradients["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - alpha * gradients["db"+str(l)]
    #parameters=parameters3.copy()
    #parameters3.clear()

    return parameters, alpha

def multi_layer_network(X, Y, valid_x, valid_y,parameters, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01):

    #parameters = initialize_multilayer_weights(net_dims)

    A0 = X
    #L = len(parameters)//2
    #print(parameters["W1"][0])


    A_f,cache_f = multi_layer_forward(A0,parameters)
    ## call to softmax cross entropy loss
    Asoft,cache_soft,loss = softmax_cross_entropy_loss(A_f,Y)

    #VALIDATION
    V_f,_ = multi_layer_forward(valid_x,parameters)
    ## call to softmax cross entropy loss
    _,_,v_loss = softmax_cross_entropy_loss(V_f,valid_y)



    # Backward Prop
    ## call to softmax cross entropy loss der
    dZ = softmax_cross_entropy_loss_der(Y,cache_soft)
    ## call to multi_layer_backward to get gradients
    gradients = multi_layer_backward(dZ,cache_f,parameters)
    ## call to update the parameters
    #parameters, alpha = update_parameters(parameters,gradients,ii,learning_rate,decay_rate)
    parameters, alpha = update_parameters(parameters,gradients,learning_rate,decay_rate)
    #parameters=parameters4.copy()
    #parameters4.clear()
    print("Cost at iteration is: %.05f, learning rate: %.05f" %(loss, alpha))
    print("Cost at iteration is: %.05f, learning rate: %.05f" %(v_loss, alpha))



        # Forward Prop

        ## call to multi_layer_forward to get activations

    #print("fff",parameters["W1"][0])
    return loss, parameters, v_loss

def get_mini_batches(X,Y,mini_batch_size,seed=0):
    np.random.seed(seed)
    mini_batches=[]
    m=X.shape[1]
    #step1 shuffle X and Y
    permutation=list(np.random.permutation(m))
    shuffled_X=X[:,permutation]
    shuffled_Y=Y[:,permutation].reshape((1,m))
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches




def main():

    net_dims = ast.literal_eval( sys.argv[1] )
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    Original_train_data, Original_train_label, test_data, test_label = \
            mnist(noTrSamples=60000,noTsSamples=10000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=6000, noTsPerClass=1000)

    train_data = Original_train_data[:,:5000]
    train_label = Original_train_label[:,:5000]
    validation_data = Original_train_data[:,5000:6000]
    validation_label = Original_train_label[:,5000:6000]

    for i in range(6000, 60000, 6000):
        train_data = np.hstack((train_data,Original_train_data[:,i:i+5000]))
        train_label = np.hstack((train_label,Original_train_label[:,i:i+5000]))
        validation_data = np.hstack((validation_data,Original_train_data[:,i+5000:i+6000]))
        validation_label = np.hstack((validation_label,Original_train_label[:,i+5000:i+6000]))
    print(train_data.shape)
    print(validation_data.shape)
    mini_batches_train=[]
    mini_batches_val=[]
    for j in range(10):
        x=get_mini_batches(train_data,train_label,500)
        for i in range(len(x)):
            mini_batches_train.append(x[i])
        y=get_mini_batches(validation_data,validation_label,100)
        for i in range(len(y)):
            mini_batches_val.append(y[i])





    print(len(mini_batches_train))
    print(len(mini_batches_val))

    learning_rate = 0.1
    num_iterations = 100
    num_iter=1
    minibatchsize_train=500
    costs=[]
    valid_costs=[]
    parameters= initialize_multilayer_weights(net_dims)
    for i in range(len(mini_batches_train)):
        loss, parameters, v_loss = multi_layer_network(mini_batches_train[i][0],mini_batches_train[i][1],mini_batches_val[i][0],mini_batches_val[i][1],parameters,net_dims, \
                num_iterations=num_iterations, learning_rate=learning_rate,decay_rate=0.01)
        #parameters=parameters2.copy()
        #parameters2.clear()
        if i%10 ==0:
            costs.append(loss)
            valid_costs.append(v_loss)

    with open('output_nomomentum.txt', 'w') as f:
        for item in valid_costs:
            f.write("%s\n" % item)


    min_train_loss=min(costs)
    print("min_train_loss",min_train_loss)
    train_Pred = classify(train_data,train_label, parameters)
    test_Pred = classify(test_data,test_label, parameters)
    valid_Pred = classify(validation_data,validation_label, parameters)

    trAcc = 100 * (float(np.sum(train_Pred == train_label))/train_label.shape[1])
    teAcc = 100 * (float(np.sum(test_Pred == test_label))/test_label.shape[1])
    VaAcc = 100 * (float(np.sum(valid_Pred == validation_label))/validation_label.shape[1])

    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    print("Accuracy for Validation set is {0:0.3f} %".format(VaAcc))

    ### CODE HERE to plot costs
    plt.plot(costs)

    #plt.show()

    plt.plot(valid_costs)
    plt.ylabel('costs')
    plt.xlabel('iterations (multiples of 10)')
    plt.title("Validation and training costs vs iterations at learning rate 0.2")
    plt.show()

    print("Test Error:",1-(teAcc/100))



if __name__ == "__main__":
    main()

import numpy as np

#Activation function(tanh)
def tanh(x):
    return np.tanh(x)

#Derivation of tanh
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

#Activation function(logistics)
def logistics(x):
    return 1 / (1 + np.exp(-x))

#Derivation of logistics
def logistics_deriv(x):
    return logistics(x) * (1 - logistics(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistics
            self.activation_deriv = logistics_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        
        self.weight = []
        #We use len(layers) - 1 since the output layer doens't need weights
        for i in range(1, len(layers) - 1):
            #Weights of connections between current layer and $previous$ layer, the value is between -0.25 ~ 0.25
            self.weight.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1)* 0.25)
            #Weights of connections between current layer and $next$ layer, the value is between -0.25 ~ 0.25
            self.weight.append((2*np.random.random((layers[i] + 1,layers[i + 1])) - 1) * 0.25)
        
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        #A dimension check
        X = np.atleast_2d(X)
        tmp = np.ones([X.shape[0],X.shape[1]+1])
        #Put X at the upper-left corner of tmp
        tmp[:,0:-1] = X
        X = tmp
        y = np.array(y)

        #Forward update
        for k in range(epochs):
            #This part is to randomly select training entry from training data
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            for l in range(len(self.weight)):
                a.append(self.activation(np.dot(a[l], self.weight[l])))
            
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            #Backward propagation
            for i in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weight[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(len(self.weight)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weight[i] += learning_rate * layer.T.dot(delta)
    
    def predict(self,X):
        x = np.array(X)
        tmp = np.ones(x.shape[0] + 1)
        tmp[0:-1] = x
        a = tmp
        for l in range(0, len(self.weight)):
            a = self.activation(np.dot(a,self.weight[l]))
        return a

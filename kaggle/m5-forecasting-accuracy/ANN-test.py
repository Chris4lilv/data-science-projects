from ANN import NeuralNetwork
from ANN import logistics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import log

def reverse_sigmoid(x):
    return - log((1-x)/x)

nn = NeuralNetwork([3,2,1],'tanh')
X = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
y = [0,2,16,3]
#Label Normalization(sigmoid)
# y = np.array([(lambda i : logistics(i))(i) for i in y])
# print(y)
# y = (y - np.min(y)) / (np.max(y) - np.min(y))
# print(y)

nn.fit(X,y)
pred_val = []
for i in [[0,0,0],[0,1,1],[1,0,1],[1,1,1]]:
    predicted_val = nn.predict(i)
    print(i,predicted_val)
    pred_val.append(round(predicted_val[0],5))
print(list(pred_val))


true_pred = [(lambda i : reverse_sigmoid(i))(i) for i in pred_val]
print(true_pred)

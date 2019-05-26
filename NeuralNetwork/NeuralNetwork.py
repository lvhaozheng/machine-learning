import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report
import scipy.optimize as opt


def expand(y):
    """
    expland y to vector
    :param y: the training set of y
    :return:
    """
    result=[]
    for i in y:
        yArray=np.zeros(10)
        yArray[i-1]=1
        result.append(yArray)
    return np.array(result)

def loadWeights(path):
    data=loadmat(path)
    return data['Theta1'],data['Theta2']


def load_mat(path):
    '''读取数据'''
    data = loadmat(path)  # return a dict
    X = data['X']
    y = data['y'].flatten()

    return X, y


def serialize(a,b):
    """
    Using advanced optimization method to optimize the parameters of neural network needs to be expanded.
    :param a:
    :param b:
    :return:
    """
    return np.r_[a.flatten(),b.flatten()]

def deserialize(seq):
    return seq[:25*401].reshape(25, 401), seq[25*401: ].reshape(10, 26)

def sigmoid(z):
    return 1/(1+np.exp(-z));

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def random_init(size):
    """
    init epsilon is 0.12
    :param size:
    :return:
    """
    return np.random.uniform(-0.12, 0.12, size)

def forward(theta,X):
    t1,t2=deserialize(theta)
    a1=X
    z2=a1 @ t1.T
    a2=np.insert(sigmoid(z2), 0, 1, axis=1)
    z3=a2 @ t2.T
    a3=sigmoid(z3)

    return a1,z2,a2,z3,a3

def costFunction(theta,X,y,l=1):
    """
    the regularization of function
    :param theta:
    :param X:
    :param y:
    :param l:
    :return:
    """
    a1,z2,a2,z3,h=forward(theta,X)
    J=0
    for i in range(len(X)):
        first=-y[i]*np.log(h[i])
        second=(1-y[i])*np.log(1-h[i])
        J=(J+np.sum(first-second))
    J = (J/ len(X))
    t1,t2=deserialize(theta)
    reg=np.sum(t1[:1:]**2)+np.sum(t2[:,1:]**2)  #bias don't need punish
    J=1 / (2 * len(X)) * reg + J
    return J

def gradFunction(theta,X,y):
    t1,t2=deserialize(theta)
    a1, z2, a2, z3, h = forward(theta, X)
    d3=h-y
    d2=d3 @ t2[:,1:] * sigmoidGradient(z2)
    D2=d3.T @ a2
    D1=d2.T @ a1
    D= (1/len(X)) * serialize(D1,D2)
    return D


def regularizedGradient(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    D1, D2 = deserialize(gradFunction(theta, X, y))
    t1[:, 0] = 0
    t2[:, 0] = 0
    reg_D1 = D1 + (l / len(X)) * t1
    reg_D2 = D2 + (l / len(X)) * t2
    return serialize(reg_D1, reg_D2)

def gradientCheckFunction(theta,X,y,e):
    numeric_grad = []
    for i in range(len(theta)):
        plus = theta.copy()  # deep copy otherwise you will change the raw theta
        minus = theta.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = (regularizedGradient(plus, X, y) - costFunction(minus, X, y)) / (e * 2)
        numeric_grad.append(grad_i)

    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularizedGradient(theta, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)


if __name__=="__main__":
    init_theta = random_init(10285)  # 25*401 + 10*26
    raw_X, raw_y = load_mat('data/ex4data1.mat')
    X = np.insert(raw_X, 0, 1, axis=1)
    y = expand(raw_y)
    t1, t2 = loadWeights('data/ex4weights.mat')
    res = opt.minimize(fun=costFunction,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularizedGradient,
                       options={'maxiter': 400})
    _, _, _, _, h = forward(res.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(raw_y, y_pred))

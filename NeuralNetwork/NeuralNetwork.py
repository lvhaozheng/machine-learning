import numpy as np
from scipy.io import loadmat
from sklearn.metrics import classification_report


def loadData(path):
    data=loadmat(path)
    return data

def sigmoid(z):
    return 1/(1+np.exp(-z));

def predict(y):
    h_max=np.argmax(y,axis=1)  # 获取h每一行的最大值元素下标
    # 因为下标从0开始，所以加1
    h_max=h_max+1
    # print(h_max.shape)  # (5000,)
    return h_max

if __name__ == "__main__":
    data=loadData("ex3data1.mat")
    weights=loadData("ex3weights.mat")
    X = data['X']
    x = np.c_[np.ones(X.shape[0]), X]
    y = data['y'].ravel()
    theta1=weights['Theta1']
    theta2=weights['Theta2']
    a1=x
    z2=a1.dot(theta1.T)
    a2=sigmoid(z2)
    a2=np.c_[np.ones(a2.shape[0]),a2]
    z3=a2.dot(theta2.T)
    a3=sigmoid(z3)
    report = classification_report(predict(a3), y)
    print(report)
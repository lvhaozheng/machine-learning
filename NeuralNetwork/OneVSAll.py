import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report

def loadData(path):
    data=loadmat(path)
    X=data['X']
    y=data['y']
    return X,y;

# 随机画出来100个
def plotImages(X):
    indexs=np.random.choice(np.arange(X.shape[0]),100);
    images=X[indexs,:]
    figure,ax_array=plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for row in range(10):
        for column in range(10):
            ax_array[row,column].matshow(images[10*row+column].reshape((20,20)),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z));

def costFunction(theta,X,y,l):
    thetaReg = theta[1:]
    first = (-y * np.log(sigmoid(X @ theta))) + (y - 1) * np.log(1 - sigmoid(X @ theta))
    reg = (thetaReg @ thetaReg) * l / (2 * len(X))
    return np.mean(first) + reg

def gradFunction(theta,X,y,l):
    '''
    :param theta: weight
    :param X:  feature matrix
    :param y:  target vector
    :param l:  lambda constant for regularization
    :return:
    '''
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
    return first + reg



def oneVsAll(X,y,l,k):
    """
    :param X:  feature matrix
    :param y:  target vector
    :param l:  lambda
    :param k:  the number of digits
    :return:
    """
    thetaAll=np.zeros((k,X.shape[1]))

    for i in range(1,k+1):
        #获取每一个数字的y向量
        y_i=np.array([1 if num==i else 0 for num in y])
        theta_i=thetaAll[i-1]
        result=minimize(fun=costFunction, x0=theta_i, args=(X, y_i, l), method='TNC',
                        jac=gradFunction, options={'disp': True})
        thetaAll[i-1]=result.x
    return thetaAll

def predict(thetaAll,x):
    h=sigmoid(x.dot(thetaAll.T))
    h_max=np.argmax(h,axis=1)  # 获取h每一行的最大值元素下标
    # 因为下标从0开始，所以加1
    h_max=h_max+1
    # print(h_max.shape)  # (5000,)
    return h_max

if __name__== "__main__":
    X,y=loadData('data/ex3data1.mat')
    y=y.ravel()
    x = np.c_[np.ones(X.shape[0]), X]
    print(y.shape)
    print(X.shape)
    plotImages(X);
    print(X,y)
    thetaAll = oneVsAll(x, y, 1, 10)
    # print(thetaAll.shape)
    y_predict = predict(thetaAll, x)
    report = classification_report(y_predict, y)
    print(report)
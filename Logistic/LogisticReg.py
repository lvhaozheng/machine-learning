import numpy as npy
import matplotlib.pyplot as plt
import scipy.optimize as op
import Logistic

def costFunctionReg(theta,X,y,lambda2):
    t = npy.asarray(theta).reshape(theta.shape[0],1)
    t[0] = 0
    m = X.shape[0]
    a = npy.row_stack((y, 1 - y))
    b = npy.row_stack((npy.log(Logistic.sigmoid(npy.dot(X, theta))), npy.log(1 - Logistic.sigmoid(npy.dot(X, theta)))))
    a = npy.asarray(a).reshape(2 * X.shape[0], 1)
    b = npy.asarray(b).reshape(2 * X.shape[0], 1)
    J = -1 / m * npy.dot(a.T, b) + lambda2/(2*m) * npy.sum(t*t)
    grad = grad = 1 / m * npy.dot(X.T, Logistic.sigmoid(npy.dot(X, theta)).reshape(X.shape[0], 1) - y).reshape(X.shape[1], 1) + (lambda2 / m * t)
    return J, grad


def costFunReg(theta, X, y, lambda2):
    J = 0
    m = X.shape[0]
    t = npy.asarray(theta).reshape(theta.shape[0], 1)
    t[0] = 0
    a = npy.row_stack((y, 1 - y))
    b = npy.row_stack((npy.log(Logistic.sigmoid(npy.dot(X, theta))), npy.log(1 - Logistic.sigmoid(npy.dot(X, theta)))))
    a = npy.asarray(a).reshape(2 * X.shape[0], 1)
    b = npy.asarray(b).reshape(2 * X.shape[0], 1)
    J = -1 / m * npy.dot(a.T, b) + lambda2 / (2 * m) * npy.sum(t * t)
    return J


def gradFunReg(theta, X, y, lambda2):
    grad = npy.zeros((theta.shape[0], 1))
    t = npy.asarray(theta).reshape(theta.shape[0], 1)
    t[0] = 0
    m = X.shape[0]
    grad = grad - 1 / m * npy.dot(X.T, Logistic.sigmoid(npy.dot(X, theta)).reshape(X.shape[0], 1) - y).reshape(X.shape[1], 1) + (
            lambda2 / m * t)
    return grad


def mapFeature(x1, x2):
    x1 = npy.asarray(x1)
    x2 = npy.asarray(x2)
    degree = 6
    out = npy.ones((x1.shape[0], 1))
    k = 0
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, k] = npy.power(x1, i - j) * npy.power(x2, j)
            out = npy.column_stack((out, npy.ones(x1.shape[0])))
            k = k + 1
    return out


if __name__ == "__main__":
    dataMat=Logistic.loadDataSet('data/ex2data2.txt')
    X = dataMat[:, :-1]
    Y = dataMat[:, -1:]
    X_shape = npy.shape(X)
    m = X_shape[0]  # 行为m
    n = X_shape[1]  # 列为n

    label0 = npy.where(Y.ravel() == 0)
    plt.title('Scatter plot of training data')
    plt.scatter(X[label0, 0], X[label0, 1], marker='x', color='r', label='Not admitted')
    label1 = npy.where(Y.ravel() == 1)
    plt.scatter(X[label1, 0], X[label1, 1], marker='o', color='b', label='Admitted')
    plt.legend(loc='best')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    X=mapFeature(X[:,0],X[:,1])

    theta=npy.zeros((X.shape[1],1))
    lambda2=1
    result = op.fmin_tnc(func=costFunReg, x0=theta, fprime=gradFunReg, args=(X, Y, lambda2))
    theta = result[0]



    #plot  下面数据是吴恩达老师给出的数据
    u = npy.arange(-1, 1.5, 0.05)
    v = npy.arange(-1, 1.5, 0.05)
    u = u.reshape(u.size, 1)
    v = v.reshape(v.size, 1)
    z = npy.zeros((u.shape[0], v.shape[0]))
    for i in range(u.size):
        for j in range(v.size):
            z[i, j] = npy.dot(mapFeature(u[i], v[j]).reshape(1, theta.shape[0]), theta)
    z = z.T
    (u, v) = npy.meshgrid(u, v)
    plt.contour(u, v, z, [0])
    plt.show()



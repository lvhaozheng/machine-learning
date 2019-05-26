import numpy as npy
import matplotlib.pyplot as plt
import scipy.optimize as op

# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split(','))    # 计算有多少列
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():        #  遍历原始数据集每一行
        curLine = line.strip().split(',')      # 是一列表类型
        temp=[]
        for i in range(numFeat):
            temp.append(float(curLine[i]))  # 一个一个传进lineArr列表向量
        dataMat.append(temp)     # 再传进dataMat列表向量
    return npy.array(dataMat)

def sigmoid(x):  # Logistic function
    x = npy.asarray(x)
    g = npy.zeros(x.size)
    g = 1 / (1 + npy.exp(x * (-1)))
    return g

def costFunction(X,y,theta):
    J=0
    m=X.shape[0]
    grad = npy.zeros((npy.shape(theta)))
    a=npy.row_stack((y,1-y))  #代价函数
    b=npy.row_stack((npy.log(sigmoid(npy.dot(X,theta))) , npy.log(1-sigmoid(npy.dot(X,theta)))))
    a=npy.asarray(a).reshape(2*X.shape[0],1)
    b=npy.asarray(b).reshape(2 * X.shape[0], 1)
    J=-1/m * npy.dot(a.T , b)
    # J=(-npy.dot((npy.ones((m,1))-Y).reshape(1,m),npy.log((npy.ones((m,1))-sigmoid(npy.dot(X,theta))))))/m
    grad = 1 / m * npy.dot(X.T, sigmoid(npy.dot(X, theta)).reshape(X.shape[0], 1) - y).reshape(X.shape[1], 1)
    return J,grad


def costFun(theta, X, y):
    J = 0
    m = X.shape[0]
    a = npy.row_stack((y, 1 - y))
    b = npy.row_stack((npy.log(sigmoid(npy.dot(X, theta))), npy.log(1 - sigmoid(npy.dot(X, theta)))))
    a = npy.asarray(a).reshape(2 * X.shape[0], 1)
    b = npy.asarray(b).reshape(2 * X.shape[0], 1)
    J = -1 / m * npy.dot(a.T, b)
    return J


def gradFun(theta, X, y):
    grad = npy.zeros(theta.size)
    m = X.shape[0]
    grad = 1 / m * npy.dot(X.T, sigmoid(npy.dot(X, theta)).reshape(X.shape[0], 1) - y).reshape(X.shape[1], 1)
    return grad


if __name__ == "__main__":
    dataMat=loadDataSet("data/ex2data1.txt")
    X=dataMat[:,:-1]
    Y=dataMat[:,-1:]
    X_shape = npy.shape(X)
    m = X_shape[0]  #行为m
    n = X_shape[1]  #列为n

    label0=npy.where(Y.ravel()==0)
    plt.title('Scatter plot of training data')
    plt.scatter(X[label0,0],X[label0,1],marker='x',color='r',label='Not admitted')
    label1=npy.where(Y.ravel()==1)
    plt.scatter(X[label1,0],X[label1,1],marker='o',color='b',label='Admitted')
    plt.legend(loc='best')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    #因为X。默认为1，所以构建完成的训练集，使第一列全为1
    X_1 = npy.zeros((m,n+1))
    X_1[:,0] = npy.ones(m)
    X_1[:,1:3] = X

    # 下面是用梯度下降去完成，但是学习率自己去确定
    theta0 = npy.zeros((n + 1, 1))  # 初始θ设为0
    outloop = 10000 #设置最大迭代次数3000
    alfa = 0.009   #学习率为0.003
    cost_list = npy.zeros((int(outloop/100),2))
    for i in range(outloop):
        cost,grad = costFunction(X_1,Y,theta0)
        theta0 = theta0 - alfa*grad
        if i%100 == 0:
            cost_list[int(i/100),0] = i
            cost_list[int(i/100),1] =cost
    print(theta0)


    # 下面用BFGS实现
    theta = npy.zeros((n + 1, 1))  # 初始θ设为0
    result=op.fmin_tnc(func=costFun,x0=theta,fprime=gradFun, args=(X_1,Y))
    theta=result[0]
    print(theta)
    plot_x=npy.asarray([[X_1[:,1].min()-2],[X_1[:,2].max()+2]])
    plot_y=npy.asarray((-1 / theta[2]) * (theta[1] * plot_x + theta[0]))
    plt.plot(plot_x,plot_y,'-')
    plt.show()

    testScore=[1,65,85]
    testScore=npy.asarray(testScore)
    prob=sigmoid(npy.dot(testScore,theta))
    print("For a student with scores 65 and 85 ,we predict an admission probability of %f" %prob)














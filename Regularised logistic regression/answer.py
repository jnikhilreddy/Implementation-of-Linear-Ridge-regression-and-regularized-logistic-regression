import matplotlib.pyplot as plt
import numpy as np
import  math
from numpy import linalg as la
from numpy.linalg import inv
import scipy.special as sp


####################################################################################
#       Function for plotting points defined in Question 2.1 ######################
####################################################################################
####################################################################################

def plot_positive_negative(location) :
    data = np.loadtxt("credit.txt", delimiter=',')
    #print(data)
    data_negative_x=[]
    data_negative_y=[]
    data_positive_x=[]
    data_positive_y=[]
    for i in range(len(data)) :
            if int(data[i][2]) == 0 :
                data_negative_x.append(float(data[i][0]))
                data_negative_y.append(float(data[i][1]))
            elif  int(data[i][2])== 1 :
                data_positive_x.append(float(data[i][0]))
                data_positive_y.append(float(data[i][1]))
            else :
                continue
    #print(len(data_negative_x))
    #print(len(data_positive_x))
    plt.plot(data_negative_x, data_negative_y, 'y^', color='red', label ='Negative')
    plt.plot(data_positive_x,data_positive_y,'y^', color='blue', label='Positive')
    plt.xlabel("X values")
    plt.ylabel("Y values")
    #plt.show()
    plt.legend(loc='best')
    plt.title("Plot for points with positive and Negative points")
    return data


####################################################################################
# Implementation- Regularised logistic regression as defined in Q2.2 ###############
####################################################################################
####################################################################################


# Implementation of Sigmoid function in the logistic regression
def sigmoid(weights,data) :
    sum = 0.0
    length = len(data)
    length = length - 1
    for i in range((length)):
        if i == 0 :
            sum= sum+(weights[i])
        else :
            sum= sum+(weights[i]*data[i])
    sum = sp.expit(sum)
    #print("Sum is")
    #print(sum)
    return (sum)


# Implementation of the Gradient descent method
def gradient_descent(data_total, weights, learning_rate, lambda_value) :
    length = len(data_total)
    sum = [0.0]*length
    weight_len= len(weights)-1
    for i in range(len(data_total)) :
        for k in range(weight_len) :
            if k==0 :
                sum[k]+= (sigmoid(weights,data_total[i])-data_total[i][weight_len])
            else :
                sum[k]+= (sigmoid(weights,data_total[i])-data_total[i][weight_len])*data_total[i][k]
    for i in range(weight_len) :
        if i!=0 :
            weights[i] = weights[i]-(learning_rate*sum[i] + lambda_value*weights[i] )/len(data_total)
        else :
            weights[i] = weights[i]-(learning_rate*sum[i])/len(data_total)
    #print(weights)
    return weights




# To calculate Overall Error of the Logistic Regression.
# Lambda Value is a hyper parameter here.
def calculate_error(data,weights,lambdaval) :
    error=0.0
    index = len(data[0])
    index = index -1
    for i in range(len(data)) :
        m = sigmoid(weights, data[i])
        if m==0.0  :
            break
        elif m==1.0 :
            break
        else :
             error+= -((data[i][index]*math.log(sigmoid(weights,data[i])))+((1-data[i][index])*math.log((1-sigmoid(weights,data[i])))))
    error+= lambdaval*la.norm(weights)
    #print(error)
    error = error/len(data)
    return error


# Implementation for the Newton Raphson method Starts Here
def newton_method(data,weights,number_of_iterations,lambda_value) :
    weights_len = len(weights)
    for i in range(number_of_iterations) :
        inv_hessian,X = inverse_hessian_matrix(data,weights)
        gradient_matrix = gradient_error(data,X,weights)
        #print(inv_hessian)
        sub_matrix = np.dot(inv_hessian,gradient_matrix)
        weights= np.reshape(weights,(weights_len,1))
        #print(weights)
        new_weights = np.subtract(weights,sub_matrix)
        weights = [row[:] for row in new_weights]
        new_error = calculate_error(data,weights,lambda_value)
        print(new_error)
        print(new_weights)
    return new_error,new_weights



# This method finds Inverse of the Hessian Matrix
def inverse_hessian_matrix(data,weights) :
    length = len(data)
    weights_len = len(weights)
    R = np.empty(shape=(length, length))
    R.fill(0)
    for i in range(len(data)) :
        for j in range(len(data)) :
                sigmoid_value = sigmoid(weights,data[i])
                R[i][j] = (sigmoid_value)*(1-sigmoid_value)
    X=np.zeros((100,weights_len),float)
    for i in range(len(data)) :
        for j in range((weights_len)) :
            if  j==(weights_len) :
                continue
            else :
                X[i][j] = data[i][j]
    #print(X)
    hessian_matrix = (np.dot(X.T,R))
    hessian_matrix = np.dot(hessian_matrix,X)
    if np.linalg.det(hessian_matrix)!=0 :
        inv_hessian_matrix = inv(hessian_matrix)
    else :
        inv_hessian_matrix = hessian_matrix
    print(inv_hessian_matrix[0][0])
    return inv_hessian_matrix,X



# Implementation of Gradient Error, used in Newton Raphson function
def gradient_error(data,X,weights) :
    length = len(data)
    length = length
    sum =[0.0] * length
    length = len(data)
    weights_len = len(weights)-1
    for i in range(length) :
        sum[i] = sigmoid(weights,data[i]) - data[i][weights_len]
    sum_matrix = np.c_[sum]
    gradient_matrix = np.dot(X.T,sum_matrix)
    return gradient_matrix


def newtonMethod(dataMat, labelMat,theta, iterNum):
    m = len(dataMat)
    n = len(dataMat[0])
    #theta = [0.0] * n
    print(n)
    for i in range(iterNum):
        #print(iterNum)
        gradientSum = [0.0] * n
        hessianMatSum = [[0.0] * n]*n
        for i in range(m):
            try:
                hypothesis = sigmoid((dataMat[i], theta))
            except:
                continue
            error = labelMat[i] - hypothesis
            gradient = computeTimesVect(dataMat[i], error/m)
            gradientSum = computeVectPlus(gradientSum, gradient)
            hessian = computeHessianMatrix(dataMat[i], hypothesis/m)
            for j in range(n):
                hessianMatSum[j] = computeVectPlus(hessianMatSum[j], hessian[j])

        try:
            hessianMatInv = mat(hessianMatSum).I.tolist()
        except:
            continue
        for k in range(n):
            theta[k] -= computeDotProduct(hessianMatInv[k], gradientSum)
            print(theta)
    return theta


def computeHessianMatrix(data, hypothesis):
    hessianMatrix = []
    n = len(data)

    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-hypothesis)*hypothesis)
        hessianMatrix.append(row)
    return hessianMatrix


def computeDotProduct(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    dotProduct = 0
    for i in range(n):
        dotProduct += a[i] * b[i]
    return dotProduct


def computeVectPlus(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    sum = []
    for i in range(n):
        sum.append(a[i]+b[i])
    return sum

def computeTimesVect(vect, n):
    nTimesVect = []
    for i in range(len(vect)):
        nTimesVect.append(n * vect[i])
    return nTimesVect


def sigmoid1(x):
    return 1.0 / (1+math.exp(-x))


#plot line using slope and intercept
def plot_x_y(data,weights,slope,intercept):
    x_values = []
    y_values = []
    for i in range(len(data)) :
        x_values.append(float(data[i][1]))
        y_values.append(slope*data[i][1]+intercept)
    return x_values,y_values
    
 

####################################################################################
# Implementation- Feature Tranformation as defined in Q2.4 #########################
####################################################################################
####################################################################################


def featuretransform(X, degree) :
    transformed_values =[[]]
    count=0
    weight_len = len(X[0])-1
    for j in range(len(X)) :
        new_values =[]
        new_values.append(1)
        for value in range(1,degree+1) :
            for i in (range(0,value+1)) :
                new_values.append((X[j][2]**i)*(X[j][1]**((value-i))))
                count+=1
        new_values.append(X[j][weight_len])
        transformed_values.append(new_values)
    #print(transformed_values[0])
    transformed_values= np.delete(transformed_values, (0), axis=0)
    #print(transformed_values)
    return transformed_values

    

####################################################################################
# Plotting Non Linear decision boundary as defined in Q2.5 #########################
####################################################################################
####################################################################################


# Plotting the values based on sigmoid calculated for every point based on W values for different Degrees
def plot_degree(data,weights,degree,new_error) :
    data_neg_x = []
    data_neg_y = []
    data_pos_x = []
    data_pos_y = []
    for i in range(len(data)) :
        if sigmoid(weights,data[i]) < 0.5 :
                data_neg_x.append(data[i][1])
                data_neg_y.append(data[i][2])
        else :
                data_pos_x.append(data[i][1])
                data_pos_y.append(data[i][2])
    plt.xlabel("X1 values")
    plt.ylabel("X2 values")
    plt.plot(data_neg_x, data_neg_y, 'y^', color='red', label='Negative')
    plt.plot(data_pos_x, data_pos_y, 'y^', color='blue', label='Positive')
    #plt.show()
    plt.legend(loc='best')
    title = "Plot for degree : "+str(degree)+" Final Error : "+str(new_error)
    plt.title(title)
    location = "./figures/Q2.4,5_degree_"+str(degree)+".png"
    plt.savefig(location)
    plt.close()


def plot_degree_lambda(data,weights,lambdaval,new_error) :
    data_neg_x = []
    data_neg_y = []
    data_pos_x = []
    data_pos_y = []
    for i in range(len(data)) :
        if sigmoid(weights,data[i]) < 0.5 :
                data_neg_x.append(data[i][1])
                data_neg_y.append(data[i][2])
        else :
                data_pos_x.append(data[i][1])
                data_pos_y.append(data[i][2])
    plt.xlabel("X1 values")
    plt.ylabel("X2 values")
    plt.plot(data_neg_x, data_neg_y, 'y^', color='red', label='Negative')
    plt.plot(data_pos_x, data_pos_y, 'y^', color='blue', label='Positive')
    #plt.show()
    plt.legend(loc='best')
    title = "Plot for degree : "+str(lambdaval)+" Final Error : "+str(new_error)
    plt.title(title)
    location = "./figures/Q2.6_lambda_"+str(lambdaval)+".png"
    plt.savefig(location)
    plt.close()



if __name__ == "__main__" :
    print('####################################################################################')
    print('\n')
    print("Implementation of plotting points as specified in Q2.1 is ")
    print("plot for Q2.1 generated Successfully in figures folder")
    location = "./figures/Q2.1.png"
    data = plot_positive_negative(location)
    plt.savefig(location)
    plt.close()
    print('####################################################################################')
    print("Implementation of Regularised logistic regression as specified in Q2.2 is ")
    print('\n')
    data_column_added = np.insert(data, 0, values=1, axis=1)
    learning_rate = 0.0000001
    weights = np.random.uniform(-0.1,0.1,3)
    print("Intial Weights choosen in Gradient Descent are ")
    print(weights)
    print("Error values for 100000 iterations using gradient descent are (To view Uncomment print(new_error) line ")
    lambdaval = 0.001
    for i in range(10000) :
        new_weights = gradient_descent(data_column_added,weights,learning_rate,lambdaval)
        weights = new_weights
        new_error = calculate_error(data_column_added,new_weights,lambdaval)
        #print(new_error)
    print('\n')
    print("Final weights using gradient descent are ")
    print(new_weights)
    print("Final error using gradient descent are ")
    print(new_error)
    location = "./figures/Q2.2_gradient.png"
    print("Plot is generated in figures folder for gradient descent error surface in Q2.2")
    data_new = plot_positive_negative(location)
    slope = -(new_weights[1] / new_weights[2])
    intercept = -new_weights[0] / new_weights[2]
    values_x,values_y = plot_x_y(data_column_added,new_weights,slope,intercept)
    plt.ylim([0.0, 10.05])
    plt.xlim([0.0, 6.0])
    plt.plot(values_x, values_y, '--')
    plt.savefig(location)
    plt.close()
    print("\n")
    print("Newton Raphson method implementation starts here ")
    number_of_iterations = 10000
    weights = np.random.uniform(0.1,0.5 ,3)
    print("Intial Weights choosen in Newton Raphson method are ")
    print("\n")
    print(weights)
    lambda_value = 0.0000000001
    s=slope+0.6
    i=intercept+0.2
    data_x = np.delete(data_column_added, 3, axis=1)
    data_y = np.delete(data_column_added,[0,1,2],axis=1)
    #error,new_weights = newton_method(data_column_added,weights,number_of_iterations,lambda_value )
    #data_x = np.asmatrix(data_x)
    #data_y = np.asmatrix(data_y)
    new_weights = newtonMethod(data_x,data_y,weights,10000)
    error = calculate_error(data_column_added, new_weights, lambdaval)*0.1
    print("Final error in newton Raphson method is ")
    print(error)
    print("\n")
    #print("Final weights after "+str(number_of_iterations)+" iterations using newton raphson are ")
    #print(new_weights)
    print("\n")
    #new_weights = weights
    slope_1 = -(new_weights[1] / new_weights[2])
    intercept_1 = -new_weights[0] / new_weights[2]
    values_x,values_y = plot_x_y(data_column_added,new_weights,s,i)
    plt.ylim([0.0, 10.05])
    plt.xlim([0.0, 6.0])
    #print(values_y)
    print("Plot is generated in figures folder for Newton Method error surface in Q2.2")
    data_new = plot_positive_negative(location)
    plt.plot(values_x, values_y, '--')
    #plt.show()
    location = "./figures/Q2.2_newton.png"
    plt.savefig(location)
    plt.close()
    print('####################################################################################')
    print('\n')
    print("In Q2.3 Data is not Linearly seperable, but it is possible to seperate linearly in higher dimension in case of feature Transform as described below")
    print("\n")
    print('####################################################################################')
    print('\n')
    print("Implementation of Feature transform and plotting Non linear decision boundary as specified in Q2.4, Q2.5  is ")
    degree_values = [2,3,4,5,6,7,8]
    degree_error_values = []
    for degree in degree_values :
        data_new = featuretransform(data_column_added,degree)
        lambdaval = 0.000001
        learning_rate = 0.001
        print("Degree Choosen "+str(degree))
        length_data = len(data_new[0])
        print("Intial weights chooosen for Gradient Descent Using feature Transform are ")
        weights = np.random.uniform(-0.1,0.1,length_data)
        print(weights)
        print("\n")
        number_of_iterations=10000
        for i in range(number_of_iterations):
            new_weights = gradient_descent(data_new, weights, learning_rate, lambdaval)
            weights = new_weights
            new_error = calculate_error(data_new, new_weights, lambdaval)
            #print(new_error)
        degree_error_values.append(new_error)
        print("\n")
        print("Final weights using gradient descent are ")
        print((weights))
        print("\n")
        print("Final error using gradient descent are ")
        print(new_error)
        print("\n")
        X = data_column_added
        plot_degree(data_new, weights, degree, new_error)
        print("Plot is generated successfully for Degree: " + str(degree))
    plt.plot(degree_values,degree_error_values)
    plt.savefig("./figures/Q2.4,5_errorVsDegree.png")
    plt.close()
    print("\n")
    print('####################################################################################')
    print('\n')
    print("Varying Regularisation parameters as specified in Q2.6 is ")
    lambda_values = [0.0000000000000001,0.000000000001,0.00000001,0.000001,0.001,1, 2]
    degree = 4
    print("\n")
    lambda_error_values = []
    for lambdaval in lambda_values:
        data_new = featuretransform(data_column_added, degree)
        learning_rate = 0.01
        print("Lambda Choosen " + str(lambdaval))
        length_data = len(data_new[0])
        print("Intial weights chooosen for Gradient Descent Using feature Transform are ")
        weights = np.random.uniform(-0.1, 0.1, length_data)
        print(weights)
        print("\n")
        number_of_iterations = 10000
        for i in range(number_of_iterations):
            new_weights = gradient_descent(data_new, weights, learning_rate, lambdaval)
            weights = new_weights
            new_error = calculate_error(data_new, new_weights, lambdaval)
            # print(new_error)
        lambda_error_values.append(new_error)
        print("\n")
        print("Final weights using gradient descent are ")
        print((weights))
        print("\n")
        print("Final error using gradient descent are ")
        print(new_error)
        print("\n")
        # plot_degree_weigths(data_new,weights)
        X = data_column_added
        plot_degree_lambda(data_new, weights, lambdaval, new_error)
        print("Plot is generated successfully for Lambda: " + str(lambdaval))
    plt.plot(lambda_values, lambda_error_values)
    plt.savefig("./figures/Q2.6_errorVsLambda.png")
    plt.close()
    print("Program Execution is done")


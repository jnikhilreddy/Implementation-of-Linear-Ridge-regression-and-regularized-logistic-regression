import numpy as np
import matplotlib.pyplot as plt


####################################################################################
#       Function of Encoding defined in Question 1.1 ###############################
####################################################################################
####################################################################################

def data_encode(file_name) :
    f = open(file_name)
    new_list=[]
    for word in f.read().split():
        for m in word.split(','):
            if m=='F':
                new_list.append(1)
                new_list.append(0)
                new_list.append(0)
            elif m=='M':
                new_list.append(0)
                new_list.append(0)
                new_list.append(1)
            elif m=='I':
                new_list.append(0)
                new_list.append(1)
                new_list.append(0)
            else :
                new_list.append(float(m))
    return new_list


####################################################################################
##   Standardise the Data as defined in Question 1.2 ###############################
####################################################################################
####################################################################################



def standardise_data(data) :
    mean = np.mean(data,axis=0)
    std  = np.std(data,axis=0)
    #print(mean)
    #print(std)
    for k in range(4177):
        for i in range(3,10) :
            data[k][i] = data[k][i] - mean[i]
            if std[i]!=0 :
                data[k][i] = data[k][i]/std[i]
    #print(np.std(data,axis=0))
    return data



####################################################################################
##   Implementation of linear regression function Q1.3  ################################
####################################################################################
####################################################################################


def mylinridgereg(matrix_x,matrix_y,value_of_lambda):
    matrix1 = np.mat( np.linalg.inv(np.mat(matrix_x.T) * np.mat(matrix_x) + value_of_lambda*Identity(matrix_x.T.shape[0])))
    matrix2 = np.mat(matrix_x.T) * np.mat(matrix_y)
    return matrix1*matrix2

def mylinridgeregeval(matrix_x,matrix_w):
        matrix3= np.mat(matrix_x) * np.mat(matrix_w)
        return matrix3

def Identity(n):
    T = np.full((n, n), 0)
    for i in range(n):
        T[i][i] = 1
    return np.mat(T)



####################################################################################
##  Partition into test and train data   Q1.4 ######################################
####################################################################################
####################################################################################



#Splitting the data into train and test data and standardising train data and test data
#with train data mean and standard deviation.

def train_test_split(data,fraction) :
    np.random.shuffle(data)
    value= int(0.2 * 4177)
    rem = 4177 -value
    left_test = 4177 - value
    rem = int(rem*fraction)
    train_data=data[:rem]
    test_data=data[left_test:4177]
    train_data_y = train_data[:, -1]
    test_data_y = test_data[:, -1]
    train_data_x = np.delete(train_data, 10, axis=1)
    test_data_x = np.delete(test_data,10,axis=1)
    test_data_y = test_data_y.reshape(value, 1)
    test_data_x = test_data_x.reshape(value, 10)
    train_data_y = train_data_y.reshape(rem, 1)
    train_data_x = train_data_x.reshape(rem, 10)
    #print(len(train_data_x))
    mean = np.mean(train_data_x, axis=0)
    std = np.std(train_data_x, axis=0)
    # print(mean)
    # print(std)
    for k in range(len(train_data_x)):
        for i in range(3, 10):
            train_data_x[k][i] = train_data_x[k][i] - mean[i]
            if std[i] != 0:
                train_data_x[k][i] = train_data_x[k][i] / std[i]
    for k in range(len(test_data_x)):
        for i in range(3, 10):
            test_data_x[k][i] = test_data_x[k][i] - mean[i]
            if std[i] != 0:
                test_data_x[k][i] = test_data_x[k][i] / std[i]
    return train_data_x,train_data_y,test_data_x,test_data_y



####################################################################################
##  Implementation of mean Square Error function Q1.5 ##############################
####################################################################################
####################################################################################

def meansquarederr(T, Tdash):
    len_prediction = len(T)
    sum_square = 0
    for i in range(0, len_prediction):
        sum_square += (T[i] - Tdash[i]) ** 2
    final_val = sum_square
    return final_val / len_prediction




####################################################################################
####  Program Execution starts from here  ##########################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    data_encoded = data_encode("linregdata")
    #print(len(data_encoded))
    data_encoded_format=np.asarray(data_encoded)
    data_encoded_format=data_encoded_format.reshape(4177, 11)
    print("Encoded data as specified in Q1.1 is ")
    print(data_encoded_format)
    print('\n')
    print('####################################################################################')
    print('\n')
    print('\n')
    print("Standardise the data as specified in Q1.2 is ")
    data_standard = standardise_data(data_encoded_format)
    print(data_standard)
    print('\n')
    print('####################################################################################')
    print('\n')
    print('\n')
    print("Implementation of Linear Regression function as specified in Q1.3 is ")
    data_y = data_standard[:,-1]
    #print(data_y)
    data_x = np.delete(data_standard,10,axis=1)
    #print(data_x)
    data_y = data_y.reshape(4177,1)
    data_x = data_x.reshape(4177,10)
    linreg_weights = mylinridgereg(data_x,data_y,3)
    linear_prediction = mylinridgeregeval(data_x,linreg_weights)
    print("Prediction of the model for lambda = 3")
    print(linear_prediction)
    #print(len(linear_prediction))
    print('\n')
    print('####################################################################################')
    print('\n')
    print('\n')
    print("Partition into test and train data as specified in Q1.4 is ")
    train_x,train_y,test_x,test_y = train_test_split(data_standard,0.9)
    train_weights = mylinridgereg(train_x,train_y,100)
    train_prediction = mylinridgeregeval(train_x,train_weights)
    print("Predicted values for fraction = 0.9 and Lambda = 100 ")
    print("Prediction for train data are ")
    print('\n')
    print(train_prediction)
    print('\n')
    print("Total number of the train predictions are "+ str(len(train_prediction)))
    print('\n')
    print('\n')
    print("Prediction for the test data are ")
    #print(len(train_prediction))
    print('\n')
    test_predict = mylinridgeregeval(test_x,train_weights)
    print(test_predict)
    print('\n')
    print('Total number of test data predictions are '+str(len(test_predict)))
    print('\n')
    print('###########################################################################################')
    print('\n')
    print('\n')
    print("Computing mean square error as specified in Q1.5 is ")
    print("\n")
    result_mean_square = meansquarederr(train_prediction,train_y)
    print("Value of mean Square error for above Train data fraction=0.9")
    print(result_mean_square)
    print('\n')
    print('###########################################################################################')
    print('\n')
    print('\n')
    print("Testing data is standardised with mean and standard deviation of the Testing data for every new fraction value")
    print("Written in Train_test_split function specified above")
    print('\n')
    print('\n')
    print("Effect of λ on error change,average mean square error as specified in Q1.6 is ")
    lambda_values = [0.0001,0.01,0.1,2,3,5,10,25,50,100]
    fraction_values =[0.1,0.2,0.3,0.4,0.45,0.55,0.6,0.7,0.9,1.0]
    sum_train_mse = 0
    sum_test_mse =0
    for lambda_val in lambda_values :
        for fraction_val in fraction_values :
            train_x,train_y,test_x,test_y=train_test_split(data_standard,fraction_val)
            train_weights = mylinridgereg(train_x,train_y,lambda_val)
            train_prediction=mylinridgeregeval(train_x,train_weights)
            mean_train_error = meansquarederr(train_prediction,train_y)
            test_prediction = mylinridgeregeval(test_x,train_weights)
            mean_test_error  = meansquarederr(test_prediction,test_y)
            sum_train_mse+=mean_train_error
            sum_test_mse+=mean_test_error
    average_train_mse= sum_train_mse/100
    average_test_mse = sum_test_mse/100
    print("Average Training MSE over 100 iterations is   ")
    print(average_train_mse)
    print("Average Testing MSE over 100 iterations  is ")
    print(average_test_mse)
    print('\n')
    print('##############################################################################################')
    print('\n')
    print('\n')
    print("Effect of mean square error Vs Lambda with respect to Fraction value as specified in Q1.7")
    lambda_values = [0.00000000001,0.0001, 0.1, 2, 3, 5, 10, 25, 50, 100]
    fraction_values = [0.03, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sum_train_mse = 0
    sum_test_mse = 0
    training_mse_values=[]
    testing_mse_values=[]
    data_standard = data_encoded_format
    print('\n')
    print("Running 100 iterations for each combination of fraction and lambda to Calculate average for 100 iterations :")
    print("\n")
    number_of_iterations=100
    for fraction_val in fraction_values:
        train_x, train_y, test_x, test_y = train_test_split(data_standard, fraction_val)
        training_mse_values = []
        testing_mse_values = []
        for lambda_val in lambda_values:
            train_mse=0
            test_mse =0
            for i in range(number_of_iterations):
                train_weights = mylinridgereg(train_x, train_y, lambda_val)
                train_prediction = mylinridgeregeval(train_x, train_weights)
                mean_train_error = meansquarederr(train_prediction, train_y)
                test_prediction = mylinridgeregeval(test_x, train_weights)
                mean_test_error = meansquarederr(test_prediction, test_y)
                train_mse+= float(mean_train_error.item(0,0))
                test_mse+= float(mean_test_error.item(0,0))
            training_mse_values.append(train_mse/number_of_iterations)
            testing_mse_values.append(test_mse/number_of_iterations)
        plt.plot(lambda_values, training_mse_values, label='Training')
        plt.xlabel("Different Lambda values")
        plt.ylabel("Average MSE values for Training And Testing ")
        axes = plt.gca()
        axes.set_ylim([0, 10])
        plt.title("Lambda values Vs training MSE error,Testing MSE error: for frac =  " + str(fraction_val))
        #plt.show()
        #plt.close()

        plt.plot(lambda_values, testing_mse_values, label='Testing')
        #plt.xlabel("Different Lambda values")
        #plt.ylabel("Average MSE values for Testing ")
        #plt.title("Lambda values Vs Testing MSE error for Fraction value : " + str(fraction_val))
        #plt.savefig("q1.7_test_" + str(fraction_val) + ".png")
        #plt.show()
        plt.legend(loc='best')
        plt.savefig("./figures/q1.7_train_test_" + str(fraction_val) + ".png")
        plt.close()
    print("Plots are generated for Q1.7")
    print('\n')
    print('############################################################################################')
    print('\n')
    print('\n')
    print("Effect of minimum mean square error vs λ with respect to fraction value as specified in Q1.8")
    average_min_training_mse_values = []
    average_min_testing_mse_values = []
    min_lambda_testing=[]
    for fraction_val in fraction_values:
        train_x, train_y, test_x, test_y = train_test_split(data_standard, fraction_val)
        training_mse_values=[]
        testing_mse_values=[]
        for lambda_val in lambda_values:
            train_mse=0
            test_mse=0
            for i in range(number_of_iterations) :
                train_weights = mylinridgereg(train_x, train_y, lambda_val)
                train_prediction = mylinridgeregeval(train_x, train_weights)
                mean_train_error = meansquarederr(train_prediction, train_y)
                test_prediction = mylinridgeregeval(test_x, train_weights)
                mean_test_error = meansquarederr(test_prediction, test_y)
                train_mse += float(mean_train_error.item(0, 0))
                test_mse += float(mean_test_error.item(0, 0))
            training_mse_values.append(train_mse/number_of_iterations)
            testing_mse_values.append(test_mse/number_of_iterations)
        average_min_training_mse_values.append(min(training_mse_values))
        average_min_testing_mse_values.append(min(testing_mse_values))
        lambdamin = testing_mse_values.index(min(testing_mse_values))
        min_lambda_testing.append(lambda_values[lambdamin])
    plt.plot(fraction_values, average_min_testing_mse_values,color='blue', marker="d")
    plt.xlabel("Different Fraction values")
    plt.ylabel("minimum Average MSE values for Testing ")
    plt.title("fraction values Vs minimum average testing MSE error")
    plt.savefig("./figures/q1.8_testing_fraction" + ".png")
    # plt.show()
    plt.close()
    plt.plot(fraction_values,min_lambda_testing,color='green', marker='o')
    plt.xlabel("Different Fraction values")
    plt.ylabel("Lambda corresponding Minimum Average MSE values for Testing ")
    plt.title("Lambda values for min Testing MSE error")
    plt.savefig("./figures/q1.8_fraction_lambda" +".png")
    # plt.show()
    plt.close()
    print("plots are generated for Q1.8")
    print('\n')
    print('##########################################################################################')
    print('\n')
    print('\n')
    print("Actual values Vs Predicted values as specified in Q1.9")
    train_x, train_y, test_x, test_y = train_test_split(data_standard,1)
    train_weights = mylinridgereg(train_x, train_y, 3)
    train_prediction = mylinridgeregeval(train_x, train_weights)
    mean_train_error = meansquarederr(train_prediction, train_y)
    test_prediction = mylinridgeregeval(test_x, train_weights)
    mean_test_error = meansquarederr(test_prediction, test_y)
    plt.plot(train_prediction, train_y, 'y^', color='brown')
    A = []
    for i in range(25):
        A.append(i)
    plt.plot(A, A, 'b-')
    plt.xlabel(" Actual Values")
    plt.ylabel(" Predicted Values")
    plt.title("Training data")
    plt.legend("plot for Training data --- Actual Vs Predicted")
    plt.savefig("./figures/q1.9_train"+".png")
    plt.close()
    plt.plot(test_prediction, test_y, 'y^', color='red')
    A = []
    for i in range(25):
        A.append(i)
    plt.plot(A, A, 'b-')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Testing data")
    plt.legend("plot for Testing data -- Actual Vs Predicted")
    plt.savefig("./figures/q1.9_test" + ".png")
    print("\n")
    print("plots are generated for Q1.9")
    plt.close()

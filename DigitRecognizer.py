import numpy as np
import matplotlib.pyplot as plt

train_data = np.load("train_data.npy")
train_labels = np.load("train_labels.npy")
test_data = np.load("test_data.npy")
test_labels = np.load("test_labels.npy")

# The data is a 2d matrix with the number of columns = 256. The 256 columns are float values between -1 and 1 indicating the
# grayscale value of a 16x16 image. Each row represents 1 image. The labels represent what digit was written (label 1 corresponds
# to digit 1 and label -1 corresponds to digit 5)

# Problem 3 Part A: Plotting a 1 and a 5
    
# Plotting a number 1:
i=0
plt.figure()
plt.imshow(train_data[i].reshape(16, 16), cmap=plt.cm.binary)
plt.title("Example of a written number 1")
plt.show()



# Plotting a number 5:
i=1200
plt.figure()
plt.imshow(train_data[i].reshape(16, 16), cmap=plt.cm.binary)
plt.title("Example of a written number 5")
plt.show()



# Problem 3 Part B: Extracting symmetry and average intensity features from the training data

# Extracting Average Intensity Data. The average intensity of each image is stored in the array average_intensity

average_intensity = np.zeros(np.shape(train_labels))
for i in range(len(train_data)): # For all of the rows (individual images)
    for j in range(len(train_data[i])): # For each grid square within each 16x16 image
        average_intensity[i] = average_intensity[i] + (train_data[i][j] / len(train_data[i]))

# Extracting Symmetry Data. The symmetry of each image is stored in the array symmetry

symmetry = np.zeros(np.shape(train_labels))
for i in range(len(train_data)):
    temp = np.array(train_data[i].reshape(16, 16)) # Must Reshape to use fliplr
    temp2 = np.fliplr(temp)
    temp = temp.flatten()
    temp2 = temp2.flatten()
    for j in range(len(train_data[i])):
        symmetry[i] = symmetry[i] + ((temp[j] - temp2[j])**2 / len(train_data[i]))
        

# Problem 3 Part C: Plotting the features that I extracted from the digits

a = 0
b = 0
data1 = []
data2 = []
plt.figure()

# Data1 corresponds to values for the digit 1. Data2 corresponds to values for the digit 5.
for i in range(len(train_labels)):
    if(train_labels[i] == 1):
        data1.append([average_intensity[i], symmetry[i]])
    else:
        data2.append([average_intensity[i], symmetry[i]])

# Plot the extracted data.
for i in range(len(data1)):
    if(i == 0):
        plt.plot(data1[i][0], data1[i][1], 'bo', label='Digit 1')
    else:
        plt.plot(data1[i][0], data1[i][1], 'bo')
for i in range(len(data2)):
    if(i == 0):
        plt.plot(data2[i][0], data2[i][1], 'rx', label='Digit 5')
    else:
        plt.plot(data2[i][0], data2[i][1], 'rx')

plt.legend()
plt.xlabel("Average Intensity")
plt.ylabel("Symmetry")
plt.title("Average Intensity and Symmetry for the Written Numbers 1 and 5")
plt.show()



# Problem 4

# Gradient Descent Algorithm
def gradient_descent(X, y, learning_rate, max_iter):
    def gradient():
        
        # Calculate gradient: -(yn)(xn) * (1 / (1 + exp((yn)(wT)(x))))
        # X = List of digits with [1, avg intensity, symmetry]
        # y = labels
        # w = weights

        # I have to calculate for each feature separately.

        wt = np.transpose(w)
        N = len(X)

        gradient = 0
        loss = 0

        for i in range(len(X)):
            yn = y[i]
            Xn = X[i]
            temp = yn * np.matmul(Xn, wt) # Because of how I created the arrays, matmul(Xm, wt) is the correct way to multiply
            
            gradient += (-1) * (yn) * (Xn) / (1 + np.exp(temp))
            loss += np.log(1 + np.exp((-1) * temp))

        gradient = (gradient/N)
        loss = (loss/N)

        return gradient, loss

         # Gradient Solver

    w = np.zeros(3) # Size of 3 because I am using 2 features and want to set the weights for each feature
    loss_list = [] # loss values for each iteration 

    # Everything below is correct
    for n in range(max_iter):

        # Adjust weights: w = w - learning_rate * gradient
        grad, loss = gradient()
        w = w - (learning_rate * grad)

        # Check if converged
        loss_list.append(loss)
        if ((n > 0) and (abs(loss_list[n] - loss_list[n - 1]) < 1e-5)): # If the change in loss is negligible, then terminate
            print("Normal Gradient Descent converged at " + str(n + 1) + " gradient computations")
            return w
        
        if n == max_iter - 1: # On the last iteration
            return w

# Data will be the feature matrix (X in the algorithm)
data = []
for i in range(len(average_intensity)):
    data.append([1, average_intensity[i], symmetry[i]])
data = np.array(data)

# Running the GD algorithm
w = gradient_descent(data, train_labels, 1, 10000)

a = -1 * w[1] / w[2] # Slope
b = -1 * w[0] / w[2] # Y intercept
c = -1 * w[0] / w[1] # X intercept

x = [-1, 0.2]   # Arbitrary X values to plot the line through so it shows well on the graph.
c = a * -1 + b
d = a * 0.2 + b
y = [c, d]


# Plot the regression line based on the points above
plt.plot(x, y, linestyle='solid', label = "Logistic Regression Line")

# Plot the data points
for i in range(len(data1)):
    if(i == 0):
        plt.plot(data1[i][0], data1[i][1], 'bo', label='1')
    else:
        plt.plot(data1[i][0], data1[i][1], 'bo')
for i in range(len(data2)):
    if(i == 0):
        plt.plot(data2[i][0], data2[i][1], 'rx', label='5')
    else:
        plt.plot(data2[i][0], data2[i][1], 'rx')

plt.legend()
plt.xlabel("Average Intensity")
plt.ylabel("Symmetry")
plt.show()
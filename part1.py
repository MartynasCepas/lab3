import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def read_data():
    results = []
    years = []
    sunspots = []

    f = open('sunspot.txt', 'r')

    lines = f.readlines()
    for line in lines:
        temp = line.split('\t')
        years.append(int(temp[0]))
        sunspots.append(int(temp[1]))

    f.close()

    results.append(years)
    results.append(sunspots)
    return results

def convert_to_matrix(data):
    years = data[0]
    sunspots = data[1]

    L = len(sunspots)
    A = np.array([[5,11],[11,16]])
    B = np.array([[16]])

    for i in range(L):
        if i > 3:
            A = np.append(A, [[sunspots[i-2], sunspots[i-1]]], axis = 0)
    for i in range(L):
        if i > 2:
            B = np.append(B, [[sunspots[i]]], axis = 0)

    return A, B


def drawPredictionComparison1(data, Tsu, Lu):
    years = data[0]
    values = data[1]
    fig = plt.figure()
    ax1 = plt.plot(years[:Lu], values[:Lu], 'bo-', label="Real values")
    ax2 = plt.plot(years[:Lu], Tsu, 'yo-', label="Predicted values")
    plt.title("Real vs Predicted values <200")
    plt.xlabel("Years")
    plt.ylabel("Sun spot amount")
    plt.legend()
    plt.show()

def drawPredictionComparison2(data, Tsu, Lu):
    years = data[0]
    values = data[1]
    years = years[Lu:]
    values = values[Lu:]
    fig = plt.figure()
    ax1 = plt.plot(years, values, 'bo-', label="Real values")
    ax2 = plt.plot(years[:-2], Tsu, 'yo-', label="Predicted values")
    plt.title("Real vs Predicted values 200>")
    plt.xlabel("Years")
    plt.ylabel("Sun spot amount")
    plt.legend()
    plt.show()

def get_error(Ts, T, year):
    error = T - Ts
    plt.plot(year, error, marker='.', label='error')
    plt.xlabel("Years")
    plt.ylabel("Sun spot amount")
    plt.title("Error vector")
    plt.legend()
    plt.show()

    plt.hist(error)
    plt.title("Error")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.show()
    return error

def get_MSE(n, error):
    error_sum = 0

    for i in error:
        error_sum += i*i

    mse_value = 1/n * error_sum
    print("MSE VALUE: ", mse_value)
    return mse_value

def get_MAD(error):
    median = np.median(np.absolute(error))
    print("Median: ", median)
    return median

# Press the green button in the gutter to run the script.

data = read_data()
plt.plot(data[0], data[1])
plt.xlabel("Year")
plt.ylabel("Sunspots")
plt.title("Year vs Sunspot data graph")
plt.show()

p, t = convert_to_matrix(data)
#  print("Input matrix: \n", p)
#  print("Output matrix: \n", t)

print("Pradinis reiksmiu saraso dydis: ", len(data[0]))
print("Ivesties reiksmiu saraso dydis: ", len(p))
print("Isvesties reiksmiu saraso dydis: ", len(t))

Lu = 200 # training data count
Pu, Tu = p[:Lu], t[:Lu]
Pu_test, Tu_test = p[Lu:], t[Lu:]

model = LinearRegression().fit(Pu, Tu)
w1 = model.coef_[[0], [0]]
w2 = model.coef_[[0], [1]]

Tsu = model.predict(Pu)
b = model.intercept_

print("W1: ", w1)
print("W2: ", w2)
print("b: ", b)
drawPredictionComparison1(data, Tsu, Lu)

years = data[0]
values = data[1]

error = get_error(Tsu[:,0], Pu[:,1], years[:Lu])
mse = get_MSE(Lu, error)
mad = get_MAD(error)

Pu, Tu = p[Lu:], t[Lu:]
Tsu = model.predict(Pu)
drawPredictionComparison2(data, Tsu, Lu)


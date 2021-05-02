from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

dataset_path = "data/final_cars_datasets.csv"


def readDataset():
    cars_data = pd.read_csv(dataset_path)
    X = cars_data.iloc[:, :-1].values
    y = cars_data.iloc[:, -1].values
    cars_data.head()
    return cars_data


def groupVariables(cars_data):
    numerical_vars = ['price', 'year', 'mileage', 'engine_capacity', ]
    numerical_vars_idx = cars_data.columns.get_indexer(numerical_vars)
    cat_vars = ['mark', 'model', 'transmission', 'drive', 'hand_drive', 'fuel']
    cat_vars_idx = cars_data.columns.get_indexer(cat_vars)
    return [cat_vars, numerical_vars]


def createDataset():
    data = readDataset()
    df = pd.DataFrame(data)
    df = df.drop(df.columns[[0]], axis=1)
    variables = groupVariables(data)
    cat_vars = variables[0]
    numerical_vars = variables[1]
    return df, cat_vars, numerical_vars


def tolydiniuAnalize():
    global df_numerical
    print("Skaitinio tipo reiksmes")
    df_numerical = df.select_dtypes(include='int64')
    print(df_numerical)
    print("\nBendras reiksmiu skaicius: ")
    print(df_numerical.count())
    print("\nTrukstamu reiksmiu skaicius:")
    print(df_numerical.isnull().sum())
    print("\nKardinalumas:")
    print(df_numerical.nunique())
    print("\nMinimali verte:")
    print(df_numerical.min())
    print("\nMaksimali verte:")
    print(df_numerical.max())
    print("\nPirmas kvartilis:")
    print(df_numerical.quantile(.25))
    print("\nTrecias kvartilis:")
    print(df_numerical.quantile(.75))
    print("\nVidurkis:")
    print(df_numerical.mean())
    print("\nMediana:")
    print(df_numerical.median())
    print("\nStandartinis nuokrypis:")
    print(df_numerical.std())


def kategoriniuAnalize():
    global df_categorical, z
    print("Kategorinio tipo reiksmes")
    df_categorical = df.select_dtypes(include='object')
    print(df_categorical)
    print("\nBendras reiksmiu skaicius: ")
    print(df_categorical.count())
    print("\nTrukstamu reiksmiu skaicius:")
    print(df_categorical.isnull().sum())
    print("\nKardinalumas:")
    print(df_categorical.nunique())
    print("\nModa:")
    print(df_categorical.mode())
    print("\nModos daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in df_categorical.mode().iterrows()]).sum())
    print("\nModos procentinis daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in df_categorical.mode().iterrows()]).sum() / len(
        df_categorical.index) * 100)
    print("\nAntroji moda:")
    temp_df = df_categorical
    modas = []
    for x, y in df_categorical.mode().iterrows():
        for z in range(0, len(y)):
            modas.append(y[z])
    for x in modas:
        temp_df = temp_df.replace(to_replace=x, value=np.nan, regex=True)
    print(temp_df.mode())
    print("\nAntrosios modos daznumas:")
    print(pd.concat([df_categorical.eq(x) for _, x in temp_df.mode().iterrows()]).sum())
    print("\nAntrosios modos procentinis daznumas:")
    print(
        pd.concat([df_categorical.eq(x) for _, x in temp_df.mode().iterrows()]).sum() / len(df_categorical.index) * 100)


def fixOutliers():
    global z, df_numerical
    print("\nOutliers aptikimas:")
    z = np.abs(stats.zscore(df_numerical))
    print(z)
    threshold = 3
    print(np.where(z > 3))
    df_numerical = df_numerical[(z < 3).all(axis=1)]
    print(df_numerical.count())


def scatterTolydinis():
    print("\nGrafikai su stipria tiesine priklausomybe: metai/rida, kaina/rida")
    ax1 = df_numerical.plot.scatter(x='year', y='mileage')
    plt.title("year vs mileage")
    ax2 = df_numerical.plot.scatter(x='price', y='mileage')
    plt.title("price vs mileage")
    print("\nGrafikai su silpna tiesine priklausomybe: metai/variklis, kaina/variklis")
    ax3 = df_numerical.plot.scatter(x='year', y='engine_capacity')
    plt.title("year vs engine_capacity")
    ax4 = df_numerical.plot.scatter(x='mileage', y='engine_capacity')
    plt.title("mileage vs engine_capacity")
    plt.show()


def fuelvsTransmissionGraph():
    c2 = df_categorical['fuel'].value_counts().plot(kind='bar')
    c1 = df_categorical.query('`transmission` == "at"')
    c1 = c1.groupby(['fuel']).size().reset_index(name='counts')
    ax5 = c1.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'transmission' == at")
    c3 = df_categorical.query('`transmission` == "mt"')
    c3 = c3.groupby(['fuel']).size().reset_index(name='counts')
    ax6 = c3.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'transmission' == mt")


def drivevsFuelGraph():
    c4 = df_categorical.query('`drive` == "2wd"')
    c4 = c4.groupby(['fuel']).size().reset_index(name='counts')
    ax10 = c4.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'Drive' == 2wd")
    c6 = df_categorical.query('`drive` == "4wd"')
    c6 = c6.groupby(['fuel']).size().reset_index(name='counts')
    ax11 = c6.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'Drive' == 4wd")
    c7 = df_categorical.query('`drive` == "awd"')
    c7 = c7.groupby(['fuel']).size().reset_index(name='counts')
    ax12 = c7.plot.bar(x="fuel", y="counts", rot=0)
    plt.title("'Fuel' when 'Drive' == awd")


def drawBoxplots():
    boxplot1 = df.boxplot(by='drive', column='price', grid=False)
    plt.show()
    boxplot2 = df.boxplot(by='mark', column='price', grid=False, rot=90)
    plt.show()
    boxplot3 = df.boxplot(by='fuel', column='engine_capacity', grid=False, rot=90)
    plt.show()


def covandcorr():
    global corr
    print("\nKovariacija:")
    print(df.cov())
    corr = df.corr()
    print("\nKoreliacija:")
    print(corr)
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True)
    plt.show()


def normalizeDf():
    normalized_df = (df_numerical - df_numerical.min()) / (df_numerical.max() - df_numerical.min())
    print("Normalizuotas duomenu rinkinys:")
    print(normalized_df)


def categoricalToNumerical():
    print("Kategoriniai i tolydinius")
    for col_name in df_categorical.columns:
        if df_categorical[col_name].dtype == 'object':
            df_categorical[col_name] = df_categorical[col_name].astype('category')
        df_categorical[col_name] = df_categorical[col_name].cat.codes
    print(df_categorical)


def drawHistograms():
    df.hist(column='price')
    df.hist(column='year')
    df.hist(column='engine_capacity')
    df.hist(column='mileage')
    df.groupby(['transmission']).size().reset_index(name='counts').plot.bar(x='transmission', y='counts')
    plt.title('transmission')
    #df.hist(column='drive')
    df.groupby(['drive']).size().reset_index(name='counts').plot.bar(x='drive', y='counts')
    plt.title('drive')
    #df.hist(column='hand_drive')
    df.groupby(['hand_drive']).size().reset_index(name='counts').plot.bar(x='hand_drive', y='counts')
    plt.title('hand_drive')
    #df.hist(column='fuel')
    df.groupby(['fuel']).size().reset_index(name='counts').plot.bar(x='fuel', y='counts')
    plt.title('fuel')
    plt.show()

def convert_to_matrix(data):
    data = data.to_numpy()
    years = data[:,1]
    price = data[:,0]

    L = len(price)
    A = np.array([[5,11],[11,16]])
    B = np.array([[16]])

    for i in range(L):
        if i > 3:
            A = np.append(A, [[price[i-2], price[i-1]]], axis = 0)
    for i in range(L):
        if i > 2:
            B = np.append(B, [[price[i]]], axis = 0)

    return A, B

def get_error(Ts, T, year):
    error = T - Ts
    plt.plot(year, error, marker='.', label='error')
    plt.xlabel("Years")
    plt.ylabel("Sun spot amount")
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
if __name__ == '__main__':
    plt.close('all')
    # Read data
    df, cat_vars, numerical_vars = createDataset()

    print(cat_vars)
    print(numerical_vars)

    # 2 Skaitinio tipo
    tolydiniuAnalize()

    # 2 Kategorinio tipo
    kategoriniuAnalize()

    # 5-6
    fixOutliers()


    # 9 Duomenu normalizacija [0;1]
    normalizeDf()

    # 10 Kategoriniai i tolydinius
    categoricalToNumerical()

    # 3.2 prediction model
    p, t = convert_to_matrix(df_numerical)
    Lu = 2000  # training data count
    Pu, Tu = p[:Lu], t[:Lu]

    model = Sequential()
    model.add(Dense(1, input_dim=2))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mse'])

    weights_before_train = model.get_weights()

    history = model.fit(Pu, Tu, epochs=1000, batch_size=50, verbose=1)

    weights_after_train = model.get_weights()
    print("Weights before train: ", weights_before_train)
    print("Weights after train: ", weights_after_train)

    model_predictions = model.predict(Pu)

    plt.plot(history.history['mse'])
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.show()

    data = df_numerical.to_numpy()
    years = data[:, 1]
    price = data[:, 0]
    fig = plt.figure()
    ax1 = plt.plot(price[:Lu],years[:Lu] , 'bo-', label="Real values")
    ax2 = plt.plot(model_predictions, years[:Lu], 'yo-', label="Predicted values")
    plt.title("Real vs Predicted values")
    plt.xlabel("Price")
    plt.ylabel("Years")
    plt.legend()
    plt.show()

    error = get_error(model_predictions[:, 0], Pu[:, 1], years[:Lu])
    mse = get_MSE(Lu, error)
    mad = get_MAD(error)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_estimators(df, independent_variable):
    # for every dependent column of the input DataFrame this function returns an estimator in a list

    i = 0
    estimators = []
    lr = LinearRegression()

    for column_index in df.axes[1]:
        if column_index == independent_variable:

            x = df.iloc[:, i].values.reshape(-1, 1)

        else:

            y = df.iloc[:, i].values.reshape(-1, 1)

            estimators.append(lr.fit(x, y))

        i += 1

    return estimators
    

def main():
    df_train = pd.read_csv(filepath_or_buffer="Datasets1/train.csv")
    x_train = df_train.iloc[:, 0].values.reshape(-1, 1)
    y1_itrain = df_train.iloc[:, 1].values.reshape(-1, 1)


    df_ideal = pd.read_csv(filepath_or_buffer="Datasets1/ideal.csv")
    x_ideal = df_ideal.iloc[:, 0].values.reshape(-1, 1)
    y1_ideal = df_ideal.iloc[:, 1].values.reshape(-1, 1)

    # print(df_ideal.iloc[:, :])
    # print(df_ideal.axes[1].__iter__)

    # step 1
    # get estimator for every of the 50 functions (--> array or list with all the estimators)
    ideal_estimators = get_estimators(df_ideal, 'x')

    # input x values from training data into estimators

    # calculate deviation (Least-Square) of train x,y and ideal x,y

    # get the lowest deviation and save the index no of the ideal function it came from









    # plot this shit
    # plt.scatter(x_ideal, y1_ideal)
    # plt.plot(x_ideal, y1_func, color='red')
    # plt.show()

if __name__ == '__main__':
    main()
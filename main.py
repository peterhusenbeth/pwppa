import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def map_deviations(df_from: pd.DataFrame, df_to: pd.DataFrame, indep_var_col_index_from: int = 0, indep_var_col_index_to: int = 0):
    # input x values from training data into estimators
    # calculate deviation (Least-Square) of train x,y and ideal x,y
    # get the lowest deviation and save the index no of the ideal function it came from

    df_deviations = pd.DataFrame(columns=df_from.columns[1:].insert(0, 'func_nr'), dtype=np.float64)

    # return None

    # insert rows as lists of deviations
    # for each estimator, calculate the sum of all y-deviations squared with every estimator
    # x = df_from.iloc[:, indep_var_col_index_from].values.reshape(-1, 1)

    func_nr = 0

    for estimator in get_estimators(df_to, indep_var_col_index_to):

        i = 0
        func_nr += 1
        row = {'func_nr': func_nr}

        for column in df_from.columns:

            if i != indep_var_col_index_from:
                
                row[column] = square_deviation(df_from[['x', column]], estimator)

            i += 1

        df_deviations.loc[len(df_deviations.index)] = row

    return df_deviations

def get_estimators(df: pd.DataFrame, indep_var_col_index: int = 0):
    # for every dependent column of the input DataFrame this function returns an estimator in a list

    estimators = []
    lr = LinearRegression()
    x = df.iloc[:, indep_var_col_index].values.reshape(-1, 1)
    i = 0

    for column in df.columns:

        if i != indep_var_col_index:

            y = df.iloc[:, i].values.reshape(-1, 1)

            estimators.append(lr.fit(x, y))

        i += 1

    return estimators

def square_deviation(data: pd.DataFrame, function: LinearRegression):
    #print("printing data: \n" + repr(data))
    return 8.908

def main():
    df_train = pd.read_csv(filepath_or_buffer="Datasets1/train.csv")
    df_ideal = pd.read_csv(filepath_or_buffer="Datasets1/ideal.csv")


    map_dev = map_deviations(df_train, df_ideal)

    print("map_deviations: ")
    print(map_dev)
    #print(get_estimators(df_ideal))

    #print(df_train.columns)
    #print(df_train.columns[1:].insert(0, 'ideal_func_no'))
    #print(["ideal_func_no"].append(df_train.columns[1:]))











    # plot this shit
    # plt.scatter(x_ideal, y1_ideal)
    # plt.plot(x_ideal, y1_func, color='red')
    # plt.show()

if __name__ == '__main__':
    main()
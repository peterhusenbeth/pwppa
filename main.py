import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def map_deviations(df_from_x: pd.DataFrame, df_from_y: pd.DataFrame, df_to_x: pd.DataFrame, df_to_y: pd.DataFrame):
    # input x values from training data into estimators
    # calculate deviation (Least-Square) of train x,y and ideal x,y
    # get the lowest deviation and save the index no of the ideal function it came from

    df_deviations = pd.DataFrame(columns=df_from_y.columns.insert(0, 'func_nr'), dtype=np.float64)

    # return None

    # insert rows as lists of deviations
    # for each estimator, calculate the sum of all y-deviations squared with every estimator
    # x = df_from.iloc[:, indep_var_col_index_from].values.reshape(-1, 1)

    func_nr = 0

    estimators = get_estimators(df_to_x, df_to_y)

    # iterate 50 ideal functions
    for estimator in estimators:

        func_nr += 1
        row = {'func_nr': func_nr}

        # iterate 4 training columns
        for column in df_from_y.columns:
                
            row[column] = square_deviation(df_from_x, df_from_y[column], estimator)

        df_deviations.loc[len(df_deviations.index)] = row

    return df_deviations

def get_estimators(df_x: pd.DataFrame, df_y: pd.DataFrame):
    # for every dependent column of the input DataFrame this function returns an estimator in a list

    estimators = []
    x = df_x.values.reshape(-1, 1)
    i = 0

    for column in df_y.columns:

        lr = LinearRegression()

        y = df_y.iloc[:, i].values.reshape(-1, 1) # df_y[column].values.reshape(-1, 1)???

        estimator = lr.fit(x, y)
        estimators.append(estimator)

        i += 1

    return estimators

def square_deviation(x_data: pd.Series, y_data: pd.Series, function: LinearRegression):
    # calculates the square deviation
    # (die Summe ist aber in Anhängigkeit davon, wie viele Datenpunkte zum Vergleich mit der Funktion genommen werden!)

    y_column = np.array(y_data).reshape(-1, 1)
    y_estimated = function.predict(np.array(x_data).reshape(-1, 1))

    print("y_column, y_estimated, differenz")
    print(y_column[0], y_estimated[0], y_column[0] - y_estimated[0])

    return np.float64(sum((y_column - y_estimated)**2))

def main():
    df_train = pd.read_csv(filepath_or_buffer="pwppa/Datasets1/train.csv")
    df_ideal = pd.read_csv(filepath_or_buffer="pwppa/Datasets1/ideal.csv")

    map_dev = map_deviations(df_train.iloc[:, 0], df_train.iloc[:, 1:5], df_ideal.iloc[:, 0], df_ideal.iloc[:, 1:51])

    print("map_deviations: ")
    print(map_dev)
    #print(get_estimators(df_ideal))

    #print(df_train.columns)
    #print(df_train.columns[1:].insert(0, 'ideal_func_no'))
    #print(["ideal_func_no"].append(df_train.columns[1:]))

    # OO Approach
    # class ideal function inherits from LinearRegression
    # class training data inherits from Dataframe?
    # class test data inherits from series?









    # plot this shit
    # plt.scatter(x_ideal, y1_ideal)
    # plt.plot(x_ideal, y1_func, color='red')
    # plt.show()

if __name__ == '__main__':
    main()
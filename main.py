import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# rewrite for only one training function at a time. return: Dataframe with columns: func_nr, estimator, square_dev, largest_dev
def get_deviations(df_from_x: pd.DataFrame, df_from_y: pd.DataFrame, estimators: list[LinearRegression]):
    # input x values from training data into estimators
    # calculate deviation (Least-Square) of train x,y and ideal x,y

    df_deviations = pd.DataFrame(columns=['func_nr', 'estimator', 'sqr_deviation', 'max_deviation'])

    # insert rows as lists of deviations
    # for each estimator, calculate the sum of all y-deviations squared with every estimator

    func_nr = 0

    # iterate 50 ideal functions
    for estimator in estimators:

        func_nr += 1
        row = {'func_nr': func_nr}

        row['estimator'] = estimator
        row['sqr_deviation'], row['max_deviation'] = deviations(df_from_x, df_from_y, estimator)

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

def deviations(x_data: pd.Series, y_data: pd.Series, function: LinearRegression):
    # calculates the square deviation
    # (die Summe ist aber in Anhängigkeit davon, wie viele Datenpunkte zum Vergleich mit der Funktion genommen werden!)

    y_column = np.array(y_data).reshape(-1, 1)
    y_estimated = function.predict(np.array(x_data).reshape(-1, 1))
    y_dev_abs = np.abs(y_column - y_estimated)

    # calculate square deviation
    sqr_deviation = np.float64(sum((y_dev_abs)**2))

    # find maximum single deviation
    max_deviation = np.float64(y_dev_abs.max())

    return sqr_deviation, max_deviation

def main():
    # read CSV files into DataFrames
    df_train = pd.read_csv(filepath_or_buffer="Datasets1/train.csv")
    df_ideal = pd.read_csv(filepath_or_buffer="Datasets1/ideal.csv")
    df_test = pd.read_csv(filepath_or_buffer="Datasets1/test.csv")

    # first step of the program
    # create table of square deviations from training data to ideal functions
    estimators = get_estimators(df_ideal.iloc[:, 0], df_ideal.iloc[:, 1:51])

    mapped_deviations = []

    for column in df_train.columns:

        if column != 'x':
            
            # get all the deviations for current column
            deviation_mapping = get_deviations(df_train['x'], df_train[column], estimators).sort_values(by = 'sqr_deviation')

            # save deviations-table in list for use after for loop
            mapped_deviations.append(deviation_mapping)

            # save the top row since this is the one with the least square deviation
            ideal_row = deviation_mapping.iloc[0]

            # convert x-values of training data to np.arra() for further calculation steps
            x_values = np.array(df_train['x']).reshape(-1, 1)
            
            # add training data to plot
            plt.scatter(x_values, df_train[column], label = column)

            # add ideal function line to plot
            plt.plot(x_values, ideal_row['estimator'].predict(x_values), color='black', label = 'func_nr: ' + str(ideal_row['func_nr']))
    
    # enhance and show plot 
    plt.xlabel('x')
    plt.xlabel('y')
    plt.legend()
    plt.show()

    # second step of the program
    # determine the largest deviation between training dataset and corresponding ideal function

    # check every test datapoint
    # for index, row in df_test.iterrows():
        









    # OO Approach
    # class ideal function inherits from LinearRegression
    # class training data inherits from Dataframe?
    # class test data inherits from series?

if __name__ == '__main__':
    main()
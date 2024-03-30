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
    df_train = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/train.csv")
    df_ideal = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/ideal.csv")
    df_test = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/test.csv")

    # first step of the program
    # create table of square deviations from training data to ideal functions
    estimators = get_estimators(df_ideal.iloc[:, 0], df_ideal.iloc[:, 1:51])

    train_ideal_mapping = pd.DataFrame(columns=['train_nr', 'func_nr', 'estimator', 'sqr_deviation', 'max_deviation'])

    i = 0

    for column in df_train.columns:

        if column != 'x':
            
            # get all the deviations for current column
            deviation_mapping = get_deviations(df_train['x'], df_train[column], estimators).sort_values(by = 'sqr_deviation')

            # save the top row since this is the one with the least square deviation
            ideal_row = deviation_mapping.iloc[0]

            # save deviations in table for use after for loop
            train_ideal_mapping.loc[i] = {'train_nr': column, 'func_nr': ideal_row['func_nr'], 'estimator': ideal_row['estimator'], 'sqr_deviation': ideal_row['sqr_deviation'], 'max_deviation': ideal_row['max_deviation']}

            # convert x-values of training data to np.arra() for further calculation steps
            x_values = np.array(df_train['x']).reshape(-1, 1)
            
            # add training data to plot
            plt.scatter(x_values, df_train[column], label = column)

            # add ideal function line to plot
            plt.plot(x_values, ideal_row['estimator'].predict(x_values), color='black', label = 'func_nr: ' + str(ideal_row['func_nr']))
        
            i += 1
    

    # second step of the program
    # determine the largest deviation between training dataset and corresponding ideal function

    df_test.insert(2, 'func_nr', None)
    df_test.insert(3, 'deviation', None)
    train_ideal_mapping.insert(5, 'test_data', None)

    i = 0

    # check every test datapoint
    for test_index, datapoint in df_test.iterrows():

        # check every ideal function
        for mapping_index, mapping in train_ideal_mapping.iterrows():

            # calculate functions estimated y value
            y_predicted = np.float64(mapping['estimator'].predict(np.array(datapoint['x']).reshape(-1, 1)))

            # calculate deviation from estimated y-value to test datapoint
            y_dev = np.abs(datapoint['y'] - y_predicted)

            # check if datapoint is in deviation range
            if y_dev <= (mapping['max_deviation'] * np.sqrt(2)):

                if df_test.loc[test_index, 'deviation'] is None or y_dev < df_test.loc[test_index, 'deviation']:

                    # add ideal func_nr and deviation from it to test database table
                    df_test.loc[test_index, 'func_nr'] = mapping['func_nr']
                    df_test.loc[test_index, 'deviation'] = y_dev

                    # add found test datapoint and corresponding deviation to the ideal function mapping table
                    new_test_data = [pd.Series([datapoint['x'], datapoint['y'], y_dev])]

                    if train_ideal_mapping.at[mapping_index, 'test_data'] is None:
                        train_ideal_mapping.at[mapping_index, 'test_data'] = new_test_data
                    else:
                        train_ideal_mapping.at[mapping_index, 'test_data'].append(new_test_data)
            
            if df_test.loc[test_index, 'func_nr'] is None:

                # visualize non-fitting datapoint in green
                plt.scatter(datapoint['x'], datapoint['y'], color = 'red')
            
            else:
                
                # visualize fitting datapoint in green
                plt.scatter(datapoint['x'], datapoint['y'], color = 'green')


    print("enhanced test dataset")
    print(df_test)

    
    # enhance and show plot 
    plt.xlabel('x')
    plt.xlabel('y')
    plt.legend()
    plt.show()


    '''

    for index, datapoint in df_test.iterrows():
        print('datapoint[x]')
        print(datapoint['x'])
        print('datapoint[y]')
        print(datapoint['y'])

    '''





    # OO Approach
    # class ideal function inherits from LinearRegression
    # class training data inherits from Dataframe?
    # class test data inherits from series?

if __name__ == '__main__':
    main()
import pwppa_classes as pwppa
import numpy as np
import pandas as pd
import sqlalchemy as sql
from sklearn.linear_model import LinearRegression
from sqlalchemy.orm import sessionmaker
from bokeh.plotting import figure, show
from bokeh.colors import RGB
from bokeh.models import Legend

def get_functions(df_x: pd.DataFrame, df_y: pd.DataFrame):
    '''
    Generates a list of linear regression functions for the input data as x/y-format.

    Parameters:
    - df_x (pd.DataFrame): The x-axis of the data to be fitted.
    - df_y (pd.DataFrame): One or more y-axes of the data to be fitted.

    Returns:
    - list: A list of the functions.

    Raises:
    - 
    '''

    functions = []
    x = df_x.values.reshape(-1, 1)
    i = 0

    for column in df_y.columns:

        lr = LinearRegression()

        y = df_y.iloc[:, i].values.reshape(-1, 1)

        function = lr.fit(x, y)
        functions.append(function)

        i += 1

    return functions

def get_deviations(df_x: pd.DataFrame, df_y: pd.DataFrame, functions: list[LinearRegression]):
    '''
    Uses a calculation function to find the least-square- and max-deviation of the x/y-input data to all the functions in the given list.
    The result is presented as a Dataframe with one row per given function.
   
    Parameters:
    - df_x (pd.DataFrame): The x-axis of the data to be checked for its deviations.
    - df_y (pd.DataFrame): The y-axis of the data to be checked for its deviations.
    - functions (list): All the functions that the deviations will be calculated from.

    Returns:
    - pd.Dataframe: A table with the following columns
                        function: Name of the function used for this row.
                        function_index: Index of the function in the list of functions.
                        sqr_deviation: Least-square-deviation of the given input data from the function of this row.
                        max_deviation: Largest single point deviation between input data and the function.
    Raises:
    - 
    '''

    df_deviations = pd.DataFrame(columns=['function', 'function_index', 'sqr_deviation', 'max_deviation'])

    # insert rows as lists of deviations
    # for each function, calculate the sum of all y-deviations squared with every function

    func_nr = 0

    # iterate 50 ideal functions
    for function in functions:

        row = {'function_index': func_nr}
        row['function'] = 'y' + str(func_nr + 1)
        row['sqr_deviation'], row['max_deviation'] = calc_deviations(df_x, df_y, function)

        func_nr += 1

        df_deviations.loc[len(df_deviations.index)] = row

    return df_deviations

def calc_deviations(x_data: pd.Series, y_data: pd.Series, function: LinearRegression):
    '''
    Calculates the least-square- and max-deviation of the x/y-input data to the given function.
   
    Parameters:
    - x_data (pd.Series): The x-axis of the data to be checked for its deviations.
    - y_data (pd.Series): The y-axis of the data to be checked for its deviations.
    - function (LinearRegression): The function that the deviations will be calculated from.

    Returns:
    - sqr_deviation: Least-square-deviation of the given input data from the function.
    - max_deviation: Largest single point deviation between input data and the function.

    Raises:
    - 
    '''
    y_column = np.array(y_data).reshape(-1, 1)
    y_estimated = function.predict(np.array(x_data).reshape(-1, 1))
    y_dev_abs = np.abs(y_column - y_estimated)

    # calculate square deviation
    sqr_deviation = np.float64(sum((y_dev_abs)**2))

    # find maximum single deviation
    max_deviation = np.float64(y_dev_abs.max())

    return sqr_deviation, max_deviation

def main():
    '''
    Carries out the steps of the program and calls helper functions.
    Step 1: It creates a database for data storage.
    Step 2: Out of the 50 functions given, find the four functions that that fit each set of y-values of the training data best. 
            The mapping criterion is the smallest sum of all y-deviations squared (Least-Square).
            The found functions are further called 'ideal functions'.
    Step 3: The next step is to check whether the test data fits into the deviation range of the training data.
            They are to be considered inside deviation range if
                the deviation of test datapoint to the ideal function does not exceed
                the largest deviation between training dataset and the ideal function chosen for it
                by more than factor sqrt(2).
            If the test datapoint fits into this criteria, the datapoint, corresponding function and deviation from it
            are to be added to the test database table.
    Step 4: It visualizes the data

    Raises:
    - 
    '''

    # Step 1
    # Database setup
    engine = sql.create_engine("mysql+pymysql://pwppa:zen@localhost:3306/pwppa_db")
    Session = sessionmaker(bind=engine)
    session = Session()

    # input tables
    # read CSV files into Dataframes
    df_train = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/train.csv")
    df_ideal = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/ideal.csv")
    df_test = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/test.csv")

    # write Dataframes to sql-Database
    df_train.to_sql("input_train", engine, if_exists='replace', index=True)
    df_ideal.to_sql("input_ideal", engine, if_exists='replace', index=True)
    df_test.to_sql("input_test", engine, if_exists='replace', index=True)

    # output tables
    # create all ORM-defined tables
    pwppa.Declarative_Base.metadata.create_all(engine)

    # Step 2
    # mapping between training datasets and ideal functions
    # get functions
    functions = get_functions(df_ideal.iloc[:, 0], df_ideal.iloc[:, 1:51])

    # Step 3
    # prepare visualization
    # create a new plot with a title and axis labels
    plt = figure(title='PWPPA Data Visualization', x_axis_label='x', y_axis_label='y')

    colors = {'y1': RGB(r=128, g=128, b=0), 'y2': RGB(r=0, g=128, b=0), 'y3': RGB(r=128, g=0, b=128), 'y4': RGB(r=0, g=128, b=128)}

    # convert x-values of training data to np.array() for further calculation steps
    x_values = np.array(df_train['x']).reshape(-1, 1)

    # loop through all the training datasets to map it to its ideal function
    for column in df_train.columns:

        # only for y-columns
        if column != 'x':

            # create plain mapping instance to work on
            train_ideal_mapping = pwppa.Train_Ideal_Mapping(training_data = column)

            # get all the deviations for current training dataset and sort ascending by sqr_deviation
            deviation_mapping = get_deviations(df_train['x'], df_train[column], functions).sort_values(by = 'sqr_deviation')

            # save deviation mapping to database
            deviation_mapping.to_sql("deviation_mapping_" + column, engine, if_exists='replace', index=True)

            # save deviations from the top row in database table to the train_ideal_mapping table
            # the top row is the one with the least square deviation
            ideal_row = deviation_mapping.iloc[0]
            train_ideal_mapping.function = ideal_row['function']
            train_ideal_mapping.function_index = ideal_row['function_index']
            train_ideal_mapping.sqr_deviation = ideal_row['sqr_deviation']
            train_ideal_mapping.max_deviation = ideal_row['max_deviation']

            # get train_ideal_mapping instance ready for commit to the database
            session.merge(train_ideal_mapping)

            # add training data to plot
            plt.scatter(
                df_train['x'],
                df_train[column],
                legend_label=column,
                size=2,
                color=colors[column].lighten(0.15),
                alpha=1
                )

            # store predicted y-values in numpy array
            y_values = functions[train_ideal_mapping.function_index].predict(x_values).reshape(-1)

            # add the function line to plot
            plt.line(
                df_train['x'],
                y_values,
                legend_label=train_ideal_mapping.function,
                line_width=2,
                line_color=colors[column].darken(0.1)
                )
            
            # add max deviation lines to plot
            plt.line(
                df_train['x'],
                y_values + train_ideal_mapping.max_deviation,
                legend_label=train_ideal_mapping.function,
                line_width=1, line_color=colors[column].darken(0.1),
                alpha=0.4
                )
            plt.line(
                df_train['x'],
                y_values - train_ideal_mapping.max_deviation,
                legend_label=train_ideal_mapping.function,
                line_width=1, line_color=colors[column].darken(0.1),
                alpha=0.4
                )
            
            # add shade to display accepted deviation range
            plt.varea(
                x = df_train['x'],
                y1 = y_values + (train_ideal_mapping.max_deviation * np.sqrt(2)),
                y2 = y_values - (train_ideal_mapping.max_deviation * np.sqrt(2)),
                fill_color = colors[column].darken(0.1),
                alpha=0.1
                )
                
    # save changes to the database
    session.commit()

    # second step
    # determine the largest deviation between training dataset and corresponding ideal function

    # check every test datapoint
    for test_index, test_datapoint in df_test.iterrows():

        # create plain Datapoint instance to work on
        datapoint = pwppa.Datapoint(x = test_datapoint['x'], y = test_datapoint['y'], function = None, deviation = None)
        datapoint_color = 'red'
        
        new_test_data = ''

        # check every ideal function
        for mapping in session.query(pwppa.Train_Ideal_Mapping).all():

            # calculate functions estimated y value
            function_y = np.float64(functions[mapping.function_index].predict(np.array(datapoint.x).reshape(-1, 1)))

            # calculate deviation from estimated y-value to test datapoint
            deviation = np.float32(np.abs(datapoint.y - function_y))

            # check if datapoint is in deviation range and has the smallest deviation from any function so far
            if deviation <= (mapping.max_deviation * np.sqrt(2)) and (datapoint.deviation is None or deviation < datapoint.deviation):

                # add ideal function and deviation to test_output database table
                datapoint.function = mapping.function
                datapoint.deviation = deviation

                # add found test datapoint and corresponding deviation to the ideal function mapping table
                new_test_data = new_test_data + 'datapoint_nr: ' + str(test_index) + ', x: ' + str(datapoint.x) + ', y: ' + str(datapoint.y) + ', deviation: ' + str(deviation)[0:] + '; '
                mapping.test_data = new_test_data

                # store color of corresponding training data for later use
                datapoint_color = colors[mapping.training_data]
        
        # plot datapoint with correct color
        if datapoint_color == 'red':
            plt.scatter(datapoint.x, datapoint.y, color=datapoint_color, legend_label='test data\nnot in range', size=4)
        else:
            plt.scatter(datapoint.x, datapoint.y, color=datapoint_color, size=4)
        
        # get datapoint instance ready for commit to the database
        session.merge(datapoint)

    # save changes to the database
    session.commit()

    # close the session
    session.close()

    # display bokeh plot
    show(plt)
    
if __name__ == '__main__':
    main()
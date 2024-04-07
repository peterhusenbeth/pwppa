import unittest as ut
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import main

class CalculationsTestCase(ut.TestCase):

    def test_get_functions(self):
        
        # create example input data
        x_data = {'x': [1, 2, 3, 4]}
        y_data = {'y1': [5, 4, 3, 2], 'y2': [1, 2, 3, 2]}
        
        func_1 = LinearRegression().fit(np.array(x_data['x']).reshape(-1, 1), np.array(y_data['y1']).reshape(-1))
        func_2 = LinearRegression().fit(np.array(x_data['x']).reshape(-1, 1), np.array(y_data['y2']).reshape(-1))

        df_x = pd.DataFrame(data = x_data)
        df_y = pd.DataFrame(data = y_data)

        # determine expected result
        expected_result = [func_1, func_2]
        
        # determine actual result
        result = main.get_functions(df_x, df_y)

        # assertions
        self.assertEqual(result.__len__(), 2, 'The number of objects in the list should be 2 but it is {}'.format(result.__len__()))
        self.assertEqual(type(result[0]), type(expected_result[0]), 'The type of objects in the list should be LinearRegression but at index 0 the type is {}'.format(type(result[0])))

    def test_get_deviations(self):

        # create example input data
        x_data = {'x': [1, 2, 3, 4]}
        func_y_data = {'y1': [5, 4, 3, 2], 'y2': [1, 2, 3, 4]}
        df_y_data = {'y1': [7, 6, 4, 2]}
        
        func_1 = LinearRegression().fit(np.array(x_data['x']).reshape(-1, 1), np.array(func_y_data['y1']).reshape(-1))
        func_2 = LinearRegression().fit(np.array(x_data['x']).reshape(-1, 1), np.array(func_y_data['y2']).reshape(-1))
        func_list = [func_1, func_2]

        df_x = pd.DataFrame(data = x_data)
        df_y = pd.DataFrame(data = df_y_data)

        # determine expected result
        expected_result = pd.DataFrame(data = {
                                       'function': ['y1', 'y2'],
                                       'function_index': [0, 1],
                                       'sqr_deviation': [9.0, 57.0],
                                       'max_deviation': [2.0, 6.0]
                                       })
        
        # determine acutal result
        result = main.get_deviations(df_x, df_y, func_list)

        print('result')
        print(result)
        print('expected_result')
        print(expected_result)

        # assertions
        for i in range(1):
            self.assertEqual(result.at[i, 'function'], expected_result.at[i, 'function'], 'The function in row {} is not set correctly. It should be {}'.format(i + 1, expected_result.at[i, 'function']))
            self.assertEqual(result.at[i, 'function_index'], expected_result.at[i, 'function_index'], 'The function_index in row {} is not set correctly. It should be {}'.format(i + 1, expected_result.at[i, 'function_index']))
            self.assertEqual(result.at[i, 'sqr_deviation'], expected_result.at[i, 'sqr_deviation'], 'The sqr_deviation in row {} is not set correctly. It should be {}'.format(i + 1, expected_result.at[i, 'sqr_deviation']))
            self.assertEqual(result.at[i, 'max_deviation'], expected_result.at[i, 'max_deviation'], 'The max_deviation in row {} is not set correctly: It should be {}'.format(i + 1, expected_result.at[i, 'max_deviation']))
            pass

    def test_calc_deviations(self):

        # create example input data
        x_data = {'x': [1., 2., 3., 4.]}
        func_y_data = {'y1': [5., 4., 3., 2.]}
        y_data = {'y1': [7., 6., 4., 2.]}
        
        func = LinearRegression().fit(np.array(x_data['x']).reshape(-1, 1), np.array(func_y_data['y1']).reshape(-1))

        df_x = pd.Series(data = x_data)
        df_y = pd.Series(data = y_data)

        # determine expected result
        expected_sqr, expected_max = 9.0, 2.0

        # determine actual result
        result_sqr, result_max = main.calc_deviations(df_x['x'], df_y['y1'], func)

        # print(func.predict(np.array(x_data['x']).reshape(-1, 1)))
        # print('expected_sqr: {}, result_sqr: {}'.format(expected_sqr, result_sqr))
        # print('expected_max: {}, result_max: {}'.format(expected_max, result_max))

        # assertions
        self.assertEqual(expected_sqr, result_sqr, 'The square deviation is {} but it should be {}.'.format(result_sqr, expected_sqr))
        self.assertEqual(expected_max, result_max, 'The maximum deviation is {} but it should be {}.'.format(result_max, expected_max))
        
if __name__ == '__main__':
    ut.main()
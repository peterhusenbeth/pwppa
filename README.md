# pwppa
#### Programming with Python - Practical Assignment

This repository contains the code for the practical assignemnt task of the IU Course "Programming with Python"

### Given Input

The following data inputs are given:  
1. Training Dataset: train.csv  
The training dataset consists of one x-column and 4 y-columns and contains scattered data.
2. Ideal Functions Dataset: ideal.csv  
The ideal functions dataset consists of one x-column and 50 y-columns and represents 50 mathematical functions.
3. Test Dataset: test.csv  
The test dataset consists of one x-column and one y-column and holds random (observed) data.

### Task Definition

The task is to write Python program executing the following tasks:  
1. Out of the 50 functions given, find the four functions that that fit each set of y-values of the training data best. The mapping criterion is the smallest sum of all y-deviations squared (Least-Square). The found functions are further called 'ideal functions'.
2. The next step is to check whether the test data fits into the deviation range of the training data. They are to be considered inside deviation range if the deviation of test datapoint to the ideal function does not exceed the largest deviation between training dataset and the ideal function chosen for it by more than factor sqrt(2). If the test datapoint fits into this criteria, the datapoint, corresponding function and deviation from it are to be added to the test database table.

So the expected output is a table of the following four columns:  
1. x test value
2. y test value
3. y-deviation
4. function no.

### Additional Requirements

On top of that there are some additional requirements for the program as a whole:  
1. It uses the Python library 'sqlalchemy' to create database tables for storing the data.
2. It visualizes the training data, the chosen ideal functions, the test data as well as the output data under an appropriately chosen representation of the deviation via the Python library 'Bokeh'.
3. It is designed sensibly object oriented and includes at least one inheritance.
4. It includes standard und user-defined exception handlings.
5. Where possible, it implements suitable unit tests.
6. It makes use of the Python library 'Pandas'.
7. The code is documented in its entirety, including docstrings.
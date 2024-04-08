import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, String, Double, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

# get the SQLAlchemy ORM base class for mapping instances to table entries
Declarative_Base = declarative_base()

class Train_Ideal_Mapping(Declarative_Base):
    '''
    Object-Relational-Mapper-Class to map the functions to the training data
    '''
    __tablename__ = 'output_train_ideal_mapping'

    training_data = Column(String(3), primary_key=True)
    function = Column(String(3), primary_key=True)
    sqr_deviation = Column(Double)
    max_deviation = Column(Double)
    test_data = Column(String(10000))

class Datapoint(Declarative_Base):
    '''
    Object-Relational-Mapper-Class to output the found functions and deviation form them for all the input test datapoints.
    '''
    __tablename__ = 'output_test'

    x = Column(Double, primary_key=True)
    y = Column(Double, primary_key=True)
    function = Column(String(3))
    deviation = Column(Double)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sqlalchemy import create_engine, Column, Integer, String, Double, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Function(Base):
    __tablename__ = 'functions'

    id = Column(Integer, primary_key=True)
    name = Column(String(3))

class Training_Data(Base):
    __tablename__ = 'training_data'

    id = Column(Integer, primary_key=True)
    name = Column(String(3))

    def find_ideal_function(functions):
        return 0

class Mapping(Base):
    __tablename__ = 'train_ideal_mapping'

    id = Column(Integer, primary_key=True)
    training_data_id = Column(Integer, ForeignKey('training_data.id'))
    function_id = Column(Integer, ForeignKey('functions.id'))
    sqr_deviation = Column(Double)
    max_deviation = Column(Double)

class Datapoint():

    def __init__(self, x: np.float64, y: np.float64, number: int) -> None:
        
        self.x = x

        self.y = y

        self.number = number
import pwppa_classes as pwppa
import pandas as pd
import sqlalchemy as sql
from sklearn.linear_model import LinearRegression
from sqlalchemy.orm import sessionmaker

def list_in_db(orm_class, dataframe: pd.DataFrame, session: sessionmaker):

    i = 0

    for column in dataframe.columns:

        if column != 'x':
            
            i += 1

            function = orm_class(id=i, name=column)

            session.add(function)

def main():

    # Database setup
    engine = sql.create_engine("mysql+pymysql://pwppa:zen@localhost:3306/pwppa_db", echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()

    # create tables and fill them

    # input data
    # read CSV files into Dataframes
    df_train = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/train.csv")
    df_ideal = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/ideal.csv")
    df_test = pd.read_csv(filepath_or_buffer="/Users/pete/Documents/python/pwppa/Datasets1/test.csv")

    # write Dataframes to sql-Database
    df_train.to_sql("train", engine, if_exists='replace', index=True)
    df_ideal.to_sql("ideal", engine, if_exists='replace', index=True)
    df_test.to_sql("test", engine, if_exists='replace', index=True)

    # program tables
    # create all ORM-defined tables
    pwppa.Base.metadata.create_all(engine)
    
    # list of ideal functions
    list_in_db(orm_class=pwppa.Function, dataframe=df_ideal, session=session)
    
    # list of training datasets
    list_in_db(orm_class=pwppa.Training_Data, dataframe=df_train, session=session)

    # mapping between training datasets and ideal functions
    
    
    session.commit()

    session.close()

    
if __name__ == '__main__':
    main()
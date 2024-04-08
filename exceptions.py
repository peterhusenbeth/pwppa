from pandas import Series

class NonNumericError(TypeError):

    def __init__(self, index: int, column: str, datapoint: Series) -> None:
        super().__init__('Test datapoint at index {} has non-numeric {}-value. x: {} y: {}'.format(index, column, datapoint.iloc[0], datapoint.iloc[1]))

        self.index = index

        self.column = column

        self.datapoint = datapoint

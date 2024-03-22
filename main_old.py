train_file = open("./Datasets1/train.csv", "r")
ideal_file = open("./Datasets1/ideal.csv", "r")

def csv_to_list(file, separator, convert_func):
    next(file)
    list = []

    for line in file:
        row = line[:len(line)-1].split(separator)
        converted_row = []

        for value in row:
            converted_row.append(convert_func(value))

        list.append(converted_row)
    
    return list


train_data = csv_to_list(train_file, ",", float)
ideal_data = csv_to_list(ideal_file, ",", float)

print(train_data[33][2])
print(ideal_data[33][2])



#for item in train_data:
#    print(f"{item!r}")
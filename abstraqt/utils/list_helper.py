def filter_duplicates(input_list):
    unique_list = []
    [unique_list.append(item) for item in input_list if item not in unique_list]
    return unique_list

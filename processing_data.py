import csv


def detect_categorical_and_numerical(data, target_column):
    categorical_columns = set()
    numerical_columns = set()

    for row in data:
        for column, value in row.items():
            if column == target_column or value in (None, ''):
                continue

            try:
                float(value)
                numerical_columns.add(column)
            except ValueError:
                categorical_columns.add(column)

    return list(categorical_columns), list(numerical_columns)


def label_encode_column(data, column_name):
    label_mapping = {value: index for index, value in
                     enumerate(sorted(set(row[column_name] for row in data if row[column_name] != '')))}

    for row in data:
        if row[column_name] in label_mapping:
            row[column_name] = label_mapping[row[column_name]]
        else:
            row[column_name] = None

    return data, label_mapping


def one_hot_encode(data, columns_to_encode):
    new_data = []
    one_hot_mappings = {}

    for row in data:
        new_row = row.copy()
        for column in columns_to_encode:
            if column == 'Name':
                continue

            value = row[column]
            if column not in one_hot_mappings:
                one_hot_mappings[column] = sorted(set(r[column] for r in data if r[column] is not None))

            for unique_value in one_hot_mappings[column]:
                new_column_name = f"{column}_{unique_value}"
                new_row[new_column_name] = 1 if value == unique_value else 0

            del new_row[column]

        new_data.append(new_row)

    return new_data



def normalize_data(data, numerical_columns):
    for column in numerical_columns:
        column_values = [float(row[column]) for row in data if row.get(column) not in (None, '')]

        if not column_values:
            continue

        min_value = min(column_values)
        max_value = max(column_values)

        if min_value == max_value:
            continue

        for row in data:
            if row.get(column) not in (None, ''):
                row[column] = (float(row[column]) - min_value) / (max_value - min_value)

    return data


def preprocess_data(input_filename, output_filename, target_column):
    with open(input_filename, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        data = [row for row in reader]

    categorical_columns, numerical_columns = detect_categorical_and_numerical(data, target_column)

    data, label_mapping = label_encode_column(data, 'Name')

    data = one_hot_encode(data, [col for col in categorical_columns if col != 'Name'])

    data = normalize_data(data, numerical_columns)

    with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print("Преобразованные данные сохранены в файл", output_filename)

preprocess_data('tanks_data.tsv', 'tanks_data.csv', target_column='Type')

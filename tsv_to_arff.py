import csv

def tsv_to_arff(tsv_filename, arff_filename, relation_name="tanks_data"):
    with open(tsv_filename, mode='r', newline='', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        headers = next(reader)

        attribute_types = {}
        data_rows = []

        for row in reader:
            data_rows.append(row)
            for index, value in enumerate(row):
                attribute_name = headers[index]

                if value == '':
                    continue

                if value.replace('.', '').isdigit():
                    attribute_types[attribute_name] = "NUMERIC"
                elif value in {"True", "False"}:
                    attribute_types[attribute_name] = "{True, False}"
                else:
                    attribute_types[attribute_name] = "STRING"

    with open(arff_filename, mode='w', newline='', encoding='utf-8') as arff_file:
        arff_file.write(f"@relation {relation_name}\n\n")

        for attribute_name, attribute_type in attribute_types.items():
            arff_file.write(f"@attribute {attribute_name} {attribute_type}\n")
        arff_file.write("\n@data\n")

        for row in data_rows:
            arff_file.write(','.join([value if value != '' else '?' for value in row]) + "\n")


tsv_filename = 'tanks_data.tsv'
arff_filename = 'tanks_data.arff'

tsv_to_arff(tsv_filename, arff_filename)

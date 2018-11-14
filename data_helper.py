import numpy as np
import csv
import json


def get_x_y_labels_from_csv(csv_file_name):
    x_raw = list()
    y_raw = list()
    labels = set()
    json_data = list()
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if (0 < len(row[0].split(' ')) <= 5) and (6 < len(row[1].split(' ')) < 99999):
                x_raw.append(row[1])
                y_raw.append(row[0])
                data = dict()
                data['type'] = row[0]
                data['content'] = row[1]
                json_data.append(data)
                labels.add(row[0])

    with open('./data.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)
    labels = sorted(list(labels))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    for pos in range(len(y_raw)):
        y_raw[pos] = label_dict[y_raw[pos]]
    return x_raw, y_raw, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

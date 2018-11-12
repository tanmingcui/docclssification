import csv


class ReadCSV(object):
    def __init__(self, doc_path=""):
        self.doc_path = doc_path
        self.stat_dict = dict()

    def read_file(self):
        with open(self.doc_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if row[0] not in self.stat_dict:
                    self.stat_dict[row[0]] = [row[1]]
                else:
                    self.stat_dict[row[0]].append(row[1])
        return self.stat_dict




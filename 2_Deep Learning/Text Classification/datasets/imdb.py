import pandas as pd


class IMDB:
    def __init__(self, train):
        data = pd.read_csv('~/Documents/3_dataset/IMDB Dataset.csv')
        train_set_length = int(len(data) * 0.8)
        if train:
            self.data = data.iloc[:train_set_length]
        else:
            self.data = data.iloc[train_set_length:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index < len(self.data)
        text = self.data.iloc[index, 0]
        label = 1 if self.data.iloc[index, 1] == 'positive' else 0
        return text, label

import argparse
import json
from torch.utils.data import Dataset, DataLoader
from saver_and_loader import save_weights, load_model
import torch
import numpy as np


class LabelLimitedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


def read_data(labels):
    data = []
    for label in labels:
        with open(f'data/json/train/{label}') as f:
            features = json.load(f)['data']
            targets = [label] * len(features)
            data.extend(list(zip(features, targets)))
    return np.array([sample[0] for sample in data], dtype=np.float32), np.array([sample[1] for sample in data])


def train(model, loss_fn, optimizer, data_loader):
    for x, y in data_loader:
        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_number', type=int)
    args = parser.parse_args()
    process_number = args.process_number
    labels = [process_number % 10, (process_number + 1) % 10, (process_number + 2) % 10]
    X, y = read_data(labels)
    dataset = LabelLimitedDataset(X, y)
    data_loader = DataLoader(dataset, shuffle=True)
    model = load_model('model_weights/shared_weights')
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, loss_fn, optimizer, data_loader)
    save_weights(model, f'model_weights/weights{process_number}')



if __name__ == '__main__':
    main()
import argparse
import json
from torch.utils.data import Dataset, DataLoader
from model import DistributedModel
import torch


class LabelLimitedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def read_data(labels):
    data = []
    for label in labels:
        with open(f'data/json/train/{label}') as f:
            features = json.load(f)['data']
            targets = [label] * len(features)
            data.extend(list(zip(features, targets)))
    return data

def get_model():
    model = DistributedModel()
    with open('model_weights/shared_weights.txt') as f:
        loaded_state_dict = json.load(f)
    model.load_state_dict({param_tensor: torch.tensor(values) for param_tensor, values in loaded_state_dict.items()})
    return model

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
    data = read_data(labels)
    dataset = LabelLimitedDataset(data)
    data_loader = DataLoader(dataset)
    model = get_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, loss_fn, optimizer, data_loader)








if __name__ == '__main__':
    main()
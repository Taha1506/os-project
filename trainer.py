import argparse
import json
from torch.utils.data import Dataset, DataLoader
from model import DistributedModel
from weight_saver import WeightSaver
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import torch
import numpy as np


class LabelLimitedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        sample_tensor = torch.tensor(sample)
        return sample_tensor, label

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
    X, y = read_data(labels)
    dataset = LabelLimitedDataset(X, y)
    data_loader = DataLoader(dataset)
    model = get_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, loss_fn, optimizer, data_loader)
    # saver = WeightSaver(model)
    # saver.save_calculated_weights()



if __name__ == '__main__':
    main()
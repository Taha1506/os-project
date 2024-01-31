from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import json
import os


def save_dataset(dataset, path):
    data_list = [[] for _ in range(10)]
    for data, label in dataset:
        data_list[label].append(data.tolist())
    for i in range(10):
        json_dic = {
            "data": data_list[i],
        }
        directory = os.path.dirname(path + f"/{i}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path + f"/{i}", 'w') as f:
            json.dump(json_dic, f)


def main():
    print("DOWNLOAD STARTED!")
    train_data = CIFAR10(root='./data/train', train=True, transform=ToTensor(), download=True)
    test_data = CIFAR10(root='./data/test', train=True, transform=ToTensor(), download=True)
    print("DOWNLOAD FINISHED!")
    print("DISTRIBUTING DATA STARTED!")
    save_dataset(train_data, path='./data/json/train')
    save_dataset(test_data, path='./data/json/test')
    print("DISTRIBUTING DATA ENDED!")


if __name__ == '__main__':
    main()
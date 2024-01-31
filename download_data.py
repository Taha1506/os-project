from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import json


def save_dataset(dataset, path):
    data_list = []
    label_list = []
    for data, label in dataset:
        data_list.append(data.tolist())
        label_list.append(label)
    json_dic = {
        "data": data_list,
        "label": label_list
    }
    with open(path, 'w') as f:
        json.dump(json_dic, f)


def main():

    train_data = CIFAR10(root='./data/train', train=True, transform=ToTensor(), download=True)
    test_data = CIFAR10(root='./data/test', train=True, transform=ToTensor(), download=True)
    save_dataset(train_data, path='./data/json/train')
    save_dataset(test_data, path='./data/json/test')


if __name__ == '__main__':
    main()
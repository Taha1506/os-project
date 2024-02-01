from saver_and_loader import load_model
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def evaluate_accuracy(model, data_loader):
    true_counts = 0
    total_counts = 0
    model.eval()
    for x, y in data_loader:
        predictions = model(x).argmax(axis=1)
        true_counts += int((predictions == y).sum())
        total_counts += len(y)
    return true_counts / total_counts


def main():
    model = load_model('model_weights/shared_weights')
    train_data = CIFAR10(root='./data/test', train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_data)
    train_accuracy = evaluate_accuracy(model, train_loader)
    print(f"Accuracy on train data is equal to: {train_accuracy}")
    test_data = CIFAR10(root='./data/test', train=False, transform=ToTensor(), download=True)
    test_loader = DataLoader(test_data)
    test_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Accuracy on test data is equal to: {test_accuracy}")


if __name__ == '__main__':
    main()
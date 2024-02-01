from model import DistributedModel
from saver_and_loader import save_weights


def main():
    model = DistributedModel()
    save_weights(model, 'model_weights/shared_weights')


if __name__ == '__main__':
    main()

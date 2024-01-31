import os
import json
from model import DistributedModel


def main():
    model = DistributedModel()
    state_dict = {param_tensor: model.state_dict()[param_tensor].numpy().tolist() for param_tensor in
                  model.state_dict()}
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    with open('model_weights/shared_weights.txt', 'w') as f:
        json.dump(state_dict, f)


if __name__ == '__main__':
    main()

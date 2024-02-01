from model import DistributedModel
import os
import json
import torch


def save_weights(model, path):
    state_dict = {param_tensor: model.state_dict()[param_tensor].numpy().tolist() for param_tensor in
                  model.state_dict()}
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    with open(path, 'w') as f:   # what should the name of the file be?
        json.dump(state_dict, f)


def load_model(path):
    model = DistributedModel()
    with open(path) as f:
        loaded_state_dict = json.load(f)
    model.load_state_dict({param_tensor: torch.tensor(values) for param_tensor, values in loaded_state_dict.items()})
    return model

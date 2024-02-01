from model import DistributedModel
import os
import json

class WeightSaver:

    def __init__(self, model):
        self.model = model

    def save_calculated_weights(self):
        state_dict = {param_tensor: self.model.state_dict()[param_tensor].numpy().tolist() for param_tensor in
                      self.model.state_dict()}
        if not os.path.exists('model_weights'):
            os.makedirs('model_weights')
        with open('model_weights/1.txt', 'w') as f:   # what should the name of the file be?
            json.dump(state_dict, f)

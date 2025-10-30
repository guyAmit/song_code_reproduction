import torch
import torch.nn as nn


class FCModel(nn.Module):
    def __init__(self, input_shape, hidden_size, n_layers, output_shape):
        super(FCModel, self).__init__()
        self.input_layer = nn.Linear(input_shape, hidden_size)
        
        # Create hidden layers
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = torch.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
FCN_parameters = dict(input_shape=28*28,
                      hidden_size=1024,
                      n_layers=3,
                      output_shape=10)

# model = FCModel(**FCN_parameters)

# optimizer = optim.AdamW(model.parameters(), lr=0.LEARNING_RATE, weight_decay=WEIGHT_DECAY)
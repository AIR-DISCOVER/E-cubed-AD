from torch import nn

def create_mlp(input_dim, hidden_dims, output_dim, dropout=None):
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    layers.append(nn.ReLU())
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.extend([nn.ReLU(), nn.Dropout(p=dropout)]) if dropout is not None else layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def create_mlp(input_dim, custom_layer_dim, dropout=None):
    assert isinstance(custom_layer_dim, list), "custom_layer_dim should be a list of integers"
    layers = []
    dims = [input_dim] + custom_layer_dim
    layers.append(nn.ReLU())
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.extend([nn.ReLU(), nn.Dropout(p=dropout)]) if dropout is not None else layers.append(nn.ReLU())
    return nn.Sequential(*layers)
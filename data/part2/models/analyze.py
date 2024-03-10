import torch

model_dict = torch.load('cls-best.state', map_location=torch.device('cpu'))

state_dict = model_dict['model_state']

print('State dictionary keys:')
print(state_dict.keys())

print('State dictionary values:')
print(state_dict.values())

num_params = sum(p.numel() for p in state_dict.values())
print(f'Number of parameters: {num_params}')

import torch

if torch.cuda.is_available():
    device_spec = "cuda"
elif torch.backends.mps.is_available():
    device_spec = "mps"
else:
    device_spec = "cpu"

print(torch.cuda.is_available())
print(device_spec)
import torch
import platform

print(platform.python_version())

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())
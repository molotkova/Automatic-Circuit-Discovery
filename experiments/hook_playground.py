import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple 2-layer network
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)   # input -> hidden
        self.fc2 = nn.Linear(2, 1)   # hidden -> output

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

net = TinyNet()

# ---- Hook functions ----
def forward_hook(module, input, output):
    print(f"[FORWARD] {module}")
    print(f"   input: {input}")
    print(f"   output: {output}\n")

def backward_hook(module, grad_input, grad_output):
    print(f"[BACKWARD] {module.__class__.__name__}")
    print(f"   grad_input: {grad_input}")
    print(f"   grad_output: {grad_output}\n")

# ---- Register hooks on the first layer ----
net.fc1.register_forward_hook(forward_hook)
net.fc1.register_backward_hook(backward_hook)

# ---- Run a forward + backward pass ----
x = torch.randn(1, 3)   # batch size = 1, input dim = 3
print(x)
y = net(x)
loss = y.mean()
# loss.backward()
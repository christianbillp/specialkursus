import torch
import torchvision
import timeit
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#%% Continue from previous
learning_rate = 0.01
momentum = 0.5

continued_network = Net()
continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,
                                momentum=momentum)

saved_model = "model_1542883196.pth"
saved_optimized = "optimizer_1542883196.pth"
network_state_dict = torch.load(saved_model)
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load(saved_optimized)
continued_optimizer.load_state_dict(optimizer_state_dict)

#%% Classify 10000 images
n_runs = 10
n_images = 10000

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=10000, shuffle=True)

def predict():
    return [(data, target) for batch_idx, (data, target) in enumerate(test_loader)]

timing_dataset = predict()[0][0]


def timed():
    """10000 images"""
    continued_network(timing_dataset)

total_runtime = timeit.timeit(timed, number=n_runs) # 0.8672915558598788 seconds
#%% Compare evaluation
s_per_run = total_runtime / n_runs
s_per_classification = s_per_run / n_images
classifications_per_second = 1 / s_per_run * n_images

print("Inference took {} microseconds, {} usec per image".format(round(s_per_run*1000, 2), round(s_per_classification*1000, 2)))
print("Classification rate: {} images per second".format(round(classifications_per_second, 2)))
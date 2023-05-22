import pip

pip install syft

import torch
import syft as sy

# Hooking PyTorch to enable adding extra functionalities like Federated and Encrypted Learning
hook = sy.TorchHook(torch)

# Creating two Virtual Workers
device_1 = sy.VirtualWorker(hook, id="device_1")
device_2 = sy.VirtualWorker(hook, id="device_2")

# A Toy Dataset
data = torch.tensor([[1.,1],[0,1],[1,0],[0,0]], requires_grad=True)
target = torch.tensor([[1.],[1], [0], [0]], requires_grad=True)

# Distribute the data across the workers, this could be encrypted heart data from wearable devices
device_1_data = data[:2].send(device_1)
device_1_target = target[:2].send(device_1)

device_2_data = data[2:].send(device_2)
device_2_target = target[2:].send(device_2)

# Create the model, this could be any model suitable for analysing heart data
model = torch.nn.Linear(2,1)

# Training logic
def train(iterations=20):
    
    for iter in range(iterations):

        # send the model to the device
        device_1_model = model.copy().send(device_1)
        device_2_model = model.copy().send(device_2)

        # perform SGD on each device
        device_1_opt = torch.optim.SGD(params=device_1_model.parameters(),lr=0.1)
        device_2_opt = torch.optim.SGD(params=device_2_model.parameters(),lr=0.1)

        for i in range(10):

            # device 1 training
            device_1_opt.zero_grad()
            device_1_pred = device_1_model(device_1_data)
            device_1_loss = ((device_1_pred - device_1_target)**2).sum()
            device_1_loss.backward()

            device_1_opt.step()
            device_1_loss = device_1_loss.get().data

            # device 2 training
            device_2_opt.zero_grad()
            device_2_pred = device_2_model(device_2_data)
            device_2_loss = ((device_2_pred - device_2_target)**2).sum()
            device_2_loss.backward()

            device_2_opt.step()
            device_2_loss = device_2_loss.get().data

        # aggregate the models
        model.weight.data.set_(((device_1_model.weight.data.get() + device_2_model.weight.data.get()) / 2))
        model.bias.data.set_(((device_1_model.bias.data.get() + device_2_model.bias.data.get()) / 2))

        print('Iteration:', iter, 'Losses:', device_1_loss, device_2_loss)
        
train()

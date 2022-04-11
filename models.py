import torch
import random
import os

# Runs model on gpu if one is avalible.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FFN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Builds layers
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        # Loads old model if it can.
        if os.path.exists('model.ct'):
            model = torch.load('model.ct')
            load = (
                list(model[list(model.keys())[0]].size())[0] == hidden_size and
                list(model[list(model.keys())[-1]].size())[0] == output_size
            )

            if load:
                self.load_state_dict(model)
                print("Loaded model!")


    def forward(self, input_data):
        relu = self.input_layer(input_data).clamp(min=0)

        # Uses the hidden layer twice.
        for i in range(2):
            relu = self.hidden_layer(relu).clamp(min=0)

        return self.output_layer(relu)

def train_ffn(data, targets, hidden_size, output_size, progress, max_epoch=50000, lr=1e-4):
    # Shuffles the data.
    temp = list(zip(data, targets))
    random.shuffle(temp)
    data, targets = zip(*temp)

    # Turns data into tensors.
    x = torch.tensor(data,dtype=torch.float).to(device)
    y = torch.tensor(targets,dtype=torch.float).to(device)

    print(x.size(), y.size())

    # Training setup
    model = FFN(len(data[0]), hidden_size, output_size).to(device)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for t in range(max_epoch):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item()) if (t % 1000 == 0) else t

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress.emit(int((t/max_epoch) * 100))

    # Saves the file.
    torch.save(model.state_dict(), 'model.ct')
    print('Saved model!')
    return model

def predict_ffn(data, model):
    x = torch.tensor(data).to(device)
    pred = model(x)
    return (torch.argmax(pred).item(), torch.max(pred).item())
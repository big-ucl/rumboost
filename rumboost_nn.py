import torch

class rumboost_nn(torch.nn.Module):
    def __init__(self, n_parameters, rum_structure, dropout_rate=0.5):
        super(rumboost_nn, self).__init__()
        self.n_parameters = n_parameters
        self.rum_structure = rum_structure
        self.layers = torch.nn.ModuleList([rumboost_parameters(dropout_rate) for _ in range(n_parameters)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        for i in range(self.n_parameters):
            x[:,i] = self.layers[i].forward(x[:,i].view(-1,1))

        output = torch.stack([torch.sum(x[:,v[0]:v[1]], dim=1) for v in self.rum_structure], dim=1)

        return output

    def train(self, x, y, n_epochs=1000):
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            output = self.forward(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print('Epoch: {0}, Loss: {1}'.format(epoch, loss.item()))

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            output = self.forward(x)
            return torch.softmax(output, dim=1)

class rumboost_parameters(torch.nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(rumboost_parameters, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(1, 1024)
        self.fc2 = torch.nn.Linear(1024, 2048)
        self.fc3 = torch.nn.Linear(2048, 1024)
        self.fc4 = torch.nn.Linear(1024, 1)
        self.relu = torch.nn.ReLU()

        self.batch_norm1 = torch.nn.BatchNorm1d(1024)
        self.batch_norm2 = torch.nn.BatchNorm1d(2048)
        self.batch_norm3 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x

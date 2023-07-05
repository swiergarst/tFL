import torch
import torch.nn as nn







class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()

        self.conv_layers = nn.Sequential(
        #TODO: check if stride and padding are correct (original fedAvg paper)
        nn.Conv2d(1, 32, 5, stride = 1, padding = 0),
        nn.MaxPool2d(2, stride = None, padding = 0 ),

        nn.Conv2d(32,64, 5, stride = 1, padding = 0),
        nn.MaxPool2d(2, stride = None, padding = 0),
        )

        self.lin_layers = nn.Sequential(
        #TODO: figure out what input (and output) size has to be
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512,62),
        nn.Softmax(dim = 0)
        )

    def forward(self, X):
        #print(X.shape)
        X = self.conv_layers(X)
        X = X.view(X.shape[0], -1)
        return(self.lin_layers(X))
    

    #TODO: implement use of K
    def train(self, X_train, y_train, opt, crit, K):
        
        out = self.forward(X_train)

        #print(out.shape, y_train.shape)
        loss = crit(out, y_train)
        loss.backward()
        opt.step()
        return loss

    def test(self):
        return None



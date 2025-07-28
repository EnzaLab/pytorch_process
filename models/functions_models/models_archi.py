#File with all model architectures used 

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

####################################################
#                     - 1 & 2 - DNN                #
####################################################
class DNN(nn.Module):
        def __init__(self, input_size, hidden_size_list, output_size, device, p=0.1):
            super(DNN, self).__init__()
            self.layers = nn.ModuleList()
            self.dropout = nn.Dropout(p)
            self.layers.append(nn.Linear(input_size, hidden_size_list[0]))
            for i in range(len(hidden_size_list) - 1):
                self.layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
            self.output = nn.Linear(hidden_size_list[-1], output_size)
            self.tanh = nn.Tanh()
    
        def forward(self, x):
            for layer in self.layers:
                x = self.dropout(self.tanh(layer(x)))
            x = self.output(x)
            return x


####################################################
#                     - 3 - CNN                    #
####################################################
class ConvRegression(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ConvRegression, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size, 60, kernel_size=7, padding=2),
            nn.BatchNorm2d(60),
            nn.Tanh(),
            nn.MaxPool2d(2),           
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(60, 120, kernel_size=3, padding=1),
            nn.BatchNorm2d(120),
            nn.Tanh(),
            nn.MaxPool2d(2),          
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(120, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.Tanh(),
            nn.MaxPool2d(3),  
            nn.Dropout2d(dropout_rate),


        )

        self.regressor = nn.Sequential(
            nn.Flatten(),            
            nn.Linear(240, 100), 
            nn.Tanh(),
            nn.Linear(100, output_size)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
####################################################
#                     - 3 - CNN                    #
####################################################
class ConvRegression_pfts(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ConvRegression_pfts, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_size, 120, kernel_size=7, padding=2),
            nn.BatchNorm2d(120),
            nn.Tanh(),
            nn.MaxPool2d(2),           
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(120, 60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.Tanh(),
            nn.MaxPool2d(2),          
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(60, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.Tanh(),
            nn.MaxPool2d(3),  
            nn.Dropout2d(dropout_rate),


        )

        self.regressor = nn.Sequential(
            nn.Flatten(),            
            nn.Linear(30, 10), 
            nn.Tanh(),
            nn.Linear(10, output_size)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

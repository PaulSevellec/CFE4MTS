import torch.nn as nn
import torch
import numpy as np   

# Models from COSCI-gan paper : https://arxiv.org/pdf/2006.15265.pdf
# https://github.com/aliseyfi75/COSCI-GAN 

class PairwiseDiscriminator(nn.Module):
    def __init__(self, n_channels, alpha):
        super().__init__()
        self.n_channels = n_channels
        n_corr_values = n_channels * (n_channels - 1) // 2
        layers = []
        while np.log2(n_corr_values) > 1:
            layers.append(nn.Linear(n_corr_values, n_corr_values // 2))
            layers.append(nn.LeakyReLU(alpha))
            layers.append(nn.Dropout(0.3))
            n_corr_values = n_corr_values // 2
        layers.append(nn.Linear(n_corr_values, 1))
        layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers)
        
        self.pairwise_correlation = torch.corrcoef
        self.upper_triangle = lambda x: x[torch.triu(torch.ones(n_channels, n_channels), diagonal=1) == 1]

    def forward(self, x):     
        final_upper_trianle = []
        for i in range(x.shape[0]):
            pairwise_correlation = self.pairwise_correlation(x[i,:].transpose(0,1))
            upper_triangle = self.upper_triangle(pairwise_correlation)
            final_upper_trianle.append(upper_triangle)
        final_upper_trianle = torch.stack(final_upper_trianle)
        return self.classifier(final_upper_trianle)
    
class LSTMDiscriminator(nn.Module):
    """Discriminator with LSTM"""
    def __init__(self, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMDiscriminator, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, c):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), x.size(1))
        return out

    
class CLSTMDiscriminator(nn.Module):
    """Discriminator with LSTM"""
    def __init__(self, ts_dim, hidden_dim=256, num_layers=1):
        super(CLSTMDiscriminator, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, c):
        input = torch.cat((x,c),1)
        input = input.reshape(input.shape[0], 1, input.shape[1])
        out, _ = self.lstm(input)
        out = self.linear(out.view(input.size(0) * input.size(1), self.hidden_dim))
        out = out.view(input.size(0), input.size(1))
        return out

class Discriminator(nn.Module):
    def __init__(self, n_samples, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_samples, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)

    def forward(self, x, c):
        input = torch.cat((x,c),1)
        input = input.reshape(input.shape[0], 1, input.shape[1])
        out, _ = self.lstm(input)
        out = self.linear(out.view(input.size(0) * input.size(1), self.hidden_dim))
        out = out.view(input.size(0), self.ts_dim)
        return out + x
    
class BiLSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(BiLSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, ts_dim)

    def forward(self, x, c):
        #input = torch.cat((x,c),1)
        input = x
        input = input.reshape(input.shape[0], 1, input.shape[1])
        out, _ = self.lstm(input)
        out = self.linear(out.view(input.size(0) * input.size(1), 2*self.hidden_dim))
        out = out.view(input.size(0), self.ts_dim)
        return out + x
    
class CBiLSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=256, num_layers=1):
        super(CBiLSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, ts_dim)

    def forward(self, x, c):
        input = torch.cat((x,c),1)
        input = input.reshape(input.shape[0], 1, input.shape[1])
        out, _ = self.lstm(input)
        out = self.linear(out.view(input.size(0) * input.size(1), 2*self.hidden_dim))
        out = out.view(input.size(0), self.ts_dim)
        return out + x

class Generator(nn.Module):
    def __init__(self, noise_len, n_samples, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_len, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(512, n_samples)
        )

    def forward(self, x,c):
        input = torch.cat((x,c),1)
        output = self.model(input)
        return output + x
    
class ToyClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(ToyClassifier, self).__init__()

        self.dense1 = nn.Linear(input_dim, 256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(256, 128)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dense3 = nn.Linear(128, n_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = x.reshape(-1, x.shape[1]*x.shape[2])
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x


class MLPGenerator(nn.Module):
    def __init__(self, dim, dropout_rate = 0.0, n_var=1, hidden_activation='relu', output_activation=None,
                 name='Noiser', shrink=False,
                 **kwargs):
        super(MLPGenerator, self).__init__(**kwargs)
        self.shrink = shrink
        hidden_dim = 128*n_var

        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.tanh1 = nn.Tanh()#nn.ReLU()
        self.tanh1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.LazyLinear(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        #self.tanh2 = nn.Tanh()#nn.ReLU()
        self.tanh2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.LazyLinear(dim*n_var)
        self.tanh = nn.Tanh()

        if self.shrink:
            self.ST = nn.Softshrink(0.01)

        self.unflatten = nn.Unflatten(-1,(n_var,int(dim/n_var)))

    def forward(self, inputs, c):
        
        output = self.flatten(inputs)

        output = self.dense1(output)
        output = self.bn1(output)
        output = self.tanh1(output)
        output = self.dp1(output)
        
        output = self.dense2(output)
        output = self.bn2(output)
        output = self.tanh2(output)
        output = self.dp2(output)
        output = self.dense3(output)
        
        #if self.output_activation is not None:
        output = self.tanh(output)
        
        if self.shrink:
            output = self.ST(output)
            
        return output.reshape(inputs.shape)

class CMLPGenerator(nn.Module):
    def __init__(self, dim, dropout_rate = 0.0, n_var=1, hidden_activation='relu', output_activation=None,
                 name='Noiser', shrink=False,
                 **kwargs):
        super(CMLPGenerator, self).__init__(**kwargs)
        self.shrink = shrink
        hidden_dim = 128*n_var

        self.flatten = nn.Flatten()
        self.dense1 = nn.LazyLinear(hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.tanh1 = nn.Tanh()#nn.ReLU()
        self.tanh1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.LazyLinear(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        #self.tanh2 = nn.Tanh()#nn.ReLU()
        self.tanh2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.LazyLinear(dim*n_var)
        self.tanh = nn.Tanh()

        if self.shrink:
            self.ST = nn.Softshrink(0.01)

        self.unflatten = nn.Unflatten(-1,(n_var,int(dim/n_var)))

    def forward(self, inputs, c):
        
        output = self.flatten(inputs)
        
        output = torch.cat((output,c),1)

        output = self.dense1(output)
        output = self.bn1(output)
        output = self.tanh1(output)
        output = self.dp1(output)
        
        output = self.dense2(output)
        output = self.bn2(output)
        output = self.tanh2(output)
        output = self.dp2(output)
        output = self.dense3(output)
        
        #if self.output_activation is not None:
        output = self.tanh(output)
        
        if self.shrink:
            output = self.ST(output)
            
        return output.reshape(inputs.shape)

class StandardScaler():
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        #dims = list(range(values.ndim - 1))
        self.mean = torch.mean(values, dim=0)
        #print("Mean", self.mean.shape)
        self.std = torch.std(values, dim=0)
        #print("Std", self.std.shape)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        print("Fitting")
        self.fit(values)
        return self.transform(values)
    
    def inverse_transform(self, values):
        return values * self.std.numpy()+ self.mean.numpy()

# date: April 4, 20:00: to see how much kernel_size is effective.
# the last thing I changed: kernel_sizes from 3 to 9 and 5
# out_channels from 7 to 15 
# epoch 7 to 70
# added primitive as well and double sized the features

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from itertools import product
from torch import nn

from dataset.tig_based_dataset import (load_gyroid_sdf_dataset,
                                       load_primitive_sdf_dataset)


repeat = 3
num_coef_chuk = 5
axis_chuk = 50
coef=list(product(np.linspace(1,4, num_coef_chuk), repeat=repeat))
_, labels1 = load_gyroid_sdf_dataset(coef=coef, axis_chuncks=axis_chuk)


_, labels2 = load_primitive_sdf_dataset(coef=coef, axis_chuncks=axis_chuk)

features  = labels1.reshape(num_coef_chuk**repeat, axis_chuk, axis_chuk, axis_chuk)
print(features.shape)

features = torch.cat((features,
              labels2.reshape(num_coef_chuk**repeat, axis_chuk, axis_chuk, axis_chuk)))
print(features.shape)

labels = features
# from plotian import plot_features
# plot_features(features[0:40])


encoder = nn.Sequential(
    nn.Conv3d(in_channels=1, out_channels=15, kernel_size=(9,9,9), padding=4, padding_mode='circular'),
    nn.MaxPool3d(kernel_size=(2,2,2), stride=2),
    # nn.ReLU(),
    nn.Conv3d(in_channels=15, out_channels=1, kernel_size=(5,5,5), padding=2, padding_mode='circular'),
    nn.MaxPool3d(kernel_size=(2,2,2), stride=2),
    nn.Tanh(),
    # nn.BatchNorm3d(1)  ,
    nn.Flatten(start_dim=1),
    nn.Linear(1728,500),
    nn.Tanh(),
    nn.Linear(500,15),
    nn.Tanh(),
    nn.Linear(15,3),    
    nn.Tanh(),
)



decoder = nn.Sequential(
    nn.Linear(3,10),
    
    # nn.ReLU(True),
    nn.Linear(10,20),
    # nn.ReLU(True),
    nn.Tanh(),

    nn.Linear(20,270),
    nn.Tanh(),

    # nn.ReLU(True),
    nn.Unflatten(dim=1, unflattened_size=(10, 3, 3,3)),
                                   # [1, 1, 30, 3, 3]
    nn.ConvTranspose3d(10, 16, 3,  output_padding=0),
    # nn.BatchNorm2d(16),
    # nn.ReLU(True),
    nn.ConvTranspose3d(16, 8, 3, stride=2, padding=0, output_padding=1),
    # # nn.BatchNorm2d(8),
    # nn.ReLU(True),
    nn.ConvTranspose3d(8, 2, 3, stride=2, padding=1, output_padding=1),
    # nn.ReLU(True),
    nn.ConvTranspose3d(2, 1, 3, stride=2, padding=0, output_padding=1)

)




features = features.reshape(2*len(coef),1, 50,50,50).float()
# features = features.reshape(len(coef),1, 50,50,50).float()

inp = torch.tensor([[0.2, 0.8, 0.9]])
decoder(inp).shape

print('before training:',encoder(features))

def load_array(data_arrays, batch_size, is_train=True): #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)



batch_size = 10
data_iter = load_array((features, features), batch_size)


lr= 0.003
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)


def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    for epoch in range(70):
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            image_batch = image_batch.to('cpu')
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Evaluate loss
            loss = loss_fn(decoded_data, image_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            # Print batch loss
        print(f' epoch: {epoch}:  {np.mean(train_loss)}')
        

    return np.mean(train_loss)


loss_fn = torch.nn.MSELoss()

train_epoch(encoder=encoder, 
            decoder=decoder,  
            dataloader=data_iter, 
            loss_fn=loss_fn, 
            optimizer=optim)

#  save the decoder and encoder.
print("Training Finished!!")
# torch.save(decoder.state_dict(), "model/model.params")
# torch.save(decoder.state_dict(), "model/model.params.pt")
torch.save(decoder, "model/enitre_decoder")
torch.save(encoder, "model/enitre_encoder")

print('after training:',encoder(features))

np.savetxt('trained_latents.txt', 
           encoder(features).detach().numpy(), 
           delimiter=',')

from utils.plot import d3_plot
scalar_field = decoder(encoder(features[0:1])).reshape(50,50,50).detach().numpy()
d3_plot(scalar_field)
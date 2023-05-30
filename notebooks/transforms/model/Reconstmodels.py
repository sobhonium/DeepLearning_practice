import torch

# Note: the entire models here assume that you intend to work on 1*150*150
# images.

class ConvAutoencoder1(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder1, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 
                            kernel_size=(3,3), 
                            stride=1, padding=1),  # 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, kernel_size=(3,3), 
                            stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            # scale_factor in upsamling can be a floating point 
            # the formula for choosing that factor number is 
            # d_out = floor(d_in*scale_factor). (Read Upsamling notebook).
            # a value between 150/148 and 150/151 for scale factor
            torch.nn.Upsample(scale_factor=((150+151)/148) / 2, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
#             torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.Tanh()
        )
    def encoder_func(self, inp):
    	return self.encoder(inp)
    def decoder_func(self, inp):
    	return self.decoder(inp)
    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded
        
class ConvAutoencoder2(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder2, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 
                            kernel_size=(3,3), 
                            stride=1, padding=1),   
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 16, kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.decoder = torch.nn.Sequential(
              torch.nn.Conv2d(16, 16, 
                            kernel_size=(3,3), 
                            stride=1, padding=1),torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),  
            torch.nn.Upsample(scale_factor=2.15),torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1),  
            torch.nn.Upsample(scale_factor=(150+151)/(2*79)),
                torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  

            

            torch.nn.Tanh()
        )
    def encoder_func(self, inp):
    	return self.encoder(inp)
    def decoder_func(self, inp):
    	return self.decoder(inp)
    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded
        
class ConvAutoencoder3(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder3, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 
                            kernel_size=(3,3), 
                            stride=1, padding=1),   
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, 
                            kernel_size=(3,3), 
                            stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, 
                            kernel_size=(3,3), 
                            stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 32, 
                            kernel_size=(3,3), 
                            stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 8, 
                            kernel_size=(3,3), 
                            stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 1, 
                            kernel_size=(3,3), 
                            stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Sigmoid()

            )

        self.decoder =  torch.nn.Sequential(
           
            torch.nn.ConvTranspose2d(1, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(8, 15, kernel_size=3, stride=2, padding=1, ),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(15, 25, kernel_size=3, stride=2, padding=1,output_padding=1),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(25, 8, kernel_size=3, stride=2, padding=1, ),
            torch.nn.ReLU(True),
            
            torch.nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(True),
            
            torch.nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(True),



            torch.nn.Tanh()
        )
    def encoder_func(self, inp):
    	return self.encoder(inp)
    def decoder_func(self, inp):
    	return self.decoder(inp)
    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded

class ConvAutoencoder4(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder4, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 
                            kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(64, 16, kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2)  
        )

        self.decoder = torch.nn.Sequential(

            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ConvTranspose2d(16, 64, 3, stride=1, padding=1), 
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=(150+151)/(2*74), mode='nearest'),
            torch.nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1), 
            torch.nn.Tanh()
        )
        
    def encoder_func(self, inp):
    	return self.encoder(inp)
    def decoder_func(self, inp):
    	return self.decoder(inp)
    		    

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded
        

class ConvAutoencoder5(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder5, self).__init__()

        self.net = ConvAutoencoder4()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 
                            kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(64, 16, kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2) ,
                torch.nn.Conv2d(16, 16, kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
                torch.nn.Conv2d(16, 16, kernel_size=(3,3), 
                            stride=1, padding=1),  
            torch.nn.ReLU(True),

        )

        self.decoder = torch.nn.Sequential(
#             torch.nn.Upsample(scale_factor=2, mode='nearest'),
#             torch.nn.ConvTranspose2d(1, 4, 3, stride=1, padding=1), 
#             torch.nn.ReLU(True),
            
#             torch.nn.Upsample(scale_factor=2, mode='nearest'),
#             torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1), 
#             torch.nn.ReLU(True),
    
            torch.nn.Upsample(scale_factor=(37+38)/(2*18), mode='nearest'),
            torch.nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1), 
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ConvTranspose2d(16, 64, 3, stride=1, padding=1), 
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=(150+151)/(2*74),mode='nearest'),
            torch.nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1), 
            torch.nn.Tanh()
        )
        
    def encoder_func(self, inp):
    	return self.encoder(inp)
    def decoder_func(self, inp):
    	return self.decoder(inp)
    		    

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded        

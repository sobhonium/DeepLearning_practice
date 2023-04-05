import numpy as np
import torch 
from vedo import Text2D, Volume, show

def plot_features(features):
    '''recives images of shape x*y*y*y where x is the number of inputs
       y is the dim. here =50
    '''
    show_pairs = []
    for feat in features:
        # print(decoder(torch.tensor(code).reshape(1,-1).float()).shape)
        # sc = decoder(encoder(torch.tensor(code).reshape(1,-1).float()))  
        sc = feat.reshape(50,50,50).detach().numpy() 
      
        
      
        vol = Volume(sc)
      
        vol.add_scalarbar3d()
      

        lego = vol.legosurface(vmin=-20, vmax=0) # volume of sdf( g(x,y,z) ) > 0
 
        show_pairs = show_pairs + [(lego, str('code'))]

        print('preparing code shape#:', len(show_pairs), ',  code:', 'code')

    show(show_pairs, N=features.shape[0], axes=True)
    


def plotting(encoder, decoder, latent_codes, axis_chuncks = 15):
    # I will change this
    codes = latent_codes
    # print(codes.shape)

    if isinstance(codes,list):
        codes = np.array(codes) 
    if codes.ndim==1:
        print('====>dim 1')
        codes = codes.reshape(1, -1)
    show_pairs = []

    for code in codes:
        print(decoder(torch.tensor(code).reshape(1,-1).float()).shape)
        # sc = decoder(encoder(torch.tensor(code).reshape(1,-1).float())).reshape(50,50,50).detach().numpy()   
        sc = decoder(torch.tensor(code).reshape(1,-1).float()).reshape(50,50,50).detach().numpy()   
      
        
      
        vol = Volume(sc)
      
        vol.add_scalarbar3d()
      

        lego = vol.legosurface(vmin=-3, vmax=0) # volume of sdf( g(x,y,z) ) > 0
 
        show_pairs = show_pairs + [(lego, str(code))]

        print('preparing code shape#:', len(show_pairs), ',  code:', code)

    show(show_pairs, N=codes.shape[0], axes=True)
    

if __name__ == "__main__":


    # loading the trained model.
    decoder = torch.load("model/enitre_decoder")
    encoder = torch.load("model/enitre_encoder")
    decoder.eval()
    encoder.eval()
    
    
        # torch.save(10,'test.txt')
  
    from itertools import product
    latent_codes = list(product(np.linspace(-1.2, 1.2, 5), repeat=3))
    print('codes length:= ', len(latent_codes))

    # latent_codes = [[2.425022423267364502e-01,6.172305718064308167e-02,-1.831861436367034912e-01]]

    plotting(encoder=encoder, decoder=decoder, latent_codes=latent_codes)

from vedo import dataurl, Volume, show, io
from vedo.applications import IsosurfaceBrowser
import numpy as np
from numpy import sin, cos, pi


def d3_plot(scalar_field):
    '''meant to be called from outside of this file'''
    vol = Volume(scalar_field)
    # Generate the surface that contains all voxels in range [1,2]
    lego = vol.legosurface(-20,0).add_scalarbar()
    show_lego = [('A', lego)]
   
    show(show_lego, N=len(show_lego), axes=True)
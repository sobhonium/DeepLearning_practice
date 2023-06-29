#!/usr/bin/env python

#TODO(SBN): put files into data folder rather than absolute path addressing (done).
# how to use this:
# > ./show_mesh.py ./dataset/SPRING0053.obj

from vedo import *
import sys

def show(filename):
	mesh = Mesh(filename)
	mesh.show(interactive=True)


if '__main__' == __name__:
	
	try:	
		filename = sys.argv[1]
		print("showing ", sys.argv[1])
	except:
		filename = './dataset/SPRING0053.obj'
		print("could not file the filename!")
		
	finally:
		show(filename)
	    
	


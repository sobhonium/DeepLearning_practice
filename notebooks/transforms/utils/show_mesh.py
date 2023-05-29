#!/usr/bin/env python

#TODO(SBN): put files into data folder rather than absolute path addressing.

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
		filename = '/home/sobhan/Downloads/datasets/SPRING_FEMALE/mesh/SPRING0014.obj'
		print("could not file the filename!")
		
	finally:
		show(filename)
	    
	


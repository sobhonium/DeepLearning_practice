import numpy as np

def read_obj(filename):
    '''
     reads .obj files (which include vertices, nvertices and faces of on 3d shape)
     
    :param
           (str) filename: conceptnet json file
    :return: 
           (tuple) (vertices, nvertices, faces) of a give .obj file.
    
    source helper: 
            https://www.programcreek.com/python/?code=martinResearch%2FDEODR%2FDEODR-master%2Fdeodr%2Fobj.py
    '''

    faces = []
    vertices = []
    nvertices = []
    fid = open(filename, "r")
    node_counter = 0
    while True:

        line = fid.readline()
        if line == "":
            break
        while line.endswith("\\"):
            # Remove backslash and concatenate with next line
            line = line[:-1] + fid.readline()
        
        if line.startswith("vn"):
            coord = line.split()
            coord.pop(0)
            node_counter += 1
            nvertices.append(np.array([float(c) for c in coord]))
            
        
        elif line.startswith("v"):
            coord = line.split()
            coord.pop(0)
            node_counter += 1
            vertices.append(np.array([float(c) for c in coord]))

        elif line.startswith("f "):
            fields = line.split()
            fields.pop(0)

            # in some obj faces are defined as -70//-70 -69//-69 -62//-62
            cleaned_fields = []
            for f in fields:
                f = int(f.split("/")[0]) - 1
                if f < 0:
                    f = node_counter + f
                cleaned_fields.append(f)
            faces.append(np.array(cleaned_fields))

    faces = np.row_stack(faces)
    vertices = np.row_stack(vertices)
    nvertices = np.row_stack(nvertices)
    return vertices, nvertices,  faces 


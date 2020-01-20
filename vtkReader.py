import numpy as np
import scipy.linalg as sp
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN

def readVTK(filename = 'input.vtk'):
    path = './data/testfiles/' + filename
    reader = vtkStructuredPointsReader()
    reader.SetFileName(path)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()
    dim = data.GetDimensions()

    #import pdb; pdb.set_trace()

    dataset = VN.vtk_to_numpy(data.GetPointData().GetArray(0))
    dataset = dataset.reshape(dim, order='F')

    nonzero_element_coordinates = dataset.nonzero()
    x, y, z = nonzero_element_coordinates

    return dataset, x, y, z

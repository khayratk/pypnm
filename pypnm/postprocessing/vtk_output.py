import glob
import os

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
except ImportError:
    pass

import numpy as np


class VTKWriter(object):
    def write_vtp(self, polydata, filename=None):
        n = self.counter
        if filename is None:
            filename = self.dir_name + '/paraview' + str(n).zfill(8) + '.vtp'
        else:
            filename = self.dir_name + '/' + filename + '.vtp'

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)

        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)

        writer.Write()
        self.counter += 1


class VtkWriterNetwork(VTKWriter):
    def __init__(self, network, dir_name="paraview", delete_existing_files=False):
        self.network = network
        self.counter = 0
        self.dir_name = dir_name

        if not os.path.exists(dir_name):
            print "output folder does not exist... it will be created"
            os.makedirs(dir_name)

        if delete_existing_files:
            files = glob.glob(dir_name+"/*")
            for f in files:
                os.remove(f)

        self.output_pore_fields = None
        self.output_tube_fields = None
        self.polydata = self.polydata_from_network(self.network)

    def write(self, name=None):
        self.write_vtk_binary_file(name)
        if name is None:
            self.counter += 1

    def add_pore_field(self, array, array_name):
        if self.output_pore_fields is None:
            self.output_pore_fields = {}
        self.output_pore_fields[array_name] = array

    def add_tube_field(self, array, array_name):
        if self.output_tube_fields is None:
            self.output_tube_fields = {}
        self.output_tube_fields[array_name] = array

    def add_point_fields(self):
        network = self.network
        polydata = self.polydata

        if self.output_pore_fields is None:
            self.add_point_data_vtp(polydata, network.pores.p_n, "P_n")
            self.add_point_data_vtp(polydata, network.pores.bbone, "PoreBB")
            self.add_point_data_vtp(polydata, network.pores.p_n, "P_n")
            self.add_point_data_vtp(polydata, network.pores.p_w, "P_w")
            self.add_point_data_vtp(polydata, network.pores.vol, "PoreVol")
            self.add_point_data_vtp(polydata, network.pores.r, "PoreRadius")
            self.add_point_data_vtp(polydata, network.pores.invaded, "PoreStatus")
            self.add_point_data_vtp(polydata, network.pores.connected, "PoreConn")
            self.add_point_data_vtp(polydata, network.pores.sat, "PoreSat")
            self.add_point_data_vtp(polydata, network.pores.p_c, "P_c")
        else:
            for key in self.output_pore_fields:
                self.add_point_data_vtp(polydata, self.output_pore_fields[key], key)

    def add_tube_fields(self):
        network = self.network
        polydata = self.polydata
        if self.output_tube_fields is None:
            self.add_cell_data_vtp(polydata, network.tubes.invaded, "TubeStatus")
            self.add_cell_data_vtp(polydata, network.tubes.r, "TubeRadius")
            self.add_cell_data_vtp(polydata, network.tubes.bbone, "TubeBB")
            self.add_cell_data_vtp(polydata, network.tubes.connected, "TubeConn")
            self.add_cell_data_vtp(polydata, network.tubes.k_w, "cond_w")
        else:
            for key in self.output_tube_fields:
                self.add_cell_data_vtp(polydata, self.output_tube_fields[key], key)

    def write_vtk_binary_file(self, filename=None):
        network = self.network
        self.polydata = self.polydata_from_network(network)
        self.add_point_fields()
        self.add_tube_fields()
        self.write_vtp(self.polydata, filename)

    @staticmethod
    def polydata_from_network(network):
        assert np.all(network.edgelist[:, 0] > -1)
        assert np.all(network.edgelist[:, 1] > -1)

        edgelist = network.edgelist
        nr_t = len(edgelist)

        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        tubes = vtk.vtkCellArray()

        # Points
        coords = np.column_stack((network.pores.x, network.pores.y, network.pores.z))
        points.SetData(numpy_to_vtk(coords, deep=1))

        # Cells
        adj_vtk = np.column_stack((2 * np.ones(nr_t, dtype=np.int), np.array(edgelist[:, 0], dtype=np.int),
                                   np.array(edgelist[:, 1], dtype=np.int)))
        adj_vtk = adj_vtk.flatten()
        tubes.SetCells(nr_t, numpy_to_vtkIdTypeArray(adj_vtk, deep=1))

        polydata.SetPoints(points)
        polydata.SetLines(tubes)
        return polydata

    @staticmethod
    def add_cell_data_vtp(polydata, data, name):
        array = numpy_to_vtk(data, deep=1)
        array.SetName(name)
        polydata.GetCellData().AddArray(array)

    @staticmethod
    def add_point_data_vtp(polydata, data, name):
        array = numpy_to_vtk(data, deep=1)
        array.SetName(name)
        polydata.GetPointData().AddArray(array)


class VTKIgraph(VTKWriter):
    def __init__(self, graph, dir_name="vtk_igraph", delete_existing_files=False):
        self.graph = graph
        self.counter = 0
        self.dir_name = dir_name

        if not os.path.exists(dir_name):
            print "output folder does not exist... it will be created"
            os.makedirs(dir_name)

        if delete_existing_files:
            files = glob.glob(dir_name+"/*")
            for f in files:
                os.remove(f)

        self.polydata = self.polydata_from_igraph(graph)
        self.add_point_fields()
        self.add_edge_fields()

    @staticmethod
    def add_cell_data_vtp(polydata, data, name):
        array = numpy_to_vtk(data, deep=1)
        array.SetName(name)
        polydata.GetCellData().AddArray(array)

    @staticmethod
    def add_point_data_vtp(polydata, data, name):
        array = numpy_to_vtk(data, deep=1)
        array.SetName(name)
        polydata.GetPointData().AddArray(array)

    def add_point_fields(self):
        graph = self.graph
        polydata = self.polydata
        for attr in graph.vs.attributes():
            self.add_point_data_vtp(polydata, graph.vs[attr], attr)

    def add_edge_fields(self):
        graph = self.graph
        polydata = self.polydata
        for attr in graph.es.attributes():
            self.add_cell_data_vtp(polydata, graph.es[attr], attr)

    @staticmethod
    def polydata_from_igraph(graph):
        edgelist = graph.get_edgelist()
        edgelist = np.array(edgelist)

        nr_t = graph.ecount()

        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        tubes = vtk.vtkCellArray()

        # Points
        try:
            coords = np.column_stack((graph.vs["x"], graph.vs["y"], graph.vs["z"]))
        except KeyError:
            coords = np.column_stack((graph.vs["x"], graph.vs["y"], np.zeros(graph.vcount())))
        points.SetData(numpy_to_vtk(coords, deep=1))

        # Cells
        adj_vtk = np.column_stack((2 * np.ones(nr_t, dtype=np.int), np.array(edgelist[:, 0], dtype=np.int),
                                   np.array(edgelist[:, 1], dtype=np.int)))
        adj_vtk = adj_vtk.flatten()
        tubes.SetCells(nr_t, numpy_to_vtkIdTypeArray(adj_vtk, deep=1))

        polydata.SetPoints(points)
        polydata.SetLines(tubes)
        return polydata


    def write(self, filename=None):
        graph = self.graph
        self.polydata = self.polydata_from_igraph(graph)
        self.add_point_fields()
        self.add_edge_fields()
        self.write_vtp(self.polydata, filename)
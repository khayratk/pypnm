import igraph as ig

import numpy as np


class StatoilReader:
    def __init__(self, filename):
        """
        Reads a statoil network file. Removes the virtual inlet and outlet pores
        Parameters
        ----------
        filename
        """
        # Filenames of data
        nodefile1 = filename + "_node1" + ".dat"
        nodefile2 = filename + "_node2" + ".dat"
        linkfile1 = filename + "_link1" + ".dat"
        linkfile2 = filename + "_link2" + ".dat"

        self.nr_p = self.nr_pores(nodefile1)
        self.network_length, self.network_width, self.network_height = self.get_network_dimensions(nodefile1)
        self.nr_t = self.nr_tubes(linkfile1)
        self.edgelist = self.get_edgelist(linkfile1)
        self.pore_prop = self.get_pores_properties(nodefile1, nodefile2)
        self.tube_prop = self.get_tube_properties(linkfile1, linkfile2)

        assert np.all(self.pore_prop["G"] <= 1 / (4 * np.pi) * 1.001)
        assert np.all(self.tube_prop["G"] <= 1 / (4 * np.pi) * 1.001)

        # self.pore_G[self.pore_G > 1/(4*np.pi)] = 1/16.
        # self.tube_G[self.tube_G > 1/(4*np.pi)] = 1/16.

        self.edgelist -= 1  # Change pore indices such that -2 is outlet and -1 is inlet
        inlet_tubes_mask = ((self.edgelist[:, 0] == -1) | (self.edgelist[:, 1] == -1))
        outlet_tubes_mask = ((self.edgelist[:, 0] == -2) | (self.edgelist[:, 1] == -2))
        domain_tubes_mask = np.logical_not(inlet_tubes_mask | outlet_tubes_mask)

        self.pi_out = np.unique(np.max(self.edgelist[inlet_tubes_mask], axis=1))
        self.pi_in = np.unique(np.max(self.edgelist[outlet_tubes_mask], axis=1))

        self.pi_domain = np.setdiff1d(np.arange(self.nr_p), np.union1d(self.pi_in, self.pi_out))

        for key in self.tube_prop:
            self.tube_prop[key] = self.tube_prop[key][domain_tubes_mask]

        self.edgelist = self.edgelist[domain_tubes_mask, :]
        self.nr_t = np.sum(domain_tubes_mask)

        assert np.all(self.tube_prop["l_tot"] >= self.tube_prop["l"])

        assert (len(self.tube_prop["G"]) == self.nr_t)
        assert (self.nr_t == len(self.edgelist))

        G = ig.Graph(self.nr_p)
        G.add_edges(self.edgelist)
        self.pt_adj = [np.asarray(x) for x in G.get_inclist()]
        self.p_adj = [np.asarray(x) for x in G.get_adjlist()]

    def file_header(self, file):
        with open(file, 'r') as f:
            header = f.readline()
            header = header.rstrip('\n').split()
        return header

    def nr_pores(self, nodefile1):
        header = self.file_header(nodefile1)
        return int(header[0])

    def nr_tubes(self, linkfile1):
        header = self.file_header(linkfile1)
        return int(header[0])

    def get_network_dimensions(self, nodefile1):
        header = self.file_header(nodefile1)
        return float(header[1]), float(header[2]), float(header[3])

    def get_lines_of_file(self, filename):
        with open(filename, 'r') as f:
            lines = [line.rstrip('\n').split() for line in f]
        return lines

    def get_pores_properties(self, nodefile1, nodefile2):
        lines = self.get_lines_of_file(nodefile2)
        nr_p = len(lines)
        pore_prop = dict()
        index = np.loadtxt(nodefile1, skiprows=1, usecols=(0,))
        sort_index = np.argsort(index)
        pore_prop["x"], pore_prop["y"], pore_prop["z"] = np.loadtxt(nodefile1, skiprows=1,
                                                                    usecols=(1, 2, 3), unpack=True)
        pore_prop["vol"], pore_prop["r"], pore_prop["G"] = np.loadtxt(nodefile2, skiprows=0,
                                                                      usecols=(1, 2, 3), unpack=True)

        for key in pore_prop:
            pore_prop[key] = pore_prop[key][sort_index]

        assert len(pore_prop["vol"]) == nr_p
        assert len(pore_prop["x"]) == nr_p

        return pore_prop

    def get_tube_properties(self, linkfile1, linkfile2):
        nr_t = self.nr_tubes(linkfile1)
        tube_prop = dict()

        tube_prop["r"], tube_prop["G"], tube_prop["l_tot"] = np.loadtxt(linkfile1, skiprows=1,
                                                                        usecols=(3, 4, 5), unpack=True)

        tube_prop["l_p1"], tube_prop["l_p2"], tube_prop["l"] = np.loadtxt(linkfile2, skiprows=0,
                                                                          usecols=(3, 4, 5), unpack=True)

        tube_prop["vol"], tube_prop["vol_clay"] = np.loadtxt(linkfile2, skiprows=0, usecols=(6, 7), unpack=True)

        assert (nr_t == len(tube_prop["r"]))

        return tube_prop

    def get_edgelist(self, linkfile1):
        edgelist = np.loadtxt(linkfile1, skiprows=1, usecols=(1, 2)).astype(np.int32)
        return edgelist


if __name__ == "__main__":
    StatoilReader("c2/C2")

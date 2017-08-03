from pypnm.percolation import graph_algs
from pypnm.porenetwork.constants import *


class BackBoneComputer(object):
    def __init__(self, network):
        self.network = network
        self.mode = NEUTRAL

    def compute(self):
        graph_algs.update_pore_and_tube_backbone(self.network)
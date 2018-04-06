# PyPNM

PyPNM is a pore-network flow solver for quasi-static and dynamic simulations.


# Example


The following code runs a quasi-static drainage simulation in a structured network

```python

from pypnm.flow_simulation.quasi_static_simulation import QuasiStaticSimulation
from pypnm.porenetwork.network_factory import structured_network
import numpy as np

fluid_properties = {"gamma":1.0, "mu_n": 1.0, "mu_w": 1.0}
network = structured_network(30, 15, 15, media_type="consolidated", periodic=False)
simulation = QuasiStaticSimulation(network, fluid_properties)
simulation.apply_initial_conditions()

for n, sat in enumerate(np.linspace(0.1, 0.9, 81)):
    simulation.update_saturation_conn(sat)
    output_filename = "ip" + str(n).zfill(8)
    network.export_to_vtk(output_filename)


```

![Alt Text](http://media.giphy.com/media/3mJSs8JhtuqGyUEnTZ/giphy.gif)


The following code runs a dynamic drainage simulation in a structured network


```python

from pypnm.porenetwork.network_factory import structured_network
from pypnm.flow_simulation.dynamic_simulation import DynamicSimulation
from pypnm.porenetwork.constants import WEST, EAST
from pypnm.flow_simulation.simulation_bc import SimulationBoundaryCondition

network = structured_network(40, 20, 20, periodic=False)
network.set_zero_volume_all_tubes()
fluid_properties = {"gamma":1.0, "mu_n": 0.1, "mu_w": 1.0}

simulation = DynamicSimulation(network, fluid_properties, delta_pc=0.01)

bc = SimulationBoundaryCondition()
q_total = 1e-8

pi_inlet = network.pi_list_face[WEST]
pi_outlet = network.pi_list_face[EAST]

bc.set_nonwetting_source(pi_inlet, q_total/len(pi_inlet)*np.ones(len(pi_inlet)))
bc.set_pressure_outlet(pi_list=pi_outlet, p_wett=0.0)

simulation.set_boundary_conditions(bc)

delta_t = 0.01 * network.total_pore_vol/simulation.total_source_nonwett

for n in xrange(20):
    simulation.advance_in_time(delta_t=delta_t)

    output_filename = "dyn" + str(n).zfill(8)
    network.export_to_vtk(output_filename)
    print "Nonwetting saturation:", simulation.nonwetting_saturation()

```

![Alt Text](http://media.giphy.com/media/3mJSs8JhtuqGyUEnTZ/giphy.gif)
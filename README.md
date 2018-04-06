===============================================================================
PyPNM
===============================================================================

**PyPNM** is a pore-network flow solver for quasi-static and dynamic simulations.



===============================================================================
Example
===============================================================================

The following code runs a quasi-static drainage simulation in a structured network

.. code-block:: python

    from pypnm.flow_simulation.quasi_static_simulation import QuasiStaticSimulation
    from pypnm.porenetwork.network_factory import structured_network
    import numpy as np

    fluid_properties = {"gamma":1.0, "mu_n": 1.0, "mu_w": 1.0}
    network = structured_network(30, 15, 15, media_type="consolidated", periodic=False)
    simulation = QuasiStaticSimulation(network, fluid_properties)
    simulation.apply_initial_conditions()

    for n, sat in enumerate(np.linspace(0.1, 0.9, 9)):
        simulation.update_saturation_conn(sat)

        output_filename = "ip" + str(n).zfill(8)
        network.export_to_vtk(output_filename)


![Alt Text](http://media.giphy.com/media/3mJSs8JhtuqGyUEnTZ/giphy.gif)


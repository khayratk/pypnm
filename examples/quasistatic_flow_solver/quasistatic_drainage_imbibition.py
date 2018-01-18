import numpy as np

from pypnm.flow_simulation.quasi_static_simulation import QuasiStaticSimulation
from pypnm.porenetwork.network_factory import structured_network


def run_quasi_static(network):
    simulation = QuasiStaticSimulation(network)
    simulation.apply_initial_conditions()

    # VTK post processing - To save space the user specifies which fields to output
    simulation.create_vtk_output_folder("paraview_quasistatic_run", delete_existing_files=True)
    simulation.add_vtk_output_pore_field(network.pores.invaded, "Pore_invaded")
    simulation.add_vtk_output_pore_field(network.pores.vol, "Pore_volume")
    simulation.add_vtk_output_tube_field(network.tubes.invaded, "Tube_invaded")
    simulation.add_vtk_output_tube_field(network.tubes.vol, "Tube_volume")

    # Define sequence of nonwetting saturations to explore.
    # The points below define a Primary Drainage- Secondary Imbibition- Secondary Drainage cycle
    # These points will not necessarily be reached due to trapping during imbibition
    saturation_points = np.concatenate(
        [np.linspace(0.0, 0.99, 50), np.linspace(0.99, 0.0, 50), np.linspace(0.0, 0.99, 50)])

    for n, sat in enumerate(saturation_points):
        if n > 0:
            if saturation_points[n] > saturation_points[n - 1]:
                if simulation.sat_comp.sat_nw_conn() < saturation_points[n]:
                    simulation.update_saturation_conn(sat)
                else:
                    continue

            if saturation_points[n] < saturation_points[n - 1]:
                if simulation.sat_comp.sat_nw_conn() > saturation_points[n]:
                    simulation.update_saturation_conn(sat)
                else:
                    continue

        sat_nw_conn = simulation.get_nonwetting_connected_saturation()
        p_c_max = np.max(network.pores.p_c[network.pores.connected == 1])
        print ("Nonwetting connected saturation: %g, P_c: %g" %(sat_nw_conn, p_c_max))

        output_filename = "ip" + str(n).zfill(8) + ".vtp"
        simulation.write_vtk_output(output_filename)


if __name__ == "__main__":
    network = structured_network(30, 15, 15, media_type="consolidated", periodic=False)

    # Hack: Needed to stop cooperative pore filling at inlet pores
    network.set_radius_pores(network.pi_in, np.max(network.pores.r) * 5)

    network.restrict_volume([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    network.set_zero_volume_at_inout()
    run_quasi_static(network)

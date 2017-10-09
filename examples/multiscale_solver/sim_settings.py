from collections import defaultdict


def make_hash():
    return defaultdict(make_hash)

sim_settings = make_hash()

sim_settings['fluid_properties']['mu_n'] = 0.1
sim_settings['fluid_properties']['mu_w'] = 1.0
sim_settings['fluid_properties']['gamma'] = 1.0

sim_settings['dynamic']['bc_type'] = "neumann"  # dirichlet or neumann
sim_settings['dynamic']['p_n_inlet'] = 100000.0
sim_settings['dynamic']['p_w_inlet'] = 000000.0
sim_settings['dynamic']['q_n_inlet'] = 1e-8
sim_settings['dynamic']['q_w_inlet'] = 0e-8
sim_settings['dynamic']['p_w_outlet'] = 0.0

sim_settings['output']['ds_out'] = 0.05
sim_settings['output']['vtk_output_folder'] = "paraview_dynamic"
sim_settings['output']['pp_bounding_box_percent'] = [0.25,  0.75, 0.0, 1.0, 0.0, 1.0]

sim_settings['simulation']['pores_have_conductances'] = False
sim_settings['simulation']['sim_bounding_box_percent'] = [0.25,  0.75, 0.0, 1.0, 0.0, 1.0]
sim_settings['simulation']['s_max'] = 0.5
sim_settings['simulation']['save_restart_file'] = False

sim_settings['network']['type'] = "structured"  # structured or statoil
sim_settings['network']['periodic'] = False
sim_settings['network']['structured_dimensions'] = [22, 12, 12]
sim_settings['network']['restrict_volume'] = False
sim_settings['network']['restrict_volume_bbox'] = [0.25,  0.75, 0.0, 1.0, 0.0, 1.0]
sim_settings['network']['structured_media_type'] = 'consolidated'  # consolidated or unconsolidated
sim_settings['network']['statoil_network_name'] = "statoilnetwork/berea/long_network/berea"
sim_settings['network']['load_from_file'] = False
sim_settings['network']['filename'] = 'network.pkl'

sim_settings['multiscale']['ds'] = 0.1
sim_settings['multiscale']['solver'] = "lu"  # Options are "lu", "petsc", "AMG", "trilinos", "mltrilinos"

#"statoilnetwork/berea/standard_network/berea"

# General settings
sim_settings['linear_solver'] = "LU"
sim_settings['enable_profiling'] = True

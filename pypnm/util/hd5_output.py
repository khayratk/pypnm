import h5py


def add_field_to_hdf_file(hdf_file_name, group_label, field_name, field):
    with h5py.File(hdf_file_name, 'a') as f:
        group_name = '/'+str(group_label).zfill(4)
        grp = f.require_group(group_name)
        g = grp.require_dataset(name=field_name, shape=field.shape, dtype=field.dtype, compression='gzip')
        g[:] = field


def add_attribute_to_hdf_file(hdf_file_name, group_label, attribute_name, val):
    with h5py.File(hdf_file_name, 'a') as f:
        group_name = '/'+str(group_label).zfill(4)

        grp = f.require_group(group_name)

        grp.attrs[attribute_name] = val



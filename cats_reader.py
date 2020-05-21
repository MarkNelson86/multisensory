import numpy as np
import h5py
import pandas as pd
from functools import reduce

class CatsReader:

    def __init__(self, all_keys, block_to_column_header):
        self.all_keys = all_keys
        self.block_to_column_header = block_to_column_header

    def get_column_headers_for_epoch(self, trail):
        # trail = (1, 2, 3, 4) is a tuple
        cat, penetration, channel, block = trail #unpacking trail  # Modify here?
        return self.block_to_column_header[block]
        
    def read_unit(self, enumerated_unit, trail):   # Modify here?
        i, unit = enumerated_unit
        data, epoch = unit[0], unit[1]
        epoch_df = pd.DataFrame(epoch, columns=self.get_column_headers_for_epoch(trail))
        index_df_single = pd.DataFrame((*trail, i), index=['cat', 'penetration', 'channel', 'block', 'unit']).T # series to combine with dataframe
        index_df_duplicated = pd.concat([index_df_single] * epoch_df.shape[0], ignore_index=True)
        final_index_df = pd.concat([index_df_duplicated, epoch_df], axis=1).reindex(columns=self.all_keys)
        multi_index = pd.MultiIndex.from_frame(final_index_df)#, names=['cat','penetration','channel','block','unit',*get_column_headers_for_epoch(trail)])
        #print(multi_index[:2])
        final_df = pd.DataFrame(data, index=multi_index).swaplevel(2,3)
        return final_df

    ##############################
    # read_group(file_hande, cats_path)
    # reading the hdf5 group and drills through branches of the object tree (down to data and epochs)
    # LOTS OF MAGICCCCC
    # tuples so variables can't change
    ####
    def read_group(self, f, e, trail):                                  # Modify here?
        # See http://book.pythontips.com/en/latest/map_filter.html
        next_keys = tuple(e.keys())
        next_refs_for_keys = tuple(map(lambda key: '{}{}{}'.format(e.name,'/', key), next_keys))
        groups = tuple(map(lambda group_ref: self.read_node(f, group_ref, trail), next_refs_for_keys))
        if len(groups) == 2 and next_keys[0] == 'data' and next_keys[1] == 'epocs':
            all_units = zip(groups[0], groups[1])
            return pd.concat(list(map(lambda unit: self.read_unit(unit, trail), enumerate(all_units))), names=self.all_keys)
        #print(list(filter(lambda group_element: type(group_element) != 'numpy.ndarray', groups[0])))
        return pd.concat(list(filter(lambda group_element: type(group_element) != np.ndarray, groups[0])), names=self.all_keys)

    ##############################
    # read_dataset(file_hande, cats_path)
    # going through nodes on the object tree, extracts data and epochs
    ####   
    def read_dataset(self, f, e, trail):
        array = np.array(e)
        if array.dtype == 'object':
            #print('Element is dataset with objects.')
            #print(f[list(array)[0][0]].shape, print(f[list(array)[1][0]]))
            return tuple(
                map(
                    lambda index_element: self.read_node(f, index_element[1], (*trail, index_element[0])), 
                    enumerate(array.flatten())
                )
            )
            
        elif array.dtype == 'uint64':
            #print('index: ', (*trail, column), 'empty: ', e)
            #print('Element is dataset with uint64.')
            return np.array([]).T
        elif array.dtype == 'float64':
            #print('index: ', (*trail, column), 'e: ', e)
            # CODE HERE WILL MAKE SERIES  :D 😍 🥰 omg too much
            return array.T
        else:
            raise Exception('Unrecognized data set')

    ##############################
    # read_node(file_handle, cats_path)
    # this function takes the mat file and reads the data. also ignores empty arrays
    ####
    def read_node(self, f, ref, trail=()): 
        #print('Looking up ref: ', ref)
        e = f[ref] # element to check
        if type(e) == h5py._hl.group.Group:
            group = self.read_group(f, e, trail)
            #print(group.index[:2])
            print(trail, group.shape)
            return group
        elif type(e) == h5py._hl.dataset.Dataset:
            to_return = self.read_dataset(f, e, trail)
            #print(len(to_return))
            return to_return
        else:
            raise Exception('Unknown element type')
        raise Exception("Unknown element type")


    ##############################
    # read_cats_file(file_path)
    # this function takes the mat file path and operates on the file object
    ####
    def read_cats_file(self, file_path):
        f = h5py.File(file_path, 'r')
        results = self.read_node(f, '/cats')    # results = self.read_node(f, ‘/S’)
        return results

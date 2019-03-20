import h5py
import goparser
import uniprotparser
import numpy as np
from collections import OrderedDict
import pickle
with h5py.File('one_hot_go.h5py', 'w-') as h5file:
    pass
inputfilename ='go-basic.obo'
ancestor = goparser.getrelationship(inputfilename)
go_ids = np.array(list(ancestor.keys()), dtype=np.int)
go_ids.sort()
n_go_ids = len(go_ids)
go_id_to_ind = dict([(go_id, ind) for ind, go_id in enumerate(go_ids)])
# And you have a one-hot array (I'll use a numpy array but a hdf5 array would behave the same):
one_hot_dag = np.zeros((n_go_ids, n_go_ids), dtype=np.bool)

# We use the ancestor_dict to set the one_hot_dag but we need a recursive function to traverse the ancestor_dict:
def recursive_is_a(dag_key, upper_leaves=()):
    """Function for recursively setting all is_a connections of a go term in one_hot_dag to True"""
    # Add the current key to the upper leaves
    upper_leaves += (dag_key,)
    # Set the key entry for itself and all it's upper leaves to True
    one_hot_dag[[go_id_to_ind[go_id] for go_id in upper_leaves], go_id_to_ind[dag_key]] = True
    # And then use this function to set all is_a connections and their is_a connections to True recursively
    is_a_connections = ancestor[dag_key]
    [recursive_is_a(is_a_keys, upper_leaves) for is_a_keys in is_a_connections]


def process_go_id(i):
    """Add some verbose output"""
    go_id = go_ids[i]
    recursive_is_a(go_id)
    print("processed go id {} of {}".format(i, len(go_ids)))


[process_go_id(i) for i in range(len(go_ids))]
print(len(go_ids))

go_id_to_ind_array = np.full((np.max(go_ids)+1,), fill_value=-1, dtype=np.int)
go_id_to_ind_array[go_ids] = np.arange(len(go_ids))

with h5py.File('one_hot_go.h5py', 'w') as h5file:
    h5file.create_dataset('one_hot_dag', data=one_hot_dag, compression="lzf", chunks=True)
    h5file.create_dataset('index_to_go_id', data=go_ids, compression="lzf", chunks=True)
    h5file.create_dataset('go_id_to_index', data=go_id_to_ind_array, compression="lzf", chunks=True)

print("Number of is_a connections for all go terms:\n{}".format(list(zip(go_ids, one_hot_dag.sum(axis=1)))))

print("Total is_a connections found: {}".format(one_hot_dag.sum()))

print("is_a relations of go_id {}: {}".format(go_ids[0], np.asarray(one_hot_dag[0], np.int)))

print("Average number of is_a relations per go_id {}".format(np.mean(one_hot_dag.sum(axis=1))))



uni = uniprotparser.uni8go
p = OrderedDict(uni)
with open('uni2go.pkl', 'wb') as f:
    # Pickle will store our object into the specified file
    pickle.dump(p, f)








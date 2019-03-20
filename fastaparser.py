import pickle
from collections import OrderedDict
import uniprotparser
import numpy as np
def sequence(inputfile):
    file = open(inputfile,'r')
    fastadic = dict()
    finaldic = dict()
    blocks = file.read()
    block = blocks.split('>')
    for entries in block[1:]:
        code = entries.split('\n')[0].split(' ')[0][len('Uniref100_'):]
        sequences = entries.split('\n')[1:]
        sequence = ''.join(sequences)
        fastadic[code] = sequence
    uniprot = uniprotparser.uni8go
    for key in fastadic.keys():
        for id in uniprot.keys():
            if key == id:
                finaldic[id] = fastadic[key]
    return finaldic
m = sequence('test.fasta')
z = OrderedDict(m)
p = max(z.values())
print(len(p))
v = ''.join(z.values())
occurences = dict()
maxoccurences = dict()
for u in v:
    occurences[u] = v.count(u)
    maxoccurences[u] = p.count(u)
print(maxoccurences)
import matplotlib.pyplot as plt
plt.hist( np.array(list(occurences.values()), dtype=np.float), color='g')
plt.xlabel('Aminoacid')
plt.ylabel('Occurences')
plt.title('Amino acid occurences in uniref sequences')
plt.savefig('aminoacid.png')
plt.close()
plt.bar(list(maxoccurences.keys()), list(maxoccurences.values()), color='g')
plt.xlabel('Aminoacid')
plt.ylabel('Occurences')
plt.title('Amino acid occurences in longest sequence')
plt.savefig('longest aminoacid.png')
plt.close()
z_keys = list(z.keys())
import random
random.seed(123)
_ = random.shuffle(z_keys)
z_train = OrderedDict([(k, z[k]) for k in z_keys[:int(len(z)/5*3)]])
z_validation = OrderedDict([(k, z[k]) for k in z_keys[int(len(z)/5*3):int(len(z)/5*4)]])
z_test = OrderedDict([(k, z[k]) for k in z_keys[int(len(z)/5*4):]])
with open('train_sequences.pkl', 'wb') as f:
    pickle.dump(z_train, f)
    print('done')
    print(z)
with open('validation_sequences.pkl', 'wb') as f:
    pickle.dump(z_validation, f)
    print('done')
    print(z)
with open('test_sequences.pkl', 'wb') as f:
    pickle.dump(z_test, f)
    print('done')
    print(z)
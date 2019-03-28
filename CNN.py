import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import h5py
import torch.tensor as t
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from preprocessing import PadToEqualLengths

#split unref file into 3 files

with open('uni2go.pkl', 'rb') as f:

    # Pickle will store our object into the specified file
    data = pickle.load(f)
annotation = data
with open('train_sequences.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    train_data = pickle.load(f)
train_sequences = train_data
with open('validation_sequences.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    validate_data = pickle.load(f)
validate_sequences = validate_data
with open('test_sequences.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    test_data = pickle.load(f)
test_sequences = test_data
with open('ggid.pkl', 'rb') as f:
    ggid = pickle.load(f)
    ggidt = ggid

gid = h5py.File('one_hot_go.h5py','r')
class TensorDataset(Dataset):

    def __init__(self, annotation, sequences,gid,ggid):
        self.ANNOTATION = list(annotation.keys())
        self.SEQUENCE = sequences
        self.ANNOTATION = [a for a in self.ANNOTATION if a in self.SEQUENCE.keys()]
        self.ANNOTATIONs = annotation
        self.goid = gid
        self.gomatrix = gid['one_hot_dag'][:]
        self.goid_to_index = gid['go_id_to_index'][:]
        self.vocab = "GPAVLIMCFYWHKRQNEDSTUX"
        self.aa_lookup = dict([(k, v) for v, k in enumerate(self.vocab)])
        self.allowed_goids = np.array(list(ggidt.values())) > 1000
        self.n_allowed_goids = self.allowed_goids.sum()
        self.nofclasses = self.n_allowed_goids

    def __len__(self):
        return len(self.ANNOTATION)

    def __getitem__(self, idx):
        uniprodid = self.ANNOTATION[idx]
        go_matrix= self.gomatrix
        list_of_go_ids = self.ANNOTATIONs[uniprodid]
        label = np.zeros_like(go_matrix[0], dtype=np.float32).flatten()
        for go_id in list_of_go_ids:
            goid_to_idx = self.goid_to_index[go_id]
            if goid_to_idx >= 0:
                label += go_matrix[goid_to_idx]
        label[:] = label > 0
        label = label[self.allowed_goids]
        sequence = self.SEQUENCE[uniprodid]#works but give key error will be solved when do it on full files
        x = np.zeros(shape=(len(sequence), len(self.aa_lookup)), dtype=np.float32)
        x[np.arange(len(sequence)), [self.aa_lookup[aa] for aa in sequence]] = 1
        return label,uniprodid,x

trainset = TensorDataset(annotation,train_sequences,gid,ggidt)
testset = TensorDataset(annotation,test_sequences,gid,ggidt)
seq_padding = PadToEqualLengths(padding_dims=(None, None, 1), padding_values=(None, None, 0))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=2,collate_fn=seq_padding.pad_collate_fn)

testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2,collate_fn=seq_padding.pad_collate_fn)
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self,n_outputs):
        super(Net, self).__init__()
        n_inputfeatures = 22
        self.conv1 = nn.Conv1d(in_channels=n_inputfeatures, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.fc3 = nn.Linear(512, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(dim=2)[0]
        x = self.fc3(x)
        return x


net = Net(n_outputs=trainset.nofclasses)

import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()#adds sigmoid
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)#it uses adam update algorithm, weight decay prevents your network from overfit(probably mention in the trainings section of method)



n_updates = 1000
update = 0
running_losslist = []

while update <= n_updates:
    running_loss = 0.0
    n_samples = 0
    for data in trainloader:
        labels, uniprodid, inputs_padded = data
        inputs, orig_lens = inputs_padded
        n_samples += len(inputs)
        print("n_samples {}".format(n_samples))

        # zero the parameter gradients
        optimizer.zero_grad()
        update += 1
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_losslist.append(running_loss/1000)
        if update % 100:    # print every 2000 mini-batches
            print('[%7d] loss: %.8f' %
                  (update, running_loss / 1000))
            running_loss = 0.0
        if update > n_updates:
            break

with open('stats.txt','w') as dar:
    for k in running_losslist:
        dar.write(str(k)+'\n')
print('Finished Training')

##############################################

        # 2. Log values and gradients of the parameters (histogram summary)




'''dataiter = iter(testloader)
sample = dataiter.__next__()
images, labels = sample["Sequence"], sample["GoID"]
outputs = net(images)

predicted = outputs'''

correct = 0
total = 0
tlist = []
plist = []
roc_list = []
ba_list = []
with torch.no_grad():#roc score
    for data in testloader:
        labels, uniprodid, inputs_padded = data
        inputs, orig_lens = inputs_padded
        outputs = net(inputs)
        predicted = torch.sigmoid(outputs).data
        tlist.append(labels)
        plist.append(predicted)
    testlabels = np.concatenate(tlist,axis=0)
    predicts = np.concatenate(plist,axis=0)
    for c in range(labels.shape[1]):
        ba = balanced_accuracy_score(testlabels[:,c], predicts[:,c].round())
        ba_list.append(ba)
        try:
            roc = roc_auc_score(testlabels[:,c], predicts[:,c])
            roc_list.append(roc)
        except ValueError:
            roc = np.nan
        print(str(c)+'balanced accuracy score: '+str(ba)+'\n'+'Roc-auc score score: '+str(roc))# cross validation needed, have a seperate validation set
        with open('roc_stats.txt','w') as dar:
            for k in roc_list:
                dar.write(str(k)+'\n')
        print('Finished Training')
        with open('ba_stats.txt','w') as dar:
            for k in ba_list:
                dar.write(str(k)+'\n')
        print('Finished Training')

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
from sklearn.metrics import confusion_matrix

#split unref file into 3 files

with open('uni2go.pkl', 'rb') as f:

    # Pickle will store our object into the specified file
    data = pickle.load(f)
annotation = data
with open('train_sequences.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    data = pickle.load(f)
sequences = data

gid = h5py.File('one_hot_go.h5py','r')
class TensorDataset(Dataset):

    def __init__(self, annotation, sequences,gid):
        self.ANNOTATION = list(annotation.keys())
        self.SEQUENCE = sequences
        self.ANNOTATION = [a for a in self.ANNOTATION if a in self.SEQUENCE.keys()]
        self.ANNOTATIONs = annotation
        self.goid = gid
        self.gomatrix = gid['one_hot_dag'][:]
        self.goid_to_index = gid['go_id_to_index'][:]
        self.vocab = "GPAVLIMCFYWHKRQNEDSTUX"
        self.aa_lookup = dict([(k, v) for v, k in enumerate(self.vocab)])


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
        sequence = self.SEQUENCE[uniprodid]#works but give key error will be solved when do it on full files
        x = np.zeros(shape=(len(sequence), len(self.aa_lookup)), dtype=np.float32)
        x[np.arange(len(sequence)), [self.aa_lookup[aa] for aa in sequence]] = 1
        sample = {'GoID': label, 'UniprotID': uniprodid, 'Sequence': x}
        return sample

trainset = TensorDataset(annotation,sequences,gid)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                         shuffle=False, num_workers=2)
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        n_inputfeatures = 22
        self.conv1 = nn.Conv1d(in_channels=n_inputfeatures, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.fc3 = nn.Linear(512, 30817)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(dim=2)[0]
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()#adds sigmoid
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)#it uses adam update algorithm, weight decay prevents your network from overfit(probably mention in the trainings section of method)



n_updates = 10
update = 0
running_losslist = []

while update <= n_updates:
    running_loss = 0.0
    n_samples = 0
    for data in trainloader:
        n_samples += len(data["Sequence"])
        print("n_samples {}".format(n_samples))
        # get the inputs
        inputs, labels = data["Sequence"], data["GoID"]

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
        running_losslist.append(running_loss/100)
        if update % 100:    # print every 2000 mini-batches
            print('[%7d] loss: %.8f' %
                  (update, running_loss / 100))
            running_loss = 0.0
        if update > n_updates:
            break
with open('stats.txt','w') as dar:
    dar.write('\t --------Running Loss-------\t\n')
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
with torch.no_grad():
    for data in testloader:
        images, labels = data["Sequence"], data["GoID"]
        outputs = net(images)
        predicted = torch.sigmoid(outputs).data
        tlist.append(labels)
        plist.append(predicted)
    testlabels = np.concatenate(tlist,axis=0)
    predicts = np.concatenate(plist,axis=0)
    ba = balanced_accuracy_score(testlabels[:,0], predicts[:,0].round())
    f1 = f1_score(testlabels[:,0], predicts[:,0].round())
    try:
        roc = roc_auc_score(testlabels[:,0], predicts[:,0])
    except ValueError:
        roc = np.nan
    print('balanced accuracy score: '+str(ba)+'\n'+'F1 score: '+str(f1)+'\n'+'Roc-auc score score: '+str(roc))# cross validation needed, have a seperate validation set


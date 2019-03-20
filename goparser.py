import copy

MultientryAtt = ['is_a','alt_id']


def read_file2list(inputFile):
    with open(inputFile, "r") as fi:
        myList = fi.readlines()
        return myList

def remove_values_from_dict(dict, keys):
    newdict = {}
    for key in dict.keys():
        if key in keys:
            newdict[key] = dict[key]
    return newdict

def create_dict(attribute, value):
    my_dict = dict()
    for i in range(len(attribute)):
        key = attribute[i]
        if key not in my_dict:
            my_dict[key] = []
        if my_dict[key] not in my_dict[key]:
            my_dict[key].append(value[i])
    for i in my_dict:
        if len(my_dict[i]) == 1 and i not in MultientryAtt:
            my_dict[i] = copy.deepcopy(my_dict[i][0])
    return my_dict
def connection(key,list):
    goidduct = {}
    goidlist = []
    for items in list:
        goid = items.split("!")[0]
        goidlist.append(int(goid.strip()[3:]))
    goidduct[int(key)] = goidlist
    return goidduct

def parseobofile(inputfilename):
    attribute = []
    value = []
    finaldict = {}
    needlist = ['id','namespace',"is_a",'alt_id']
    fi = read_file2list(inputfilename)[24:]
    maxi = []
    for ln in fi:
        if not ln.startswith("["):
            if not ln.startswith("\n"):
                try:
                    a = ln.split(": ")[0]
                    v = (ln.split(': ')[1]).strip()
                    attribute.append(a)
                    value.append(v)
                except:
                    pass
        else:
            obo_dict = create_dict(attribute, value)
            k = remove_values_from_dict(obo_dict,needlist)
            try:
                if k['id'].startswith('GO'):
                    maxi.append(k['namespace'])
                    if k['namespace'] == 'biological_process':
                        finaldict[int(k['id'][3:])] = k
                    if 'alt_id' in k.keys():
                        for alids in k['alt_id']:
                            finaldict[int(k[alids][3:])] = k
            except KeyError:
                pass
            attribute = []
            value = []
    return finaldict,maxi

def getrelationship(inputfilename):
    relationship = {}
    finaldict = parseobofile(inputfilename)[0]
    for keys in finaldict.keys():
        try:
            for values in connection(keys,finaldict[keys]['is_a']).values():
                relationship[keys] = values
        except KeyError:
            relationship[keys] = []
    return relationship

rel = getrelationship('go-basic.obo')
p = parseobofile('go-basic.obo')[1]
processes = dict()
county = {}
for process in p:
    if process in processes.keys():
        pass
    else:
        processes[process] = p.count(process)
print(processes)
import matplotlib.pyplot as plt
plt.bar(list(processes.keys()), processes.values(), color='g')
plt.xlabel('namespace')
plt.ylabel('Occurences')
plt.title('Entry distribution regarding namspace')
plt.savefig('distribution.png')
plt.close()
plt.bar(list(county.keys()), list(county.values()), color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Entry distribution regarding namspace')
plt.savefig('distribution.png')
plt.close()
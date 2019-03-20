import goparser
import copy
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
file = open('goa_human2.gaf','r')
text = []
keys = []
values = []
for lines in file:
    liness = lines.strip()
    if not liness.startswith("!"):
        text.append(liness)
for entry in text:
    items = entry.split('\t')
    keys.append(items[1])
    values.append(int(items[4][3:]))
MultientryAtt = keys
uni8go = create_dict(keys,values)
ids = goparser.getrelationship('go-basic.obo')
for keyy in uni8go.keys():
    for key in ids.keys():
        if key in uni8go[keyy]:
            uni8go[keyy].extend(ids[key])

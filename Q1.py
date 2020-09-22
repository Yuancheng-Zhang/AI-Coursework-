import pandas as pd

""" Q1: Basic Data Processing
Question: Determine how many distinct source IP addresses, destination IP addresses, and
classifications there are in this dataset.
"""

"Read File:"
data = pd.read_csv('coursework1.csv')
# data = pd.read_csv('coursework2.csv')

"Convert the File to List:"
sourceip = data['sourceIP'].tolist()
destip = data['destIP'].tolist()
classfication = data['classification'].tolist()

"Function for Counting"
def count(entry):
    store = []
    S = range(0, len(entry))
    
    for s in S:
        store.append(entry[s])
    entry_list = set(store)
    entry_count = len(entry_list)
    
    return entry_count






print('SourceIP: ', count(sourceip))
print('DestIP: ', count(destip))
print('Classification: ', count(classfication))
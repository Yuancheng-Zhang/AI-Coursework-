from collections import Counter
import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split

"""Q5: Decision Trees
Question: Write some of your own code to learn a decision tree using the 2 features above 
(i.e. the source cluster and the destination cluster) to predict the classification field. 
You should not use the built in sklearn decision tree as this does not deal well with categorical features.
Display your learnt decision tree. In how many cases does your learnt decision tree give an 
unambiguous answer (or a fairly certain answer)?
"""
###############################################################################

"Read the file: "
data = pd.read_csv('coursework1.csv')
# data = pd.read_csv('coursework2.csv')

"Data Pre-processing: "
sourceip = data['sourceIP'].tolist()
destip = data['destIP'].tolist()
classification = data['classification'].tolist()
dataframe = pd.DataFrame(data=data) 

"Convert the csv into list and re-order the sequence"
d = pd.Series(destip).value_counts().reset_index().values.tolist()
s = pd.Series(sourceip).value_counts().reset_index().values.tolist()
c = pd.Series(classification).value_counts().reset_index().values.tolist()


###############################################################################

"Calculate the numbers of occurrence of each element"
s1, s2, s3, s4 = 0, 0, 0, 0
d1, d2, d3, d4 = 0, 0, 0, 0
sf, ss, st, sh = [], [], [], []
df, ds, dt, dh = [], [], [], []

###############################################################################

"Clustering the SourceIP addresses"
Ks = range(0, len(s))

for k in Ks:
    if s[k][1] <= 20:
        s1 += 1
        sf.append(s[k][0])
    elif s[k][1] <= 200:
        s2 += 1
        ss.append(s[k][0])
    elif s[k][1] <= 400:
        s3 += 1
        st.append(s[k][0])
    # elif s[k][1] > 400:
    else:
        s4 += 1
        sh.append(s[k][0])
        
ssum = s1 + s2 + s3 + s4

"Replace the SourceIP addresses with their cluster names"
K = range(0, len(data))

for k in K: 
    if dataframe.iat[k, 1] in sf:
        dataframe.iat[k, 1] = 'SourceIP Cluster 1'
    elif dataframe.iat[k, 1] in ss:
        dataframe.iat[k, 1] = 'SourceIP Cluster 2'
    elif dataframe.iat[k, 1] in st:
        dataframe.iat[k, 1] = 'SourceIP Cluster 3'
    else:
        dataframe.iat[k, 1] = 'SourceIP Cluster 4'


###############################################################################

"Clustering the DestIP addresses"
Kd = range(0, len(d))

for k in Kd:
    if d[k][1] <= 40:
        d1 += 1
        df.append(d[k][0])
    elif d[k][1] <= 100:
        d2 += 1
        ds.append(d[k][0])
    elif d[k][1] <= 400:
        d3 += 1
        dt.append(d[k][0])
    # elif d[k][1] > 400:
    else:
        d4 += 1
        dh.append(d[k][0])
        
dsum = d1 + d2 + d3 + d4

"Replace the DestIP addresses with their cluster names"
for k in K: 
    if dataframe.iat[k, 3] in df:
        dataframe.iat[k, 3] = 'DestIP Cluster 1'
    elif dataframe.iat[k, 3] in ds:
        dataframe.iat[k, 3] = 'DestIP Cluster 2'
    elif dataframe.iat[k, 3] in dt:
        dataframe.iat[k, 3] = 'DestIP Cluster 3'
    else: 
        dataframe.iat[k, 3] = 'DestIP Cluster 4'

###############################################################################
"Prepare the dataset for making decision tree."

sourceIP_cluster = dataframe['sourceIP'].tolist()
destIP_cluster = dataframe['destIP'].tolist()
classification_name = dataframe['classification'].tolist()

DataSet = [ [] for _ in range(len(sourceIP_cluster)) ]
K = range(0, len(sourceIP_cluster))

for k in K:
    DataSet[k].append(sourceIP_cluster[k])
    DataSet[k].append(destIP_cluster[k])
    DataSet[k].append(classification_name[k])


"""Dataset is now prepared,
   in the form of ["SourceIP Cluster name, DestIP Cluster name, Label"]
"""
# print(DataSet)

###############################################################################
"Create training and testing dataset."

"Split the data randomly: "
train, test = train_test_split(DataSet, test_size=0.1, random_state=0)

"Set the training data and labels: "
training_data = train
label = ['sourceIP', 'destIP', 'classification']


###############################################################################
"""
    Decision Tree (CART)
"""

###############################################################################


def class_counts(dataset):
    "Counting the number of each type of labels"
    counts = {}  # a dictionary of label -> count.
    for data in dataset:
        label = data[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
           
    return counts

def class_counts_prob(counts):

    total = sum(counts.values()) * 1.0
    for classification in counts.keys():
        counts[classification] = round((counts[classification] / total), 3)
    return counts

###############################################################################

def numeric(value):
    "Return TRUE if 'value' is numeric; if not, return FALSE"
    return isinstance(value, int) or isinstance(value, float)

###############################################################################

class Question:
    "Ask A Question to partition the dataset"
    
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an 'example' to that in this 'question'.
        val = example[self.column]
        
        if numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print the question in a readable format.
        condition = "=="
        if numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            label[self.column], condition, str(self.value))


def partition(dataset, question): 
    # input question = Question
    "Partition the dataset: if match the question, then be true."
    
    true_data, false_data = [], []
    for data in dataset:
        if question.match(data):
            true_data.append(data)
        else:
            false_data.append(data)
    return true_data, false_data

###############################################################################

"Opt 1: gini index for impurity. (Used in this version)"
def gini(dataset):
    counts = class_counts(dataset) # def class_counts
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(dataset))
        impurity -= prob_of_lbl ** 2
    return impurity

"Opt 2: Shannon Entropy. (Not used in this version)"
def calcShannonEnt(dataset):
    countDataSet = len(dataset)
    labelCounts={}
    for featVec in dataset:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/countDataSet
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

###############################################################################

def info_gain(left, right, current_uncertainty):
    "Information Gain"
    p = float(len(left)) / (len(left) + len(right))
    ### current_uncertainty = gini(dataset)
    # return current_uncertainty - p * calcShannonEnt(left) - (1 - p) * calcShannonEnt(right)
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(dataset):
    "Find the best question to ask then split."
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    # current_uncertainty = calcShannonEnt(dataSet)
    current_uncertainty = gini(dataset)
    n_features = len(dataset[0]) - 1  # number of columns

    for feature in range(n_features):  # for each feature
        values = set([data[feature] for data in dataset])  # unique values in the column
        for val in values:  # for each value
            question = Question(feature, val)
            true_data, false_data = partition(dataset, question)
            if len(true_data) == 0 or len(false_data) == 0:
                continue
            gain = info_gain(true_data, false_data, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

###############################################################################

class Leaf:

    def __init__(self, dataset):
        self.predictions = class_counts_prob(class_counts(dataset))


class Decision_Node:

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(dataset):
    
    gain, question = find_best_split(dataset)
    if gain == 0:
        return Leaf(dataset)

    true_data, false_data = partition(dataset, question) # partition()
    true_branch = build_tree(true_data)
    false_branch = build_tree(false_data)

    return Decision_Node(question, true_branch, false_branch)

###############################################################################

def print_tree(node, spacing=""):

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        
        return
    
    print (spacing + str(node.question))
    
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

###############################################################################

def classify(dataset, tree):
    "input = dataset, build_tree"
    
    if isinstance(tree, Leaf): 
        return tree.predictions

    if tree.question.match(dataset):
        return classify(dataset, tree.true_branch)
    else:
        return classify(dataset, tree.false_branch)

###############################################################################

def leaf(op_cla):
    "input = output of classify."
    total = sum(op_cla.values()) * 1.0
    prob = {}
    
    for entry in op_cla.keys():
        prob[entry] = str(int(op_cla[entry] / total * 100)) + "%"
    
    "return the probability of decision."
    return prob

###############################################################################

# The next two sections are for calc accu and counting the number of unambiguous answers
def accuracy(dataset, my_tree):
    
    data_length = len(dataset)
    true_case = 0
    
    for data in dataset:
        classify_result = classify(data, my_tree)
        
        if max(classify_result, key=lambda x:classify_result[x]) == data[-1]:
            true_case += 1
            
    accuracy = true_case / data_length
    
    return accuracy

###############################################################################

def unambiguous(dataset, my_tree, thres):
    
    unambiguous_data = []
    
    for data in dataset:
        classify_result = classify(data, my_tree)
        if max(classify_result.values()) >= thres:
        # if len(classify(data, my_tree)) == 1:    
            if data not in unambiguous_data:
                unambiguous_data.append(data)
                
    return unambiguous_data


###############################################################################



if __name__ == '__main__':
    
    "Build Decision Tree and print"
    tree = build_tree(training_data)
    print("....................................................")
    print_tree(tree)
    print("....................................................")
    
    "Test data"
    testing_data = test    
    
    "Classify the test data."
    # for data in testing_data:
    #     print ("Actual: %s. Predicted: %s" %
    #             (data[-1], leaf(classify(data, tree))))
        
    "Print accuracy and the unambiguous answers"
    traacc = accuracy(training_data, tree)    
    print("The accuracy of training data =", traacc)
    
    testacc = accuracy(testing_data, tree)
    print("The accuracy of testing data =", testacc)
    
    threshold = 1
    unambiguous_ans = unambiguous(testing_data, tree, threshold)
    # print(unambiguous_ans)
    print("There are %s unambiguous answers." % (len(unambiguous_ans)))
    
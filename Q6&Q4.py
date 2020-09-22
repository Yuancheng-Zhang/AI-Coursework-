from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

"""Q6 Extension
Examine the dataset coursework2.csv. It contains similar data but has a few more
IP addresses. Using the same clusters of IP addresses (plus sets of previously unseen
source and destinations), are the patterns observed in Q4 still valid? How about
the decision tree in Q5?
"""

#########################################################################################

data = pd.read_csv('coursework2.csv')

sourceip = data['sourceIP'].tolist()
destip = data['destIP'].tolist()

d = pd.Series(destip).value_counts().reset_index().values.tolist()
s = pd.Series(sourceip).value_counts().reset_index().values.tolist()



###############################################################################

# Calculate the numbers of occurrence of each element
s1, s2, s3, s4 = 0, 0, 0, 0
d1, d2, d3, d4 = 0, 0, 0, 0
sf, ss, st, sh = [], [], [], []
df, ds, dt, dh = [], [], [], []

ks = range(0, len(s))
for k in ks:
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
print('SourceIP Cluster')
print(s1, s2, s3, s4)
# print('Probability of SourceIPCluster 1: ', s1 / ssum )
# print('Probability of SourceIPCluster 2: ', s2 / ssum )
# print('Probability of SourceIPCluster 3: ', s3 / ssum )
# print('Probability of SourceIPCluster 4: ', s4 / ssum )
# print(sf)
# print(ss)
# print(st)
# print(sh)
print('..................................................................................................................')

##################################################################################################################################################################################

kd = range(0, len(d))
for k in kd:
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
print('DestIP Cluster')
print(d1, d2, d3, d4)
# print('Probability of DestIPCluster 1: ', d1 / dsum )
# print('Probability of DestIPCluster 2: ', d2 / dsum )
# print('Probability of DestIPCluster 3: ', d3 / dsum )
# print('Probability of DestIPCluster 4: ', d4 / dsum )
# print(df)
# print(ds)
# print(dt)
# print(dh)
print('..................................................................................................................')


###############################################################################

def S_D(SIP):
    M = range(0, len(SIP))
    index = []  # position of each ss element
    F = []  # Now is ss
    for m in M:
        # index = [n for n, x in enumerate(sourceip) if x == sf[m]]
        # index_sourceip.append(index)
        index_sourceip = [n for n, x in enumerate(sourceip) if x == SIP[m]]
        index.append(index_sourceip)

    def findElements(lst1, lst2):
        return list(map(lst1.__getitem__, lst2))

    for m in M:
        Find = findElements(destip, index[m])  # Find the match in destip
        F.append(Find)

    I = range(0, len(F))
    s_df, s_ds, s_dt, s_dh = 0, 0, 0, 0
    for i in I:
        K = range(0, len(F[i]))
        for k in K:
            if F[i][k] in df:
                # print(F[i][k], 'IS in the group of df.')
                s_df += 1
            elif F[i][k] in ds:
                # print(F[i][k], 'IS in the group of ds.')
                s_ds += 1
            elif F[i][k] in dt:
                # print(F[i][k], 'IS in the group of dt.')
                s_dt += 1
            elif F[i][k] in dh:
                # print(F[i][k], 'IS in the group of dh.')
                s_dh += 1

    sum = s_df + s_ds + s_dt + s_dh
    S_D_result = (s_df, s_ds, s_dt, s_dh)
    # S_D_result = [s_df / sum, s_ds / sum, s_dt / sum, s_dh / sum]
    return S_D_result

##################################################################################################################################################################################

def D_S(DIP):
    M = range(0, len(DIP))
    index = []  # position of each ss element
    F = []  # Now is ss
    for m in M:
        # index = [n for n, x in enumerate(sourceip) if x == sf[m]]
        # index_sourceip.append(index)
        index_destip = [n for n, x in enumerate(destip) if x == DIP[m]]
        index.append(index_destip)

    def findElements(lst1, lst2):
        return list(map(lst1.__getitem__, lst2))

    for m in M:
        Find = findElements(sourceip, index[m])  # Find the match in destip
        F.append(Find)

    I = range(0, len(F))
    d_sf, d_ss, d_st, d_sh = 0, 0, 0, 0
    for i in I:
        K = range(0, len(F[i]))
        for k in K:
            if F[i][k] in sf:
                # print(F[i][k], 'IS in the group of sf.')
                d_sf += 1
            elif F[i][k] in ss:
                # print(F[i][k], 'IS in the group of ss.')
                d_ss += 1
            elif F[i][k] in st:
                # print(F[i][k], 'IS in the group of st.')
                d_st += 1
            elif F[i][k] in sh:
                # print(F[i][k], 'IS in the group of sh.')
                d_sh += 1

    sum = d_sf + d_ss + d_st + d_sh
    # D_S_result = (d_sf, d_ss, d_st, d_sh)
    D_S_result = [d_sf / sum, d_ss / sum, d_st / sum, d_sh / sum]

    return D_S_result


###############################################################################

print('SourceIP in DestIP')
print('Conditional Probability that SourceIP Cluster 1 contact DestIP Cluster 1 =', S_D(sf)[0])
print('Conditional Probability that SourceIP Cluster 1 contact DestIP Cluster 2 =', S_D(sf)[1])
print('Conditional Probability that SourceIP Cluster 1 contact DestIP Cluster 3 =', S_D(sf)[2])
print('Conditional Probability that SourceIP Cluster 1 contact DestIP Cluster 4 =', S_D(sf)[3])
print('Conditional Probability that SourceIP Cluster 2 contact DestIP Cluster 1 =', S_D(ss)[0])
print('Conditional Probability that SourceIP Cluster 2 contact DestIP Cluster 2 =', S_D(ss)[1])
print('Conditional Probability that SourceIP Cluster 2 contact DestIP Cluster 3 =', S_D(ss)[2])
print('Conditional Probability that SourceIP Cluster 2 contact DestIP Cluster 4 =', S_D(ss)[3])
print('Conditional Probability that SourceIP Cluster 3 contact DestIP Cluster 1 =', S_D(st)[0])
print('Conditional Probability that SourceIP Cluster 3 contact DestIP Cluster 2 =', S_D(st)[1])
print('Conditional Probability that SourceIP Cluster 3 contact DestIP Cluster 3 =', S_D(st)[2])
print('Conditional Probability that SourceIP Cluster 3 contact DestIP Cluster 4 =', S_D(st)[3])
print('Conditional Probability that SourceIP Cluster 4 contact DestIP Cluster 1 =', S_D(sh)[0])
print('Conditional Probability that SourceIP Cluster 4 contact DestIP Cluster 2 =', S_D(sh)[1])
print('Conditional Probability that SourceIP Cluster 4 contact DestIP Cluster 3 =', S_D(sh)[2])
print('Conditional Probability that SourceIP Cluster 4 contact DestIP Cluster 4 =', S_D(sh)[3])
S_D_sum = S_D(sf)[0] + S_D(sf)[1] + S_D(sf)[2] + S_D(sf)[3] + S_D(ss)[0] + S_D(ss)[1] + S_D(ss)[2] + S_D(ss)[3] + S_D(st)[0] + S_D(st)[1] + S_D(st)[2] + S_D(st)[3] + S_D(sh)[0] + S_D(sh)[1] + S_D(sh)[2] + S_D(sh)[3]
# print(S_D(sf))
# print(S_D(ss))
# print(S_D(st))
# print(S_D(sh))
print('---------------------------------------------------------------------------------------')
print('DestIP in SourceIP')
print('Conditional Probability that DestIP Cluster 1 contact SourceIP Cluster 1 =', D_S(df)[0])
print('Conditional Probability that DestIP Cluster 1 contact SourceIP Cluster 2 =', D_S(df)[1])
print('Conditional Probability that DestIP Cluster 1 contact SourceIP Cluster 3 =', D_S(df)[2])
print('Conditional Probability that DestIP Cluster 1 contact SourceIP Cluster 4 =', D_S(df)[3])
print('Conditional Probability that DestIP Cluster 2 contact SourceIP Cluster 1 =', D_S(ds)[0])
print('Conditional Probability that DestIP Cluster 2 contact SourceIP Cluster 2 =', D_S(ds)[1])
print('Conditional Probability that DestIP Cluster 2 contact SourceIP Cluster 3 =', D_S(ds)[2])
print('Conditional Probability that DestIP Cluster 2 contact SourceIP Cluster 4 =', D_S(ds)[3])
print('Conditional Probability that DestIP Cluster 3 contact SourceIP Cluster 1 =', D_S(dt)[0])
print('Conditional Probability that DestIP Cluster 3 contact SourceIP Cluster 2 =', D_S(dt)[1])
print('Conditional Probability that DestIP Cluster 3 contact SourceIP Cluster 3 =', D_S(dt)[2])
print('Conditional Probability that DestIP Cluster 3 contact SourceIP Cluster 4 =', D_S(dt)[3])
print('Conditional Probability that DestIP Cluster 4 contact SourceIP Cluster 1 =', D_S(dh)[0])
print('Conditional Probability that DestIP Cluster 4 contact SourceIP Cluster 2 =', D_S(dh)[1])
print('Conditional Probability that DestIP Cluster 4 contact SourceIP Cluster 3 =', D_S(dh)[2])
print('Conditional Probability that DestIP Cluster 4 contact SourceIP Cluster 4 =', D_S(dh)[3])
# print(D_S(df))
# print(D_S(ds))
# print(D_S(dt))
# print(D_S(dh))




###############################################################################


"""
=============================================
Discrete distribution as horizontal bar chart
=============================================

"""

SDnames = ['DestIP Cluster 1', 'DestIP Cluster 2',
           'DestIP Cluster 3', 'DestIP Cluster 4']
DSnames = ['SourceIP Cluster 1', 'SourceIP Cluster 2',
           'SourceIP Cluster 3', 'SourceIP Cluster 4']
SDPlot = {
    #'SourceIP Cluster 1': S_D(sf),
    'SourceIP Cluster 2': (0, 0, 0, 1),
    'SourceIP Cluster 3': (S_D(st)[0]/ S_D_sum, S_D(st)[1]/ S_D_sum, S_D(st)[2]/ S_D_sum, S_D(st)[3]/ S_D_sum),
    'SourceIP Cluster 4': (S_D(sh)[0]/ S_D_sum, S_D(sh)[1]/ S_D_sum, S_D(sh)[2]/ S_D_sum, S_D(sh)[3]/ S_D_sum)
    }
DSPlot = {
    'DestIP Cluster 1': D_S(df),
    'DestIP Cluster 2': D_S(ds),
    'DestIP Cluster 3': D_S(dt),
    'DestIP Cluster 4': D_S(dh)
    }


def HSB(results, category_names):

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
               label=colname, color=color)
        xcenters = starts + widths / 2
        r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, widths)):
        #     ax.text(x, y, str(int(c)),
        #             ha='center', va='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax

###############################################################################

HSB(SDPlot, SDnames)
HSB(DSPlot, DSnames)
plt.show()


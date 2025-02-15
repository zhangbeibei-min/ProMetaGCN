# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

a = pd.read_csv(r'.\Meta-GCN\Data\string_interactions.tsv', sep='\t', encoding='utf-8')


a1 = a['#node1'].tolist()
a2 = a['node2'].tolist()
a3 = a1 + a2
a4 = np.unique(a3).tolist()
a5 = pd.Series(a4)
a5.to_csv(r'.\Meta-GCN\Data\Cytokines_list.csv')


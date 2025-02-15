# ProMetaGCN

ProMetaGCN is a model that integrates meta-learning, graph convolutional networks, and protein-protein interaction (PPI) data to assess immune status through plasma proteomics.
![Workflow ](https://github.com/zhangbeibei-min/ProMetaGCN/tree/main/Workflow)
 
# Installation

## **[link](https://github.com/zhangbeibei-min/ProMetaGCN.git)**

# Requirements

- Python 3.7.6
- sklearn 0.22.1
- numpy 1.21.6
- scipy 1.5.2
- pandas 1.0.1
- lightgbm 3.2.0
- xgboost 1.5.2


# Usage

The model consists of two parts:

1.  **Meta-GCN** identifies immune-related plasma proteins.
    
2.  **Immune status prediction** calculates an individual's immune status score.

**Implementation Steps**:

- **Step 1**: Use the code in the **Meta-GCN** folder to identify immune-related proteins.
- > Prepare label data (a small set of positive labels) and a full list of node names.

- **Step 2**: Use the code in the **ImmunescorePrediction** folder to compute the immune status score.
- > Prepare expression levels of each protein and the age-corresponding average immune status score.

# Ciation
Please cite our paper if ProMetaGCN is helpful.

Zhang M, et al. .
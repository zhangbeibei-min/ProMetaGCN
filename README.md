# ProMetaGCN

ProMetaGCN is a model that integrates meta-learning, graph convolutional networks, and protein-protein interaction (PPI) data to assess immune status through plasma proteomics.
![Workflow](https://github.com/zhangbeibei-min/ProMetaGCN/tree/main/Workflow/Workflow.jpg)
## ProMetaGCN Project Structure Documentation
```plaintext
ProMetaGCN/
├── Codes/                      # Core source code directory
│   ├── ImmunescorePrediction/  # Immune status prediction module
│   │   ├── ImmuneScorePrediction.py       # Main prediction program
│   │   └── ImmuneStatusScoreValTest.py    # External validation set prediction
│   ├── Meta-GCN/               # Meta Graph Convolution Network implementation
│   │   ├── Data/               # Protein interaction data
│   │   ├── args.py             # Central parameter configuration
│   │   ├── citation.py         # Main entry for model training/testing
│   │   ├── cytokine_*.py       # Protein data processing utilities
│   │   ├── Label*.py           # Label processing modules
│   │   ├── metrics.py          # Evaluation metrics calculation
│   │   ├── models.py           # Neural network architecture definitions
│   │   └── utils.py           # General utility functions library
├── Data/                       # Dataset repository
│   ├── COVID19.csv            # COVID-19 Dataset2
│   ├── HealthScore.csv        # Healthy control group data
│   ├── NSCLC.csv              # Non-small cell lung cancer dataset
│   └── Omicron.csv            # COVID-19 Dataset1
├── Figure/                     # Visualization results
│   └── Change of plasma protein number with predicted frequency.svg  # Protein quantity vs prediction frequency diagram
└── Workflow/                   # System flowchart
    └── Workflow.png           # Project architecture workflow diagram
  ```

##  **Procedure to Implement**
### **1. Clone the Repository**
   ```
    git clone https://github.com/zhangbeibei-min/ProMetaGCN.git
    cd ProMetaGCN
  ```
### **2. Set Up the Environment**
- Python 3.7.6
- sklearn 0.22.1
- numpy 1.21.6
- scipy 1.5.2
- pandas 1.0.1
- lightgbm 3.2.0
- xgboost 1.5.2
### 3.Usage

The model consists of two parts:

1.  **Meta-GCN** identifies immune-related plasma proteins.
    
2.  **Immune status prediction** calculates an individual's immune status score.

**Implementation Steps**:

- **Step 1**: **Identify Immune-Related Proteins with Meta-GCN**

  -   First, obtain the interaction data of all measured proteins.
  -   Second, Prepare label data (a small number of positive labels) and a complete list of node names.
  -   Run the following code:
  ```
   python Codes/Meta-GCN/cytokine_list.py # Extract unique protein names from protein interactions
   python Codes/Meta-GCN/cytokine_number_name.py # Generate two CSV files for    protein ID-node number mapping and node number-protein name correspondence
   python Codes/Meta-GCN/LabelClass.py # Functions related to label classification
   python Codes/Meta-GCN/Label_ID_encode.py # Map labels to protein identifiers
     ```
  Use the code in the `Meta-GCN`folder. You can adjust the parameters as needed:
  ```
  python Codes/Meta-GCN/citation.py --train_shot_0 10 --train_shot_1 5 --test_shot_0 100 --test_shot_1 35 --epochs 50 --lr 0.0006
  ```
- **Step 2:  Compute Immune Status Score with ImmuneScorePrediction.py**
  -   Train and evaluate the model using the healthy dataset. According to the evaluation index, four machine learning methods with the best performance are obtained.
  ```
  python Codes/ImmunescorePrediction/ImmuneScorePrediction.py
  ```
  -   Validate the model using the external validation dataset. First, the intersection of the immune-related features obtained by Step1 and the features of the external dataset is used as the input features.. 
  - Then, retrain the model and then use the trained model to make predictions on the external dataset.
  ```
  python Codes/ImmunescorePrediction/ImmuneStatusScoreValTest.py
  ```
 
## Ciation
Please cite our paper if ProMetaGCN is helpful. For more detailed research content, please refer to our paper.

Zhang M, et al. ***Immune Status Assessment based on plasma proteomics with Meta Graph Convolutional Networks***.2025.

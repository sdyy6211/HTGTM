# HTGTM

Codes for paper 'Htgtm: Hybrid Temporal-Graph Tabular Model For Complex Multimodal Tabular Data Processing' (http://dx.doi.org/10.1109/MLSP55844.2023.10285918)

**Context**:

The paper is the winner of IEEE MLSP 2023 Data Challenge: Volunteer Retention (https://www.kaggle.com/competitions/MLSP2023-volunteers-01). 

**To run the codes**:

Please place the following files in this folder: "*retention_train_feat.csv*", "*retention_test_feat.csv*" and "*retention_gt_train.csv*"

The files "*gdforeign_ma.csv*" and "*gdlocal_ma.csv*" contain the 7-day moving average of Guangdong's imported COVID-19 cases and locally transmitted COVID-19 cases, respectively.

"*user_id_remapping.pkl*" contains a mapping from user_id to data index. Additionally, "*train_index.pkl*" and "*val_index.pkl*" are used to split the original training dataset into a new training set and a holdout validation set.

Please use "*htgtm.ipynb*" to obtain an evaluation on the holdout validation set.

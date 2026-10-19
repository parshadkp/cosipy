# Radio classes for the BAT sensitivity notebook

`phillipson_2023_table1_radio_classes.csv` contains the 46 RL/RQ labels from
Table 1 of Phillipson, Vogeley, and Boyd (2023),
[arXiv:2211.13774](https://arxiv.org/abs/2211.13774). It was extracted from the
paper's downloadable arXiv LaTeX source. `is_radio_quiet` translates the table's
RQ/RL label to true/false, and `jet_like_core` preserves the table's dagger
annotation. `cosipy_source` records exact matches to the preselected source
names used in the notebook; a blank means that paper source is not in that list.

The paper combines several radio loudness criteria. These labels should be
read as the authors' classifications rather than as BAT catalog fields.

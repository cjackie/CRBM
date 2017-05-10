### CRBM
Code for train a Convolutional Restricted Boltzmann Machine. To test CRBM. run
`python test.py`, assuming requirements are met,

### Requirements
Python 3.6+, and packages in `requriements.txt`. install matplotlib>=2.0.0 for 
running `test.py`

### Preprocessing
It is recommended for data being in the range of [-1,1]. RBM is a model for
binary numbers.

For data far away from 0 (i.e. mean of data is greater than 3). Normalize 
the data before using CRBM, so that the data are centered on 0, with a variance 1.

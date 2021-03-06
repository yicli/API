# API
Python implementation of Automated Problem Identification: https://arxiv.org/abs/1707.00703

Part of Automated Machine Learning 2019/20 at Leiden University

## AUTHORS
Gideon Hanse, Yichao Li and Jaco Tetteroo

## REQUIREMENTS
* python      >= 3.6
* tensorflow  >= 2.0
* numpy
* sklearn
* pandas
* matplotlib
* pydot

Note: for those with access to the LIACS DSLab, a virtual environment with these prerequisites is available:

source /data/s2372029/aml-venv/bin/activate

## INSTRUCTIONS
All scripts in this repository are intended to be ran from the root directory: /API

Most scripts will import from /API/Project; this should work out of the box, but if import errors are encountered, try adding this manually to PYTHONPATH

### USAGE
<pre>
python Project/GA.py            Runs the API algorithm for Boston Housing for 5 generations only, with a small
                                population of 10; and random search with a pool of 20 chromosomes.
python Project/GA_adjusted.py   Runs the adjusted algorithm for Boston Housing for 5 generations only, with a
                                small population of 10.
python Project/RandomForest.py  Runs the random forest model for all datasets.
</pre>

## REPOSITORY CONTENTS
<pre>
API Paper            Published paper for the API algorithm, this is kept in the root directory
/Project             Contains all python scripts
    GA.py            Our implementation of the API algorithm as described in the paper.
    GA_adjusted.py   API algorithm with modified genetic operations
    RandomForest.py  Contains the random forest model.
    preproc.py       Preprocessing script that populated /processed_data. The original datasets are not include 
                     in this repository; please download these to /Project/datasets if you wish to repopulated
                     /processed_data by running this script.
    /result_scripts  Contains python scripts that were used to generate GA results over 20 runs; provided for
                     reference, will take a long time to run. plot_results.py contains functions for extracting
                     and plotting the results.
    /*res            Result folders containing pickled results.
</pre>

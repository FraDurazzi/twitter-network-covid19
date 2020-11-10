# About
This repository contains data and code in order to reproduce the results from the study "International expert communities on Twitter become more isolated during the COVID-19 pandemic". 

# Data
## Twitter dataset
Twitter data related to this study can be found [here](https://zenodo.org/record/4267033#.X6sgdpNKi50).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4267033.svg)](https://doi.org/10.5281/zenodo.4267033)

## Aggregated data
Aggregated data (such as counts) are provided in this GitHub repo under `./data/`. These files are also the basis for generating the figures (see below).

# Code
## Install
The code was run with Python 3.8.2. Install the dependencies first by cloning the repo and running
```bash
pip install -r requirements.txt
```

## Figures
Genearte the figures by first changing the directory and then running the script. For example:
```bash
cd fig4
python fig4e.py
```
This generates a new figure png and pdf in a subfolder `./plots/fig4e/`.

# References
## Categories
In this work we made use of a Machine Learning model to predict user categories. You can download the model from [this GitHub repo](https://github.com/digitalepidemiologylab/experts-covid19-twitter).

## Local-geocode
In order to enrich geo-information of Twitter data we used the library [local-geocode](https://github.com/mar-muel/local-geocode).

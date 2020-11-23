## Notebooks
In the IPython's notebooks we show how we analyzed the data starting from the raw data. 
In particular, `community_detection.ipynb` shows how we built the retweet network and assigned the users to a community. 
This particular task is computationally intensive. For this, we run the notebooks onto a machine with 252GB of RAM and 32 cores. 

Additionally, in order to run the notebooks, one needs to:
1. Collect the Tweets that we listed on Zenodo.  
2. Build a Pandas dataframe having as columns the values we assigned to the variable `col_names` of `notebooks/community_detection.ipynb`.
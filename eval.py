import scanpy as sc
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import matplotlib.pylab as plt


def load_data(link1, link2):
    '''load data'''

    train_data = scanpy.read_h5ad(link1)
    test_data = scanpy.read_h5ad(link2)

    return train_data, test_data

def get_obs_and_vars(anndata):
    ''' return the number of observations and variables'''
    n_obs = anndata.n_obs
    n_vars = anndata.n_vars

    return n_obs, n_vars

def data_to_array(train_data, test_data):
    '''access preprocessed and raw data and convert it to arrays'''
    train_norm = train_data.X.toarray()
    train_raw = train_data.layers['counts'].toarray()
    test_norm = test_data.X.toarray()
    test_raw = test_data.layers['counts'].toarray()

    return train_norm, train_raw, test_norm, test_raw

def clip(train_norm, train_raw, test_norm, test_raw, max):
    ''''clip the arrays of preprocessed and raw data'''
    train_norm_clipped = np.clip(train_norm, 0, max)
    train_raw_clipped = np.clip(train_raw, 0, max)
    test_norm_clipped = np.clip(test_norm, 0, max)
    test_raw_clipped = np.clip(test_raw, 0, max)

    return train_norm_clipped, train_raw_clipped, test_norm_clipped, test_raw_clipped


def hists(train_norm_clipped, train_raw_clipped, test_norm_clipped, test_raw_clipped):
    ''' creates histograms for (clipped) preprocessed and raw train data and (clipped) preprocessed and raw test data
    
    (it doesn't have to be clipped data but I just wrote it here to point out that I used in my project clipped data)'''

    fig, axes = plt.subplots(2, 2, figsize = (12, 12))
    label_setter = np.vectorize(lambda ax: [ax.set_xlabel('Value'), 
                                    ax.set_ylabel('Frequency')])
    axes[0,0].hist(train_norm_clipped.reshape(-1))
    axes[0,0].set_title('Preprocessed train data')
    axes[0,1].hist(train_raw_clipped.reshape(-1))
    axes[0,1].set_title('Raw train data')
    axes[1,0].hist(test_norm_clipped.reshape(-1))
    axes[1,0].set_title('Preprocessed test data')
    axes[1,1].hist(test_raw_clipped.reshape(-1))
    axes[1,1].set_title('Raw test data')

    return fig, axes

 

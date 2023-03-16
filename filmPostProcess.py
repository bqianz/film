import numpy as np
import pandas as pd

def display_table(meta_params, scores):
    n_classes = meta_params['n_classes']
    df = np.reshape(scores, [n_classes,4], order ='F')
    df = pd.DataFrame(df)
    
    # df.style.set_table_attributes("style='display:inline'").set_caption(mode)
    
    df.columns = ['precision', 'recall', 'f1', 'support']
    # df.index = ['unfrozen', 'frozen']
    # df.index = ['Visible ice', 'No visible ice']
    
    display(df)
    
def display_results(meta_params, results):
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    
    print("mean")
    display_table(meta_params, mean[1:])
    
    print("std")
    display_table(meta_params, std[1:])
    
    print("Accuracy mean: {}, std: {}".format(mean[0], std[0]))
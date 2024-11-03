import pandas as pd
import numpy as np
import math
import torch.nn as nn
import torch 
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score




def Multiclass_classification_metrices(y_true, y_pred, num_classes):

    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC :- 
    auc = []
    for i in range(num_classes):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = [1 if pred == i else 0 for pred in y_pred]
        try:
            auc_class = roc_auc_score(y_true_class, y_pred_class)
        except ValueError as e:
            print("ValueError occurred:", e)
            auc_class = 0
        auc.append(auc_class)
    macro_AUC = sum(auc)/num_classes

    return macro_AUC, accuracy








def calculating_prob_mass(df, label):
    
    DF = df
   
    df = DF[DF['label'] == label].iloc[:, 2:].values

    # calculating avg prob mass for each feature(0-767) for each class(0 - 7) :-

    prob_mass_per_value_sum = [0 for i in range(df.shape[0])]
    for i in range(df.shape[1]):
        values = df[:, i]
        hist_values, bin_edges = np.histogram(values, bins=3, density=False)
        prob_mass = hist_values / np.sum(hist_values)
        bin_indices = np.digitize(values, bins=bin_edges, right=False)
        bin_indices = np.clip(bin_indices - 1, 0, len(prob_mass) - 1)
        prob_mass_per_value = prob_mass[bin_indices]
        prob_mass_per_value_sum += prob_mass_per_value
    
    prob_mass_avg = prob_mass_per_value_sum/768

    values = prob_mass_avg
    hist_values, bin_edges = np.histogram(values, bins=2, density=False)
    prob_mass = hist_values / np.sum(hist_values)
    bin_indices = np.digitize(values, bins=bin_edges, right=False)
    bin_indices = np.clip(bin_indices - 1, 0, len(prob_mass) - 1)
    prob_mass_per_value = prob_mass[bin_indices]

    unique_values, _ =  np.unique(prob_mass_per_value, return_counts=True)

    transformed_arr = np.where(prob_mass_per_value == min(unique_values), 0, 1)


    dic = {
    'prob_mass' : transformed_arr
    }  

    df_prob_mass = pd.DataFrame(dic)

    df = DF[DF['label'] == label].reset_index(drop=True)

    df_concat =  pd.concat([df_prob_mass, df], axis=1)

    df_concat_low_density_regions = df_concat[df_concat['prob_mass'] == 0].reset_index(drop=True)

    df_concat_high_density_regions = df_concat[df_concat['prob_mass'] == 1].reset_index(drop=True)

    return df_concat_low_density_regions, df_concat_high_density_regions
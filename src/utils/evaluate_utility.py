import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from .train_utility import PearsonCC

def load_ckp(checkpoint_fpath, model, device):
    checkpoint = torch.load(checkpoint_fpath)#,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['epoch']

def accuracy(y_pred, y_test):
    num_elements = int(len(y_pred))
    sorted_pred = np.argsort(y_pred)
    sorted_truth = np.argsort(y_test)

    lowest_pred = sorted_pred[:int(0.2 * num_elements)]
    lowest_truth = sorted_truth[:int(0.2 * num_elements)]

    highest_pred = sorted_pred[int(0.8 * num_elements):]
    highest_truth = sorted_truth[int(0.8 * num_elements):]

    accuracy_low = len(np.intersect1d(lowest_pred,lowest_truth))/len(lowest_pred)
    accuracy_high = len(np.intersect1d(highest_pred,highest_truth))/len(lowest_pred)
    return (accuracy_low + accuracy_high)*100/2

def test(model, test_loader, device, train_index, Na_ind):
    sample_times = [10, 100, 1000, 10000, 50000, 100000, 500000, 1000000]
    propensities = []
    pred_propensities = []
    if train_index in ("all", "all-10"):
        time_index = []
        
    for data in test_loader:
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            propensities.append(data.y[Na_ind].T.cpu().numpy())
            pred_propensities.append(model(data)[Na_ind].cpu().numpy())
            if train_index in ("all", "all-10"):
                time_index.append(data.time_features[0].cpu().numpy())
                
    if train_index in ("all", "all-10"):
        unique_time = np.unique(time_index)
        PCC = []
        pred_all = []
        prop_all = []
        for j in unique_time:
            pred_j = np.stack([np.array(pred_propensities[i].flatten()) for i in np.where(np.array(time_index)==j)[0]]).flatten()
            prop_j = np.stack([np.array(propensities[i].flatten()) for i in np.where(np.array(time_index)==j)[0]]).flatten()       
            PCC.append(PearsonCC(prop_j,pred_j))
            pred_all.append(pred_j)
            prop_all.append(prop_j)
        time = sample_times
        
    else:    
        prop_all = np.array(propensities).flatten()
        pred_all = np.array(pred_propensities).flatten()
        PCC = PearsonCC(prop_all,pred_all)
        time = sample_times[train_index]
    
    return prop_all, pred_all, PCC, time

def eval_plot(y_pred, y_test, pcc, ax=None):
    if ax is None:
        ax = plt.gca()
        
    x_max = np.max(y_test)*1.1
    y_max = np.max(y_pred)*1.1
    tot_max = np.max([x_max,y_max])

    data , x_e, y_e = np.histogram2d(y_pred,y_test, bins = 20, density = True )
    z = interpn(( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([y_pred,y_test]).T , method = "splinef2d", bounds_error = False)
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    y_pred,y_test, z = y_pred[idx], y_test[idx], z[idx]
        
    ep = ax.scatter(y_test, y_pred, c=z, s=1, rasterized=True)
    ax.plot([-1,100],[-1,100], "k")
    ax.set_xlim(0,tot_max)
    ax.set_ylim(0,tot_max)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_title(r"$\rho$ = " +  f"{'%.2f' % pcc}", loc="left", x = 0.05, y = 0.87, fontsize=10)
    return ep
    
def density_plot(y_pred, y_test, ax=None):
    if ax is None:
        ax = plt.gca()
        
    x_max = np.max(y_test)*1.2
    y_max = np.max(y_pred)*1.2
    tot_max = np.max([x_max,y_max])
    test_his = np.histogram(y_test, bins = 100, range = (0,tot_max), density = True)
    pred_his = np.histogram(y_pred, bins = 100, range = (0,tot_max), density = True)           
    
    dp = ax.plot(test_his[1][1:]-((test_his[1][0] - test_his[1][1])/2), test_his[0], label = "Ground_Truth")             
    dp = ax.plot(pred_his[1][1:]-((pred_his[1][0] - pred_his[1][1])/2), pred_his[0], label = "Predicted")
    ax.set_xlabel("$\Delta$r$_i$(t) (Å)")
    ax.set_ylabel("Density")
    return dp
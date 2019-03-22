import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def plot_learning_curve(history_history, model_name, ylim=None, xlim=None):
    plt.figure()
    plt.plot(np.sqrt(history_history['loss'])*1e3, label='learning')
    plt.plot(np.sqrt(history_history['val_loss'])*1e3, label='validation')
    
    plt.ylabel('RMSE [mm]')
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.xlabel('epoch')
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.legend()
    plt.grid()
    
    plt.title(model_name)
    plt.tight_layout()
    plt.show()
    
    return None


def plot_results(model, x, x_type, data, random=True, member=None, time=None, n_train=40, n_val=5, n_test=None):
     
    if x_type == 'train':
        original = data[0:n_train,:,:,:]
    elif x_type == 'val':
        original = data[n_train:n_train+n_val,:,:,:]
    elif x_type == 'test':
        original = data[n_train+n_val:n_train+n_val+n_test,:,:,:]

    if random:
        i = np.random.randint(x.shape[0])
        i_member = np.unravel_index(i, (original.shape[0],original.shape[1]))[0]
        i_time = np.unravel_index(i, (original.shape[0],original.shape[1]))[1]
        
        original = original[i_member, i_time]
        true = data[50,i_time]

    elif random == False:
        i_member = np.argwhere(original.indexes['member'].values == member).item(0)
        i_time = np.argwhere(original.indexes['time'].values == np.datetime64(time)).item(0)
        i = np.ravel_multi_index(np.array([i_member,i_time]), (original.shape[0],original.shape[1]))
        
        original = original.sel(member=member, time=time)
        true = data.sel(member='emean', time=time)
        
    pred = true.copy()
    pred.values = (model.predict(x[i:i+1])).reshape(data.shape[2],data.shape[3])

    concat = xr.concat([original, true, pred-true, pred], 
                       pd.Index(['member = ' + original.member.item(0), 'emean true', 'emean predicted-true', 'emean predicted']))
    concat = concat*1e2
    concat.name = '[cm]'
        
    p = concat.plot.pcolormesh(x='nav_lon', y='nav_lat', transform=ccrs.PlateCarree(), col='concat_dim', col_wrap=2, robust=True,
                               vmin=-10, vmax=10, subplot_kws={'projection': ccrs.Mercator()}, cmap='RdBu_r', figsize=(8,8))
    
    for i, ax in enumerate(p.axes.flat):
        ax.set_title(concat.concat_dim.values[i])
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        if i in [0,1]:
            gl.xlabels_bottom = False
        if i in [1,3]:
            gl.ylabels_left = False
            
        gl.ylocator = mticker.FixedLocator(range(-34,24,2))
        gl.xlocator = mticker.FixedLocator(range(0,12,2))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    plt.suptitle(model.name + ' / time = ' + str(concat.time_counter.values.astype('M8[s]')))
#     plt.savefig('../result.png')
    plt.show()
    
    return None
import psutil
import xarray as xr

def print_memory_usage():
    print('\nMemory usage: '+str(psutil.virtual_memory().percent)+'%')
    
def get_data():
    data_path = '../data/sla_5d/south_atlantic/agulhas_eddies_48x40'

    # Group all data in one dataset (need dask to be performed) and get the variable sea level anomalies
    # The ensemble mean is the last value of the dimension 'member'

    # Note: xarray use lazy load, so until not loaded it is not stored into the memory

    sla = xr.open_mfdataset(data_path+'/*.nc').sla
    print(sla)
    print_memory_usage()
    
    return sla
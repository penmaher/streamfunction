import numpy as np
import pdb
from scipy.integrate import cumtrapz

__author__ = 'Penelope Maher'

'''
   See text book: P141-142 of global physical climatology by Hartmann 

   [v] = (g/(2 pi a cos (lat))) (partial Phi/ partial p) 
   [omega] = (-g/(2 pi a^2 cos (lat))) (partial Phi/ partial lat) 

   integrate [v] wrt pressure

   Psi = ( (2 pi a cos(lat)) /g) integrate [v] dp 

   and [omega] wrt lat

   Psi = ( (-2 pi a^2 cos(lat)) /g) integrate [omega] dlat 

   Psi units -> Pa m s = kg/(ms^2) * ms = kg /s
                and taking into account the scale factor kg/s 1x10^10
'''

def cal_stream_fn_trap_rule(data, pres_var_name, name_flag=None):
    '''Calculates the mass stream function. 
       Data is a dictionary containing masked zonal-mean arrays of omega and vcomp
       Pressure can be in hPa or Pa.
       Return: a masked numpy array with units x10^10 kg/s
 
      When to use cal_stream_fn_trap_rule in StreamFunction_trapz.py?
           When data is not masked as it is faster.
      When to use cal_stream_fn_riemann?
           When data is masked as use cal_stream_fn_trap_rule does not handle 
           missing data properly.
    '''
  
    #constants
    grav=9.81 #m/s
    rad_earth=6.371e6 #m
    psi_scale_factor = 1e10

    #data properties
    input_dtype =  data['omega'].dtype
    fill_value = data['omega']._fill_value

    #convert degree to rad
    lat_rad=np.radians(data['lat'], dtype=input_dtype)

    #check data
    assert isinstance(data['omega'],np.ma.core.MaskedArray)  == True , 'Data is not masked' 
    assert len(data['omega'].shape)  == 3, 'Data shape is not 3D, investigate' 

    if data[pres_var_name].max() > 1100:
        #data is already in Pa
        pres_factor = 1.
    else:
        #data in hPa, convert to Pa
        pres_factor = 100.

    #integrate psi wrt latitude
    int_lat_omega  = cumtrapz(np.cos(lat_rad) * data['omega'], x=lat_rad, axis=-1, initial=0)
    psi_lat= ((-2.*np.pi*rad_earth*rad_earth)/grav) * int_lat_omega /psi_scale_factor
    #mask where omega was masked
    psi_lat = np.ma.masked_array(psi_lat, data['omega'].mask)

    #integrate psi wrt pressure
    int_v  = cumtrapz(data['vcomp'], x=data[pres_var_name]*pres_factor, axis=1, initial=0)
    psi_pres = ((2.*np.pi*rad_earth*np.cos(lat_rad)) /grav ) * int_v/psi_scale_factor
    #mask where vcomp was masked
    psi_pres = np.ma.masked_array(psi_pres, data['vcomp'].mask)

    #take the mean over the stream functions
    psi = np.mean( np.array([psi_lat,psi_pres]), axis=0, dtype=input_dtype)
    psi_mask = np.logical_or(data['vcomp'].mask, data['omega'].mask)

    #mask if both omega and vcomp are both masked, else not masked
    psi = np.ma.masked_array(psi, psi_mask, fill_value=fill_value)

    make_plot=False
    if make_plot:
        plot_stream_function(psi_lat,psi_pres, psi, data['lat'], 
                             data[pres_var_name]*pres_factor, data['omega'], data['vcomp'], name_flag)

    return psi

def plot_stream_function(psi_lat, psi_pres, psi, lat, pres, omega, vwind, name_flag):

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cmap=plt.get_cmap('RdBu_r')
    names = ['psi_lat','psi_pres','psi', 'omega', 'vwind']
    data_list = [psi_lat,psi_pres, psi, omega, vwind]

    for count, data in enumerate(data_list):
        if count ==3:
            bounds=np.arange(-0.05,0.05,.001)
            label_format = '%.2f'
        elif count ==4:
            bounds=np.arange(-2.5,2.6,0.5)
            label_format = '%.2f'
        else:
            bounds=np.arange(-10,10.1,1)
            label_format = '%d'

        plot_single(data, pres, lat, bounds, names[count], label_format, cmap, name_flag)

def plot_single(data, pres, lat, bounds, name, label_format, cmap=None, name_flag=None):

    #incoming pressure is in Pa

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if cmap== None:
        cmap = plt.get_cmap('RdBu_r')

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    
    data_plot = np.ma.mean(data, axis=0)
    filled_map = ax.pcolormesh(lat, pres/100., data_plot,
                               cmap=cmap,norm=norm, shading='nearest')
    ax.set_ylim(1000,100)
    ax_cb=fig.add_axes([0.15, 0.05, 0.8, 0.03]) #[xloc, yloc, width, height]
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm, ticks=bounds,
                                     orientation='horizontal',format=label_format,
                                     extend='both')    
    filename='testing_psi_{0}{1}.eps'.format(name, name_flag)
    plt.savefig(filename)
    plt.show()



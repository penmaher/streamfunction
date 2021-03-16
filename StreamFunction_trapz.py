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

def cal_stream_fn_trap_rule(data, pres_var_name):
    '''Calculates the mass stream function. 
       Data is a dictionary containing masked zonal-mean arrays of omega and vcomp
       Pressure can be in hPa or Pa.
    '''
  
    grav=9.81 #m/s
    rad_earth=6.371e6 #m
    lat_rad=np.radians(data['lat'])
    psi_scale_factor = 1e10

    omega = data['omega']
    vcomp = data['vcomp']
    assert isinstance(omega,np.ma.core.MaskedArray)  == True , 'Data is not masked' 
    assert len(omega.shape)  == 3, 'Data shape is not 3D, investigate' 

    if data[pres_var_name].max() > 1100:
        #data in Pa
        pres_factor = 1.
    else:
        #data in Pa
        pres_factor = 100.

    #integrate psi wrt latitude
    int_lat_omega_forward  = cumtrapz(np.cos(lat_rad) * omega, x=lat_rad, axis=-1, initial=0)
    int_lat_omega_backward = cumtrapz(np.cos(lat_rad) * omega[:,:,::-1], x=lat_rad[::-1], axis=-1, initial=0)

    psi_lat_frwd = ((-2.*np.pi*rad_earth*rad_earth)/grav) * int_lat_omega_forward /psi_scale_factor
    psi_lat_bkwd = ((-2.*np.pi*rad_earth*rad_earth)/grav) * int_lat_omega_backward/psi_scale_factor

    #mask where omega was masked
    psi_lat_frwd = np.ma.masked_array(psi_lat_frwd, omega.mask)
    psi_lat_bkwd = np.ma.masked_array(psi_lat_bkwd[:,:,::-1], omega.mask)
    pdb.set_trace()
    #integrate psi wrt pressure  both in the forward and backward directions
    int_v_forward  = cumtrapz(vcomp, x=data[pres_var_name]*pres_factor, axis=1, initial=0)
    int_v_backward = cumtrapz(vcomp[:,::-1,:], x=data[pres_var_name][::-1]*pres_factor, axis=1, initial=0)

    psi_pres_frwd = ((2.*np.pi*rad_earth*np.cos(lat_rad)) /grav ) * int_v_forward/psi_scale_factor
    psi_pres_bkwd = ((2.*np.pi*rad_earth*np.cos(lat_rad)) /grav ) * int_v_backward/psi_scale_factor

    #mask where vcomp was masked
    psi_pres_frwd = np.ma.masked_array(psi_pres_frwd, vcomp.mask)
    psi_pres_bkwd = np.ma.masked_array(psi_pres_bkwd[:,::-1,:], vcomp.mask)

    #take the mean over the stream functions
    psi = np.mean( np.array([psi_pres_frwd,psi_pres_bkwd,psi_lat_frwd,psi_lat_bkwd]), axis=0 )

    make_plot=True
    if make_plot:
        plot_stream_function(psi_pres_frwd, psi_pres_bkwd,
                             psi_lat_frwd, psi_lat_bkwd, 
                             psi, data['lat'], data[pres_var_name], omega, vcomp, pres_factor)
    pdb.set_trace()
    return psi

def plot_stream_function(psi_pres_frwd, psi_pres_bkwd,
                         psi_lat_frwd, psi_lat_bkwd, 
                         psi, lat, pres, omega, vwind, pres_factor):

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    bounds=np.arange(-10,10,1)
    cmap=plt.get_cmap('RdBu_r')
    names = ['psi_pres_frwd','psi_pres_bkwd','psi_lat_frwd','psi_lat_bkwd',
             'psi', 'omega', 'vwind']
    data_list = [psi_pres_frwd,psi_pres_bkwd,psi_lat_frwd,psi_lat_bkwd,
                 psi]#, omega, vwind]

    for count, data in enumerate([psi_lat_bkwd]):#(data_list):
        if count ==5:
            bounds=np.arange(-0.1,0.11,.01)
            label_format = '%.2f'
        elif count ==6:
            bounds=np.arange(-2.5,2.6,0.5)
            label_format = '%.2f'
        else:
            bounds=np.arange(-10,10,1)
            label_format = '%d'

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        for t in range(data.shape[0]):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
            #data_plot = np.mean(data, axis=0)
            data_plot = data[t,:,:]
            filled_map = ax.pcolormesh(lat, (pres/pres_factor)/100., data_plot,
                                       cmap=cmap,norm=norm, shading='nearest')
            ax.set_ylim(1000,100)
            ax_cb=fig.add_axes([0.15, 0.05, 0.8, 0.03]) #[xloc, yloc, width, height]
            cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm, ticks=bounds,
                                             orientation='horizontal',format=label_format,
                                             extend='both')
            filename='testing_psi_{0}_{1}.eps'.format(names[count],t)
            plt.savefig(filename)
            plt.show()
    pdb.set_trace()



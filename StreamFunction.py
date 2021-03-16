import numpy as np
import pdb

__author__ = 'Penelope Maher'

#Code Description: 
#   eg see text book: P141-142 of global physical climatology by Hartmann 

#   Psi = ( (2 pi a cos(lat)) /g) integrate ^P _0 [v] dp 
#   [v] = (g/(2 pi a cos (lat))) (partial Phi/ partial p) 
#   [omega] = (-g/(2 pi a^2 cos (lat))) (partial Phi/ partial lat) 
#   Psi units -> Pa m (s2/m) (m/s) = Pa m/s = kg/(ms2) (m/s) = kg /s 

'''Please note this code is no longer in use. It has been replaced by the 
   file StreamFunction_trapz. Evaluation of the two methods reveals that this
   method is not performing as expected.
'''

def cal_stream_fn(data, pres_var_name):
  '''Calculates the mass stream function. 
     Data is a dictionary containing zonal-mean arrays of omega and vcomp
     The dims are named: time, lat, and the input pres_var_name (hPa).
     Method: Apply the left and right Riemann Sum to approximate the area
     (i.e. the trapezoidal rule).'''
  
  grav=9.81 #m/s
  rad_earth=6.371e6 #m
  lat_rad=np.radians(data['lat'])
  fn_const_dlat = (-2*(np.pi)*(rad_earth*rad_earth))/grav  
  psi_scale_factor = 1e10

  #integrate right to left [omega] and then integrate left to right [omega]
  
  psi_p = np.zeros((len(data['time']),len(data[pres_var_name]),len(data['lat'])))
  psi_lat = np.zeros((len(data['time']),len(data[pres_var_name]),len(data['lat'])))
  psi_p_rev = np.zeros((len(data['time']),len(data[pres_var_name]),len(data['lat'])))
  psi_lat_rev = np.zeros((len(data['time']),len(data[pres_var_name]),len(data['lat'])))
  omega = data['omega']
  vcomp = data['vcomp']
  assert len(omega.shape)  == 3, 'Data shape is not 3D, investigate' 


  if data[pres_var_name].max() > 1100:
    #data in Pa
    pres_factor = 1.
  else:
    #data in Pa
    pres_factor = 100.

  
  print('... Integrate first direction')
  
  for t in range(len(data['time'])): 
    for p in range(len(data[pres_var_name]) -1): 
      psi_lat[t,p,0]=0
      psi_lat_rev[t,p,0]=0

      for l in range(len( data['lat']) -1):

        w_val = np.array([ omega[t,p,l],   omega[t,p+1,l],
                           omega[t,p,l+1], omega[t,p+1,l+1]])
        mask_val   = np.array([ omega.mask[t,p,l], omega.mask[t,p+1,l],
                                omega.mask[t,p,l+1], omega.mask[t,p+1,l+1] ])
        w_val = np.ma.masked_array(w_val, mask_val)

        dlat=lat_rad[l] - lat_rad[l+1]
        #w_val_mean = 0.25*(np.nansum(w_val)
        w_val_mean = np.mean(w_val)

        psi_lat[t,p,l+1] = (psi_lat[t,p,l] + dlat * fn_const_dlat * np.cos(lat_rad[l+1]) * w_val_mean ) /psi_scale_factor
        num_missing = np.where(mask_val == True)[0]

      for l in range(len(data['lat'])-1,0,-1 ): 
        w_val_rev = np.array([omega[t,p,l],  omega[t,p+1,l-1],
                              omega[t,p,l-1],omega[t,p+1,l]])

        mask_val = np.array([omega.mask[t,p,l],  omega.mask[t,p+1,l-1],
                             omega.mask[t,p,l-1],omega.mask[t,p+1,l]])
        w_val_rev = np.ma.masked_array(w_val_rev, mask_val)

        dlat=lat_rad[l] - lat_rad[l-1]   #positive
        w_val_rev_mean = np.mean(w_val_rev)#0.25*(np.nansum(w_val_rev))
        psi_lat_rev[t,p,l-1] = (psi_lat_rev[t,p,l]+ dlat*fn_const_dlat*np.cos(lat_rad[l-1])* w_val_rev_mean)/psi_scale_factor

  #integrate  bottom to top and then top to bottom [v] 
  print('... Integrate second direction')  
  for t in range(len(data['time'])): 
    for l in range(len(data['lat']) -1):
      fn_const_dp = (2*np.pi*rad_earth*np.cos(lat_rad[l+1]))/grav
      psi_p[t,0,l] = 0
      psi_p_rev[t,0,l] = 0
      
      for p in range(len(data[pres_var_name])-1):

        v_val=np.array([vcomp[t,p,l],  vcomp[t,p,l+1],
                        vcomp[t,p+1,l],vcomp[t,p+1,l+1]])

        mask_val=np.array([vcomp.mask[t,p,l],  vcomp.mask[t,p,l+1],
                           vcomp.mask[t,p+1,l],vcomp.mask[t,p+1,l+1]])

        v_val = np.ma.masked_array(v_val, mask_val)

        dp = (data[pres_var_name][p]-data[pres_var_name][p+1])*pres_factor    #in Pa and negative
        v_val_mean = np.mean(v_val) #0.25*(np.nansum(v_val))
        psi_p[t,p+1,l]=(psi_p[t,p,l]+dp * fn_const_dp * v_val_mean) /psi_scale_factor
  
      for p in range(len(data[pres_var_name])-1,0,-1 ):
        v_val_rev=np.array([vcomp[t,p-1,l],vcomp[t,p-1,l+1],
                            vcomp[t,p,l+1],vcomp[t,p,l]])
        mask_val=np.array([vcomp.mask[t,p-1,l],vcomp.mask[t,p-1,l+1],
                           vcomp.mask[t,p,l+1],vcomp.mask[t,p,l]])
        v_val_rev = np.ma.masked_array(v_val_rev, mask_val)

        dp=(data[pres_var_name][p] - data[pres_var_name][p-1])*pres_factor   #in Pa and positive
        v_val_rev_mean = np.mean(v_val_rev)#0.25*(np.nansum(v_val_rev))
        psi_p_rev[t,p-1,l] = (psi_p_rev[t,p,l] + dp*fn_const_dp* v_val_rev_mean)/psi_scale_factor

  print('  Average the stream functions')       
  psi_final = np.zeros([4,len(data['time']),len(data[pres_var_name]),len(data['lat'])])
  psi_final[0,:,:,:] = psi_lat
  psi_final[1,:,:,:] = psi_p
  psi_final[2,:,:,:] = psi_p_rev
  psi_final[3,:,:,:] = psi_lat_rev
  
  #take the mean over the four stream funstions
  psi = np.mean(psi_final,axis=0)

  make_plot=False
  if make_plot:
        plot_stream_function(psi_p, psi_p_rev,
                             psi_lat, psi_lat_rev, 
                             psi, data['lat'], data[pres_var_name], omega, vcomp, pres_factor)
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

    for count, data in enumerate(data_list):
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

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        data_plot = np.mean(data, axis=0)
        filled_map = ax.pcolormesh(lat, (pres/pres_factor)/100., data_plot,
                                   cmap=cmap,norm=norm, shading='nearest')
        ax.set_ylim(1000,100)
        ax_cb=fig.add_axes([0.15, 0.05, 0.8, 0.03]) #[xloc, yloc, width, height]
        cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm, ticks=bounds,
                                         orientation='horizontal',format=label_format,
                                         extend='both')
        filename='testing_psi_{0}_old_code.eps'.format(names[count])
        plt.savefig(filename)
        plt.show()


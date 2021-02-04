import numpy as np
import pdb

__author__ = 'Penelope Maher'

#Code Description: 
#   eg see text book: P141-142 of global physical climatology by Hartmann 

#   Psi = ( (2 pi a cos(lat)) /g) integrate ^P _0 [v] dp 
#   [v] = (g/(2 pi a cos (lat))) (partial Phi/ partial p) 
#   [omega] = (-g/(2 pi a^2 cos (lat))) (partial Phi/ partial lat) 
#   Psi units -> Pa m (s2/m) (m/s) = Pa m/s = kg/(ms2) (m/s) = kg /s 

def cal_stream_fn(data):
  'Calculates the mass stream function. \
  Data is a dictionary containing masked arrays of omega, v wind, lat, time and pressure (hPa).'

  
  grav=9.80665 #m/s
  rad_earth=6378100.0 #m
  lat_rad=np.radians(data['lat'])
  fn_const_dlat = (-2*(np.pi)*(rad_earth*rad_earth))/grav  

  #integrate right to left [omega] and then integrate left to right [omega]
  
  psi_p = np.zeros((len(data['time']),len(data['pfull']),len(data['lat'])))
  psi_lat = np.zeros((len(data['time']),len(data['pfull']),len(data['lat'])))
  psi_p_rev = np.zeros((len(data['time']),len(data['pfull']),len(data['lat'])))
  psi_lat_rev = np.zeros((len(data['time']),len(data['pfull']),len(data['lat'])))
  omega = data['omega']
  vcomp = data['vcomp']
  assert isinstance(omega,np.ma.core.MaskedArray)  == True , 'Data is not masked' 
  
  print('... Integrate first direction')
  
  for t in range(len(data['time'])): 
    for p in range(len(data['pfull']) -1): 
      psi_lat[t,p,0]=0
      psi_lat_rev[t,p,0]=0

      for l in range(len( data['lat']) -1):
        w_val = np.array([omega[t,p,l],omega[t,p+1,l],omega[t,p,l+1],omega[t,p+1,l+1]])

        if np.ma.count(w_val) == 4:
          dlat=lat_rad[l] - lat_rad[l+1]
          psi_lat[t,p,l+1] = psi_lat[t,p,l] + dlat * fn_const_dlat * np.cos(lat_rad[l+1]) * 0.25*(np.nansum(w_val))
        else:  
          print('Address missing values if required')
          pdb.set_trace()

      for l in range(len(data['lat'])-1,0,-1 ): 
        w_val_rev = np.array([omega[t,p,l],omega[t,p+1,l-1],omega[t,p,l-1],omega[t,p+1,l]])
        if np.ma.count(w_val_rev) == 4:
          dlat=lat_rad[l] - lat_rad[l-1]   #positive
          psi_lat_rev[t,p,l-1] = psi_lat_rev[t,p,l]+ dlat*fn_const_dlat*np.cos(lat_rad[l-1])* 0.25*(np.nansum(w_val_rev))
        else:
          print('Address missing values if required')
          pdb.set_trace()
    
  #integrate  bottom to top and then top to bottom [v] 
  print('... Integrate second direction')  
  for t in range(len(data['time'])): 
    for l in range(len(data['lat']) -1):
      fn_const_dp = (2*np.pi*rad_earth*np.cos(lat_rad[l+1]))/grav
      psi_p[t,0,l] = 0
      psi_p_rev[t,0,l] = 0
      
      for p in range(len(data['pfull'])-1):
        v_val=np.array([vcomp[t,p,l],vcomp[t,p,l+1],vcomp[t,p+1,l],vcomp[t,p+1,l+1]])
        if np.ma.count(v_val) == 4:
          dp = (data['pfull'][p]-data['pfull'][p+1])*100   #in Pa and negative
          psi_p[t,p+1,l]=psi_p[t,p,l]+dp * fn_const_dp * 0.25*(np.nansum(v_val))
        else: 
          print('Address missing values if required')
          pdb.set_trace()  
  
      for p in range(len(data['pfull'])-1,0,-1 ):
        v_val_rev=np.array([vcomp[t,p-1,l],vcomp[t,p-1,l+1],vcomp[t,p,l+1],vcomp[t,p,l]])
        if np.ma.count(v_val_rev) == 4:
          dp=(data['pfull'][p] - data['pfull'][p-1])*100   #in Pa and positive
          psi_p_rev[t,p-1,l] = psi_p_rev[t,p,l] + dp*fn_const_dp* 0.25*(np.nansum(v_val_rev))
        else:  
          print('Address missing values if required')
          pdb.set_trace()
    
  print('  Average the stream functions')       
  psi_final = np.zeros([4,len(data['time']),len(data['pfull']),len(data['lat'])])
  psi_final[0,:,:,:] = psi_lat
  psi_final[1,:,:,:] = psi_p
  psi_final[2,:,:,:] = psi_p_rev
  psi_final[3,:,:,:] = psi_lat_rev
  
  #take the mean over the four stream funstions
  psi = np.mean(psi_final,axis=0)
  

  return psi

import numpy as np
import pdb

from .StreamFunction_trapz import plot_stream_function, plot_single

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

def set_fill_value(data, fill_value):
    #note this returns a unmasked array!
    for var in ['omega', 'vcomp']:
        #mask  = data[var].mask
        data[var] = data[var].filled(fill_value=-333.)
        #data[var] = np.ma.masked_array(data[var], mask fill_value=-333.)
    return data

def data_ordering(data)
    #if SH pole is not the first element, then reverse direction
    if np.where(data['lat'] == data['lat'].min())[0] !=0:
        lat = data['lat'][::-1].copy()
        omega = data['omega'][:,:,::-1].copy()
        vcomp = data['vcomp'][:,:,::-1].copy()
    else:
        lat = data['lat'].copy()
        omega = data['omega'].copy()
        vcomp = data['vcomp'].copy()

    # the pressure integral is from surface up.
    if np.where(data[pres_var_name] == data[pres_var_name].min())[0] ==0:
        pres = data[pres_var_name][::-1]*pres_factor
        omega = data['omega'][:,::-1,:].copy()
        vcomp = data['vcomp'][:,::-1,:].copy()
    else:
        pres  = data[pres_var_name]*pres_factor 
        omega = data['omega'].copy()
        vcomp = data['vcomp'].copy()
    return lat, omega, vcomp

def mask_array(data_shape, fill_value):
    #create a new array without missing values but is masked
    data = np.zeros(data_shape)
    mask = np.zeros(data_shape) #all false equivalent           
    data = np.ma.masked_array(data, mask,  fill_value=fill_value)
    return data


def cal_stream_fn_riemann(data, pres_var_name, name_flag=None, testing=False):
    '''Calculates the mass stream function. 
       Data is a dictionary containing masked zonal-mean arrays of omega and vcomp
       Pressure can be in hPa or Pa.
       Return: a masked numpy array with units x10^10 kg/s

       When to use cal_stream_fn_trap_rule in StreamFunction_trapz.py?
           When data is not masked as it is faster.

       When to use cal_stream_fn_riemann?
           When data is masked as use cal_stream_fn_trap_rule does not handle 
           missing data properly.

       Note: in other peoples code (e.g. TrapD by Ori Admas and in Martin Juckers
           (https://github.com/mjucker/aostools/blob/master/climate.py), psi is
           only computed using vcomp and note the omega component.
    '''

    #https://www.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
  
    #constants
    grav=9.81 #m/s
    rad_earth=6.371e6 #m
    psi_scale_factor = 1e10

    #data properties
    input_dtype =  data['omega'].dtype
    fill_value = data['omega'].fill_value

    #check data
    assert isinstance(data['omega'],np.ma.core.MaskedArray)  == True , 'Data is not masked' 
    assert len(data['omega'].shape)  == 3, 'Data shape is not 3D, investigate' 

    #test if data is in Pa or hPa - multiply data by 100 if in hPa
    if data[pres_var_name].max() > 1100:        
        pres_factor = 1.
    else:
        pres_factor = 100.

    #ensure data is from SH to NH and surface to TOA. 
    lat, omega, vcomp = data_ordering(data)    

    #degree to radians
    lat_rad=np.radians(lat, dtype=input_dtype)

    #If cumtrap is used, then need to set the fill value to zero
    data_unmasked = set_fill_value(data, 0.0)

    #new masked data arrays
    left_riemann_sum_lat   = mask_array(omega.shape, fill_value)
    right_riemann_sum_lat  = mask_array(omega.shape, fill_value)
    left_riemann_sum_pres  = mask_array(omega.shape, fill_value)
    right_riemann_sum_pres = mask_array(omega.shape, fill_value)

    #start the loop at 1 (not 0) and integrate cos(lat) x omega from 90S to 90N
    for count in range(1, len(lat_rad)):
        dlat = lat_rad[count] - lat_rad[count-1]  #dlat > 0 

        #Find the area in the current block and then add to previous sum
        #If the current block is masked, keep the previous left sum. 
        #int_segment = np.cos(lat_rad[count]) * omega[:,:,count-1] * dlat
        #left_riemann_sum_lat[:s,:,count] = np.ma.array(left_riemann_sum_lat[:,:,count-1].data + int_segment.data,
        #                                              mask=list(map(and_,left_riemann_sum_lat.mask[:,:,count-1],int_segment.mask)))

        left_riemann_sum_lat[:,:,count]  = (left_riemann_sum_lat[:,:,count-1] + 
                                            np.cos(lat_rad[count]) * omega[:,:,count-1] * dlat)                                          

        #int_segment = np.cos(lat_rad[count]) * omega[:,:,count] * dlat
        #right_riemann_sum_lat[:,:,count] = np.ma.array(right_riemann_sum_lat[:,:,count-1].data + int_segment.data,
        #                                              mask=list(map(and_,right_riemann_sum_lat.mask[:,:,count-1],int_segment.mask)))

        right_riemann_sum_lat[:,:,count] = (right_riemann_sum_lat[:,:,count-1] +
                                           np.cos(lat_rad[count]) * omega[:,:,count] * dlat)
       
    if testing:
    #if False:
        testing_riemann_sum(vcomp, omega, pres, lat, lat_rad, 'lat',
                            right_riemann_sum_lat, left_riemann_sum_lat)

    for count in range(1,len(pres)):

        dp = (pres[count] - pres[count-1])  #dp<0
        
        left_riemann_sum_pres[:,count,:] = (left_riemann_sum_pres[:,count-1,:]+
                                           vcomp[:,count-1,:] * dp )

        right_riemann_sum_pres[:,count,:] = (right_riemann_sum_pres[:,count-1,:]+
                                            vcomp[:,count,:] * dp )

        
    if testing:
    #if False:
        testing_riemann_sum(vcomp, omega, pres, lat, lat_rad, 'pres',
                           right_riemann_sum_pres, left_riemann_sum_pres)

    pdb.set_trace()

    #psi should be masked if left and right sums are both masked
    mask_psi_lat = np.logical_and(left_riemann_sum_lat.mask,right_riemann_sum_lat.mask)
    mask_psi_pres = np.logical_and(left_riemann_sum_pres.mask,right_riemann_sum_pres.mask)

    riemann_lat_avg = avg_two_masked_arrays(left_riemann_sum_lat, 
                                            right_riemann_sum_lat, 
                                            fill_value,
                                            mask_psi_lat)

    riemann_pres_avg = avg_two_masked_arrays(left_riemann_sum_pres, 
                                             right_riemann_sum_pres, 
                                             fill_value,
                                             mask_psi_pres)            

    #Apply the scaling for the psi intergrals
    psi_lat  = ((-2.*np.pi*rad_earth*rad_earth)/grav) * riemann_lat_avg /psi_scale_factor
    psi_pres = ((2.*np.pi*rad_earth*np.cos(lat_rad))/grav)[np.newaxis,np.newaxis,:] * riemann_pres_avg/psi_scale_factor

    #if False:
    if testing:
        testing_psi(lat, pres, 'pres', psi_pres, riemann_pres_avg)
        testing_psi(lat, pres, 'lat', psi_lat, riemann_lat_avg)

    #psi should be masked if psi_lat and psi_omegs are both masked
    #psi_mask = np.logical_and(vcomp.mask, omega.mask)

    psi = avg_two_masked_arrays(psi_lat, psi_pres, mask_psi_lat, fill_value, mask2 = mask_psi_pres)     


    make_plot=True
    if make_plot:
        plot_stream_function(psi_lat,psi_pres, psi, lat, 
                             pres, omega, vcomp, name_flag)

    #if lat or pressure was reversed, then put back to original form
    if np.where(data['lat'] == data['lat'].min())[0] !=0:
        psi = psi[:,:,::-1].copy()

    if np.where(data[pres_var_name] == data[pres_var_name].min())[0] ==0:
        psi = psi[:,::-1,:].copy()

    pdb.set_trace()


    return psi



def test_for_missing_data(data, lat_start, pres_start):
    
    wh_missing = np.where(data.mask == True)[0]
    data_mask = data.mask
    #if there is missing data, then loop over dims to find its location.
    if len(wh_missing) != 0:
        for lat_dim in range(lat_start, data.shape[2]):
            for pres_dim in range(pres_start, data.shape[1]):                
                for time_dim in range(data.shape[0]):
                    if data_mask[time_dim,pres_dim,lat_dim] == True:
                        print('Missing values to deal with')
                        print(time_dim,pres_dim,lat_dim)
                        pdb.set_trace()


def avg_two_masked_arrays(array1, array2, mask1, fill_value, mask2=None):
    
    assert array1.shape == array2.shape, 'Input data is different shapes. Cant take average' 

    #expand the 3d mask into a 4d mask
    array_dim = [2] + [x for x in array1.shape]
    mask = np.zeros(tuple(array_dim), dtype=bool)
    if mask2 is None:
        mask[:,...] = mask1
    else:
        mask[:,...] = np.logical_and(mask1,mask2)

    #fill the first dim with the two arrays and then take the mean
    data_to_avg = np.array([array1, array2])
    array = np.ma.masked_array(data_to_avg, mask, fill_value=fill_value)
    array_avg = np.ma.mean(array, axis=0)

    return array_avg

def testing_riemann_sum(vcomp, omega, pres, lat, lat_rad, test_type,
                        right_riemann_sum, left_riemann_sum):

    ''' The cumtrapz function does not manage masked data. But can be used
        as a sanity check to make sure integration looks reasonable.'''

    print('Testing riemann sums for ', test_type)

    name_flag = '_fluxum_fixed_sst'
#    name_flag = '_era'

    if test_type == 'lat':
        plot_data = omega
        var_flag = 'omega'
        data_factor = 100
        plot_scale_factor = 1e3
    else:
        plot_data = vcomp
        var_flag = 'vcomp'
        data_factor = 1
        plot_scale_factor = 2e-5

    from scipy.integrate import cumtrapz
    if test_type == 'lat':
        int_val  = cumtrapz((np.cos(lat_rad) * plot_data), x=lat_rad, axis=-1, initial=0)
        #int_val2  = cumulative_trapezoid((np.cos(lat_rad) * plot_data), x=lat_rad, axis=-1, initial=0)
    else:
        int_val  = cumtrapz(plot_data, x=pres, axis=1, initial=0)
        #int_val2  = cumulative_trapezoid(plot_data, x=pres, axis=1, initial=0)

    plot_single(plot_data*data_factor, pres, lat, np.arange(-1,1.1,0.25), var_flag, '%.1f', name_flag=name_flag)

    plot_single(right_riemann_sum*plot_scale_factor, pres, lat, np.arange(-1,1.1,0.25), 'right_'+ test_type, '%.1f', name_flag=name_flag)

    plot_single(left_riemann_sum*plot_scale_factor, pres, lat, np.arange(-1,1.1,0.25), 'left_'+ test_type, '%.1f', name_flag=name_flag)        


    plot_single(int_val*plot_scale_factor, pres, lat,  np.arange(-1,1.1,0.25), 'cumtrapz_'+ test_type, '%.1f', name_flag=name_flag)
    #plot_single(int_val2*plot_scale_factor, pres, lat,  np.arange(-1,1.1,0.25), 'cumtrapz2_'+ test_type, '%.1f', name_flag=name_flag)

    pdb.set_trace()

def testing_psi(lat, pres, test_type,
                psi, riemann_avg):

    print('Testing psi output for ', test_type)

    name_flag = '_fluxum_fixed_sst'

    if test_type == 'lat':
        plot_scale_factor = 1
    else:
        plot_scale_factor = 1e-4
 
    plot_single(psi, pres/100., lat, np.arange(-20,20.2,2), 'psi', '%d', name_flag=name_flag)
    plot_single(riemann_avg*plot_scale_factor, pres/100., lat, np.arange(-1,1.1,0.1), 'riemann_avg_'+test_type, '%.1f', name_flag=name_flag)
    pdb.set_trace()

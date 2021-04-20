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
    '''

    #https://www.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/
  
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

    #if SH pole is the first element, then integration is correct order,
    #else reverse direction
    if np.where(data['lat'] == data['lat'].min())[0] !=0:
        lat_int_factor = -1
    else:
        lat_int_factor =  1

    #new data arrays, not masked at this point
    left_riemann_sum_lat  = np.zeros_like(data['omega'].data)
    right_riemann_sum_lat = np.zeros_like(data['omega'].data)
    left_riemann_sum_pres  = np.zeros_like(data['omega'].data)
    right_riemann_sum_pres = np.zeros_like(data['omega'].data)

    for count in range(1, len(lat_rad)):
        #start the loop at 1 and integrate (cos(lat) x omega)
        dlat =  (lat_rad[count] - lat_rad[count-1])* lat_int_factor  #dlat > 0 
        left_riemann_sum_lat[:,:,count] = (left_riemann_sum_lat[:,:,count-1] + 
                                          np.cos(lat_rad[count]) * 
                                          data['omega'][:,:,count] * dlat)

        right_riemann_sum_lat[:,:,count] = (right_riemann_sum_lat[:,:,count-1] +
                                           np.cos(lat_rad[count]) * 
                                           data['omega'][:,:,count-1] * dlat)
    #if testing:
    if False:
        testing_riemann_sum(data, pres_var_name, pres_factor, lat_rad, 'lat',
                           right_riemann_sum_lat, left_riemann_sum_lat, lat_int_factor)

    # the pressure integral is from surface up. So if the surface is the last element,
    # then add a -1 to the pressure integral.
    if np.where(data[pres_var_name] == data[pres_var_name].max())[0] ==0:
        pres_int_factor =  1
    else:
        pres_int_factor = -1

    for count in range(1,len(data[pres_var_name])):

        dp = (data[pres_var_name][count]*pres_factor - 
              data[pres_var_name][count-1]*pres_factor) * pres_int_factor #dp<0
        
        print(count, '', dp)
        
        left_riemann_sum_pres[:,count,:] = (left_riemann_sum_pres[:,count-1,:]+
                                           data['vcomp'][:,count,:] * dp )

        right_riemann_sum_pres[:,count,:] = (right_riemann_sum_pres[:,count-1,:]+
                                            data['vcomp'][:,count-1,:] * dp )

        
        #data['lat'][25:39]
        #data['vcomp'][:,-1,25:39]
    pdb.set_trace()
    if testing:
    #if False:
        testing_riemann_sum(data, pres_var_name, pres_factor, lat_rad, 'pres',
                           right_riemann_sum_pres, left_riemann_sum_pres, pres_int_factor)
    pdb.set_trace()
    #mask if the left and right riemann sums are wieldly different. 
    #left_riemann_sum_lat,right_riemann_sum_lat =  mask_sums_if_diff(
    #    left_riemann_sum_lat, right_riemann_sum_lat)

    #left_riemann_sum_pres,right_riemann_sum_pres =  mask_sums_if_diff(
    #    left_riemann_sum_pres, right_riemann_sum_pres)

    psi_mask = np.logical_and(data['vcomp'].mask, data['omega'].mask)

    riemann_lat_avg = avg_two_masked_arrays(left_riemann_sum_lat, 
                                            right_riemann_sum_lat, 
                                            psi_mask, 
                                            fill_value)

    riemann_pres_avg = avg_two_masked_arrays(left_riemann_sum_pres, 
                                             right_riemann_sum_pres, 
                                             psi_mask, 
                                             fill_value)            

    #Apply the scaling for the psi intergrals
    psi_lat  = ((-2.*np.pi*rad_earth*rad_earth)/grav) * riemann_lat_avg /psi_scale_factor
    psi_pres = ((2.*np.pi*rad_earth*np.cos(lat_rad))/grav)[np.newaxis,np.newaxis,:] * riemann_pres_avg/psi_scale_factor

    if testing:
        testing_psi(data, pres_var_name, pres_factor, 'pres', psi_pres, riemann_pres_avg)
        testing_psi(data, pres_var_name, pres_factor, 'lat', psi_lat, riemann_lat_avg)

    pdb.set_trace()

    psi = avg_two_masked_arrays(psi_lat, psi_pres, psi_mask, fill_value)     


    #for t in range(len(data['time'])):
    #    wh_large = psi_pres[t,-1,0:33]> 40
    #    if len(np.where(wh_large == True)[0]) != 0:
    #        pdb.set_trace()


    make_plot=True
    if make_plot:
        plot_stream_function(psi_lat,psi_pres, psi, data['lat'], 
                             data[pres_var_name]*pres_factor, data['omega'], data['vcomp'], name_flag)
    pdb.set_trace()
    return psi

def avg_two_masked_arrays(array1, array2, mask_in, fill_value):
    
    assert array1.shape == array2.shape, 'Input data is different shapes. Cant take average' 

    #expand the 3d mask into a 4d mask
    array_dim = [2] + [x for x in array1.shape]
    mask = np.zeros(tuple(array_dim), dtype=bool)
    mask[:,...] = mask_in

    #fill the first dim with the two arrays and then take the mean
    data_to_avg = np.array([array1, array2])
    array = np.ma.masked_array(data_to_avg, mask, fill_value=fill_value)
    array_avg = np.ma.mean(array, axis=0)

    return array_avg

def mask_sums_if_diff(left_sum, right_sum):

    # mask if the difference in the two sums is outside 4 standard deviations 
    # i.e. in the 0.04% of each tail.

    #the variance of the difference is the sum of the variances
    var = np.var(left_sum,axis=0) + np.var(right_sum,axis=0) 

    diff_sum = (left_sum - right_sum)
    diff_sum_tm = np.mean(diff_sum, axis=0)

    #how to mask? The below is a 2d array

    #note the standard deviation is the sqrt of the variance
    pos_tail = np.where( diff_sum > (diff_sum_tm + 4*np.sqrt(var)))
    neg_tail = np.where( diff_sum < (diff_sum_tm - 4*np.sqrt(var)))

    pdb.set_trace()

def testing_riemann_sum(data, pres_var_name, pres_factor, lat_rad, test_type,
                        right_riemann_sum, left_riemann_sum, 
                        int_factor):

    ''' The cumtrapz function does not manage masked data. But can be used
        as a sanity check to make sure integration looks reasonable.'''

    print('Testing riemann sums for ', test_type)

    name_flag = '_test'

    if test_type == 'lat':
        var_flag = 'omega'
        data_factor = 10
        plot_scale_factor = 1e2
    else:
        var_flag = 'vcomp'
        data_factor = 1
        plot_scale_factor = 1e-4

    plot_single(data[var_flag]*data_factor, data[pres_var_name]*pres_factor, data['lat'], np.arange(-1,1.1,0.25), var_flag, '%.1f', name_flag=name_flag)

    plot_single(right_riemann_sum*plot_scale_factor, data[pres_var_name]*pres_factor, data['lat'],
                np.arange(-1,1.1,0.25), 'right_'+ test_type, '%.1f', name_flag=name_flag)
    plot_single(left_riemann_sum*plot_scale_factor, data[pres_var_name]*pres_factor, data['lat'], 
                np.arange(-1,1.1,0.25), 'left_'+ test_type, '%.1f', name_flag=name_flag)
        
    from scipy.integrate import cumtrapz
    if test_type == 'lat':
        int_val  = cumtrapz(int_factor*(np.cos(lat_rad) * data[var_flag]), x=lat_rad, axis=-1, initial=0)
    else:
        int_val  = cumtrapz(int_factor*(data[var_flag]), x=data[pres_var_name]*pres_factor, axis=1, initial=0)

    plot_single(int_val*plot_scale_factor, data[pres_var_name]*pres_factor, data['lat'], 
                np.arange(-1,1.1,0.25), 'cumtrapz_'+ test_type, '%.1f', name_flag=name_flag)

def testing_psi(data, pres_var_name, pres_factor, test_type,
                psi, riemann_avg):

    print('Testing psi output for ', test_type)

    name_flag = '_test'

    if test_type == 'lat':
        plot_scale_factor = 1
    else:
        plot_scale_factor = 1e-4
 
    plot_single(psi, data[pres_var_name]*pres_factor, data['lat'], 
                np.arange(-20,20.2,2), 'psi', '%d', name_flag=name_flag)
    plot_single(riemann_avg*plot_scale_factor, data[pres_var_name]*pres_factor, data['lat'], 
                np.arange(-1,1.1,0.1), 'psi_'+test_type, '%.1f', name_flag=name_flag)
    pdb.set_trace()

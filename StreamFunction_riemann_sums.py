import numpy as np
import pdb

from scipy.integrate import cumtrapz

import matplotlib.pyplot as plt

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



def mask_array(data_shape, fill_value):
    #create a new array without missing values but is masked
    data = np.zeros(data_shape)
    mask = np.zeros(data_shape) #all false equivalent           
    data = np.ma.masked_array(data, mask,  fill_value=fill_value)
    return data

def set_fill_value(data, fill_value):
    #take a masked array, update the fill value and return a masked array
   
    mask  = data.mask.copy()

    #replace the data.masked value with the specificied fill_value
    data = data.filled(fill_value=fill_value) #returns an unmasked arrary
    data_out = np.ma.masked_array(data, mask, fill_value=fill_value)
 
    return data_out


class MeridionalStreamFunction():

    '''Calculates the mass stream function. 
           Data is a dictionary with masked zonal-mean arrays of omega and vcomp.
           Pressure can be in hPa or Pa.

           Return: masked array with units x10^10 kg/s (i.e. it is scaled)

           Why compute the Riemann sums when you can use cumtrapz directly?
               Cumtrapz works great for unmasked arrays. I used it to validate
               the code. But for masked arrays, it ignores the mask which can 
               make for problematic stream functions.

           Note: in other peoples code (e.g. TrapD by Ori Admas and in Martin Juckers
               (https://github.com/mjucker/aostools/blob/master/climate.py), 
               psi is only computed using vcomp and not both the vcomp and omega
               components in this code.
    '''
    #for more information on riemann sums, see for example
    #https://www.math.ubc.ca/~pwalls/math-python/integration/riemann-sums/

    def __init__(self, pres_var_name, testing=False, name_flag=None):

        #constants
        self.grav=9.81 #m/s
        self.rad_earth=6.371e6 #m

        #scale factor for outputting psi
        self.psi_scale_factor = 1e10

        #the data's pressure name
        self.pres_var_name = pres_var_name

        #testing flag is for debugging
        self.testing = testing
        #syntax end of filename for plots during debugging
        self.name_flag = name_flag 

        #data to be assigned during the script
        self.pres     = None
        self.lat      = None
        self.lat_rad  = None
        self.omega_zm = None
        self.vcomp_zm = None

        #assigned during method
        self.input_dtype = None
        self.fill_value  = None

        #was the data reordered?
        self.flip_lat = None
        self.flip_pres = None

    def data_properties(self, data):

        #data properties
        self.input_dtype =  data['omega'].dtype
        self.fill_value = data['omega'].fill_value

        #check data
        assert isinstance(data['omega'],np.ma.core.MaskedArray)  == True , 'Data is not masked' 
        assert len(data['omega'].shape)  == 3, 'Data shape is not 3D, investigate' 

    def data_ordering(self, data):

        #if SH pole is not the first element, then reverse direction
        if np.where(data['lat'] == data['lat'].min())[0] !=0:
            self.lat      = data['lat'][::-1].copy()
            self.omega_zm = data['omega'][:,:,::-1].copy()
            self.vcomp_zm = data['vcomp'][:,:,::-1].copy()
            self.flip_lat = True
        else:
            self.lat = data['lat'].copy()
            self.omega_zm = data['omega'].copy()
            self.vcomp_zm = data['vcomp'].copy()
            self.flip_lat = False

        #test if data is in Pa or hPa - multiply data by 100 if in hPa
        if data[self.pres_var_name].max() > 1100:        
            pres_factor = 1.
        else:
            pres_factor = 100.

        # the pressure integral is from surface up.
        if np.where(data[self.pres_var_name] == data[self.pres_var_name].min())[0] ==0:
            self.pres = data[self.pres_var_name][::-1]*pres_factor
            self.omega_zm = self.omega_zm[:,::-1,:].copy()
            self.vcomp_zm = self.vcomp_zm[:,::-1,:].copy()
            self.flip_pres = True
        else:
            self.pres  = data[self.pres_var_name]*pres_factor 
            self.flip_pres = False

    def revert_data_ordering(self, data, psi):

        #if lat or pressure was reversed, then put back to original form
        if self.flip_lat:
            psi = psi[:,:,::-1].copy()

        if self.flip_pres:
            psi = psi[:,::-1,:].copy()

        return psi

    def set_fill_value_zero(self):
        # note this is only valid for surface missing values and not 
        # missing values in obs data for eg.

        #note that this method assigns a np array that is NOT masked
        self.omega_zm = set_fill_value(self.omega_zm, 0.0)
        self.vcomp_zm = set_fill_value(self.vcomp_zm, 0.0)

    def cumtrap_method_lat(self):

        #compute using scipy generated integration as a sanity check
        trap_lat  = cumtrapz( np.cos(self.lat_rad) * self.omega_zm.data, 
                              x=self.lat_rad, axis=-1, initial=0)

        return trap_lat

    def cumtrap_method_pres(self):

        #compute using scipy generated integration
        trap_pres  = cumtrapz(self.vcomp_zm.data, x=self.pres, axis=1, initial=0)

        return trap_pres

    def riemann_sum_lat(self):
        #compute the left and right riemann sums of psi_lat

        #new masked data arrays
        left_riemann_sum_lat   = mask_array(self.omega_zm.shape, self.fill_value)
        right_riemann_sum_lat  = mask_array(self.omega_zm.shape, self.fill_value)

        #start the loop at 1 (not 0) and integrate cos(lat) x omega from 90S to 90N
        for count in range(1, len(self.lat_rad)):
            #when integrating, use omega as unmasked data, else strips will be removed
            #from the resulting integral. Instead the missing values have been set to
            #zero and when it is integrated it will not contriute to the integral.

            #get the integrtaions step - same value used for left and right sums
            dlat = self.lat_rad[count] - self.lat_rad[count-1]  #dlat > 0 

            #compute the left riemann sum
            left_riemann_sum_lat[:,:,count]  = (left_riemann_sum_lat[:,:,count-1] + 
                np.cos(self.lat_rad[count]) * self.omega_zm[:,:,count-1].data * dlat)                                          

            #keep track of the data mask
            left_riemann_sum_lat[:,:,count].mask = self.omega_zm[:,:,count-1].mask

            #compute the right riemann sum
            right_riemann_sum_lat[:,:,count] = (right_riemann_sum_lat[:,:,count-1] +
                np.cos(self.lat_rad[count]) * self.omega_zm[:,:,count].data * dlat)

            #keep track of the data mask
            right_riemann_sum_lat[:,:,count].mask = self.omega_zm[:,:,count].mask

        return left_riemann_sum_lat, right_riemann_sum_lat

    def riemann_sum_pres(self):
        #compute the left and right riemann sums of psi_pres

        left_riemann_sum_pres  = mask_array(self.vcomp_zm.shape, self.fill_value)
        right_riemann_sum_pres = mask_array(self.vcomp_zm.shape, self.fill_value)
           
        for count in range(1,len(self.pres)):

            dp = (self.pres[count] - self.pres[count-1])  #dp<0
            
            #compute the left riemann sum
            left_riemann_sum_pres[:,count,:] = (left_riemann_sum_pres[:,count-1,:]+
                                               self.vcomp_zm[:,count-1,:].data * dp )

            #keep track of the data mask
            left_riemann_sum_pres[:,:,count].mask = self.vcomp_zm[:,:,count-1].mask


            #compute the left riemann sum
            right_riemann_sum_pres[:,count,:] = (right_riemann_sum_pres[:,count-1,:]+
                                                self.vcomp_zm[:,count,:].data * dp )

            #keep track of the data mask
            right_riemann_sum_pres[:,:,count].mask = self.vcomp_zm[:,:,count].mask

        return left_riemann_sum_pres, right_riemann_sum_pres

    def avg_masked_array(self, array1, array2):
        #inputs are two masked arrays of the same shape

        assert array1.shape == array2.shape, 'Input data is different shapes.' 

        #create a new masked array with dim [2, time, pre, lat]
        new_array = np.ma.array((array1, array2))
        #take the mean - if one array is masked, the value is the other array
        array_avg  = np.ma.mean(new_array, axis=0)

        return array_avg

    def apply_psi_scaling(self, riemann_lat_avg, riemann_pres_avg):

        psi_lat  = (-2.*np.pi*self.rad_earth*self.rad_earth)/self.grav * riemann_lat_avg 
        
        pres_const = ((2.*np.pi*self.rad_earth*np.cos(self.lat_rad))/self.grav)
        psi_pres = pres_const[np.newaxis,np.newaxis,:] * riemann_pres_avg

        return psi_lat/self.psi_scale_factor, psi_pres/self.psi_scale_factor

    def cal_stream_fn_riemann(self, data, plot_result=True):

        '''This part of the code does all the heavy lifting'''      

        #check data and get dtype and fill value
        self.data_properties(data)

        #ensure data is from SH to NH, surface to TOA and pres in Pa 
        self.data_ordering(data)

        #degree to radians
        self.lat_rad=np.radians(self.lat, dtype=self.input_dtype)

        #Set the fill value to zero, else the surface masking will block out
        #strips of the integration
        self.set_fill_value_zero()

        #Do the integration using the inbuild function cumtrap
        # this is for a visual check for debugging the riemann sums
        # the cumtrap function is a black box that does not handle masked data
        # where the left and right sums I know exactly how it is computed
        trap_lat = self.cumtrap_method_lat()
        trap_pres = self.cumtrap_method_pres()

        #Do the integration with the left and right riemann sums
        left_riemann_sum_lat, right_riemann_sum_lat = self.riemann_sum_lat()
        left_riemann_sum_pres, right_riemann_sum_pres = self.riemann_sum_pres()

        #get the average of the left and right sums
        riemann_lat_avg = self.avg_masked_array(left_riemann_sum_lat, right_riemann_sum_lat)
        riemann_pres_avg = self.avg_masked_array(left_riemann_sum_pres, right_riemann_sum_pres)

        #Apply the scaling for the psi intergrals
        psi_lat, psi_pres = self.apply_psi_scaling(riemann_lat_avg, riemann_pres_avg)

        #take the avergae of the two psi functions
        psi = self.avg_masked_array(psi_lat, psi_pres)     

        TS = TestingStreamfunction(self.omega_zm, self.vcomp_zm, 
                                   self.pres, self.lat, self.lat_rad, self.name_flag)

        if self.testing:
            #make extra plots if testing code
            TS.testing_riemann_sum('lat', right_riemann_sum_lat, left_riemann_sum_lat, 
                                   trap_lat, riemann_lat_avg)

            TS.testing_riemann_sum('pres', right_riemann_sum_pres, left_riemann_sum_pres, 
                                   trap_pres, riemann_pres_avg)

        if plot_result:
            #always plot the psi values as a visual check the code is working
            TS.testing_psi(psi_lat, psi_pres, psi)

        #reverse the pressure back ot original coordinates if needed
        psi = self.revert_data_ordering(data, psi)

        return psi


class TestingStreamfunction():

    def __init__(self, omega_zm, vcomp_zm, pres, lat, lat_rad, name_flag):

        self.omega_zm = omega_zm
        self.vcomp_zm = vcomp_zm
        self.pres     = pres
        self.lat      = lat
        self.lat_rad  = lat_rad

        self.name_flag = name_flag
 
    def testing_riemann_sum(self, test_type, right_riemann_sum, left_riemann_sum, 
                            trap_val, riemann_avg, time_mean=True):

        print('Testing riemann sums for ', test_type)

        data_range = np.arange(-1,1.1,0.25)

        if test_type == 'lat':
            var_flag = 'omega'
            data_factor = 10
            plot_scale_factor = 1e2
            data = self.omega_zm
        else:
            var_flag = 'vcomp'
            data_factor = 1
            plot_scale_factor = 2e-5
            data = self.vcomp_zm

        self.contour_plot(data*data_factor, data_range, var_flag, time_mean)

        self.contour_plot(right_riemann_sum*plot_scale_factor, data_range, 'right_'+ test_type, time_mean)

        self.contour_plot(left_riemann_sum*plot_scale_factor, data_range, 'left_'+ test_type, time_mean)        

        self.contour_plot(riemann_avg*plot_scale_factor, data_range, 'avg_'+ test_type, time_mean)        

        self.contour_plot(trap_val*plot_scale_factor, data_range, 'cumtrapz_'+ test_type, time_mean)

    def testing_psi(self, psi_lat, psi_pres, psi, time_mean=True):
     
        scale_factor = 1
        bounds = np.arange(-20.,20.1,5)
        self.contour_plot(psi_lat*scale_factor,  bounds, 'psi_lat',time_mean)
        self.contour_plot(psi_pres*scale_factor, bounds, 'psi_pres',time_mean)
        self.contour_plot(psi*scale_factor, bounds, 'psi',time_mean)

    def contour_plot(self, data, bounds, name, time_mean):

        #incoming pressure is in Pa

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        cmap = plt.get_cmap('RdBu_r')

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        if time_mean:
            start_loop = 1
            time_loop_len = 1
        else:
            #sorry this is hard coded - I will change it in the future.
            start_loop = 120
            time_loop_len = 120+12#len(data.shape[0])

        for count in range(start_loop, time_loop_len+1):
            print('Time is: ', count)
            if time_mean:
                data_plot = np.ma.mean(data, axis=0)
                filename='testing_psi_{0}.eps'.format(name, self.name_flag)
            else:
                data_plot = data[count,...].copy()

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
            
            filled_map = ax.pcolormesh(self.lat, self.pres/100., data_plot,
                                       cmap=cmap,norm=norm, shading='nearest')
            ax.set_ylim(1000,100)
            ax_cb=fig.add_axes([0.15, 0.05, 0.8, 0.03]) #[xloc, yloc, width, height]
            cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,norm=norm, ticks=bounds,
                                             orientation='horizontal',format='%.1f',
                                             extend='both')    
            if time_mean:
                plt.savefig(filename)
            plt.show()

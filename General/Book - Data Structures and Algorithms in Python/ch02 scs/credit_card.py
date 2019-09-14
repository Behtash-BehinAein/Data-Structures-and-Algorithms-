class MRAM:
    '''
    This MRAM software simulates MRAM circuit based on MTJ electrical characteristics, MRAM array statistics and sense-amp statistics and specs
   
    It includes Si process, voltage and temperature effects.
    It also provides predictions for:
        - READ-margin and read-error-rates (RER)
        - WRITE-margin and write_error-rates (WER)
        - Yield
    The module has various methods that are designed for convenient data extraction and illustration 
    '''
    
    # Python modules that maybe used  =====s
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.utils import resample
    import random 
    import matplotlib.pyplot as plt
    from IPython.display import display
    plt.rcParams.update({'font.size': 20})
    # =====================================
    
    
    # Class variables ==================================================
    _n_bs              = 500     # Number of bootstrap samples 
    _tmr_t_cfc         = 0.0028  # TMR temp coefficient 
    _rho_ramin_ramax   = 0.55    # Correlation of RAmin and RAmax
    _rref_sig_poly     = 43.64   # [Ohm] sigma of poly reference 
    _numdb_per_mtjrb   = 64      # DB: data bits  | RB: reference bits. Number of data bits hadled for each reference bit.
    _numdb_per_polyrb  = 256e3   # DB: data bits  | RB: reference bits . Number of data bits hadled for each reference bit.
    _sa_trim_step      = 60      # [Ohm]
    # MTJ sense-amp referance             ---------------------------------
    '''
    Circuit calibration
        - sigma: actual(from ckt simulation) i.e. 137.5 / rmS(rmin_sig, rmax_sig) i.e. 240.0  
        - loc:   actual(from ckt simulation) i.e. 3701 /  rmin || rmax_sig i.e. 1701  
    '''
    _rref_sig_calib = 0.573 
    _rref_loc_calib = 2.176
    # ==================================================================
    
    
    # ==================================================================
    
    # Define the main attributes
    def __init__(self, tmr, temp_c , vmtj = [], imtj = [] , cd=80e-9, cd_sigma=1.75e-9, e=3e-9 , e_sigma=0.3e-9, ra=10):
        #
        self._vmtj = vmtj
        self._imtj = imtj
        # tmr T-dependence at 0 Bias --------------------------------
        self._temp_c = temp_c
        self._tmr    = tmr
        # Resistance-area product
        self.ra      = ra       
        # Size of the statistical distributions
        self._n_smp  = int(1e5)    
        # -----------------------------------------------------------
        # 
        # Fundamental process parameters        -------------------
        self._cd       = cd
        self._cd_sigma = cd_sigma
        self._e        = e
        self._e_sigma  = e_sigma
        self._ecdnm    = (self._cd - 2*self._e)*1e9
        # ---------------------------------------------------------
        #
        # tmr   -----------------------------------------------------------------------------------------
        if not isinstance(self._tmr, (int,float)):
            raise TypeError('tmr must be an integer or a float.')
        if self._tmr < 0 or self._tmr>2:
            raise ValueError('tmr must be a positive ratio between 0 and 2.')
        # -----------------------------------------------------------------------------------------------   
        #
        # MTJ IV: float, list or numpy array, can have 1 or more elements but not zero    =========================================
        if not (bool(imtj) != bool(vmtj)):  # One and only one input excitation must be provided.
            raise ValueError('EITHER MTJ-voltage (vmtj) OR MTJ-current(imtj) must be given as input.') 
        #----------------------------------------
        # Input Voltage   [V]
        if self._vmtj:
            if isinstance(vmtj, self.np.ndarray):
                self._vmtj =  vmtj
            elif isinstance(vmtj, list):
                self._vmtj = self.np.array(vmtj)
            elif isinstance(vmtj,(int, float)):
                self._vmtj = self.np.array([vmtj])                
            if max(self._vmtj)>1 or min(self._vmtj)<-1:
                raise ValueError(' The acceptable range for VMTJ is (-1, 1)V')
        # ----------------------------------------
        # Input Current  in [uA]
        if self._imtj:
            if isinstance(imtj, self.np.ndarray):
                self._imtj =  imtj *1e-6
            elif isinstance(imtj, list):
                self._imtj = self.np.array(imtj) *1e-6
            elif isinstance(imtj,(int, float)):
                self._imtj = self.np.array([imtj]) *1e-6
        # =========================================================================================================================   

        
    # Base Statistics      ======================
    def _set_rnd_variables(self):
        
        self.np.random.seed(1)
        rnd1 = self.np.random.normal(0, 1, self._n_smp) 
       
        self.np.random.seed(2)
        rnd2 = self.np.random.normal(0, 1, self._n_smp)
        
        self.np.random.seed(3)
        rnd3 = self.np.random.normal(0, 1, self._n_smp) 
        
        self.np.random.seed(4)
        rnd4 = self.np.random.normal(0, 1, self._n_smp)
        
        
        return rnd1, rnd2, rnd3, rnd4
    # ============================================
    

    # Process               =======================
    def _set_prc_dist(self):
        cd_smp   = self._cd_sigma * self._set_rnd_variables()[0] + self._cd   
        e_smp    = self._e_sigma  * self._set_rnd_variables()[1] + self._e    
        return cd_smp, e_smp 
    # =============================================

    
    # Geometry               ======================
    def _calc_geom(self):
        a         = self.np.pi*((self._cd - 2*self._e) / 2)**2                                  # Electrical area
        a_smp     = self.np.pi*((self._set_prc_dist()[0] - 2*self._set_prc_dist()[1]) / 2)**2   # Distribution of electrical area
        a_mag     = self.np.pi*(self._cd/2)**2                                                  # Magnetic area
        a_mag_smp = self.np.pi*(self._set_prc_dist()[0]/2)**2                                   # Distribution of magnetic area       
        return a, a_smp , a_mag, a_mag_smp  
    # ============================================


    # Rmin     ====================================================
    def _calc_rmin(self):
        
        ra_min           = self.ra*1e-12 # Ohm.m**2
        rmin             = self.np.round(ra_min / self._calc_geom()[0],1)
        ra_min_sig_ratio = 4.56797808e-6 * self._ecdnm**2 - 8.99833497e-4 * self._ecdnm +  (0.94) * 6.57200054e-2 # 0.94 is to hit target         
        ra_min_sigma     = ra_min_sig_ratio * ra_min
        ra_min_smp       = ra_min_sigma * self._set_rnd_variables()[2] + ra_min
        rmin_smp         = ra_min_smp/self._calc_geom()[1]
        
        return ra_min , rmin , ra_min_sig_ratio, rmin_smp
    
    #==============================================================    

    # Rmax at room temperature and 0 bias ============================
    def _rmax_atrt_0bias(self):   
        '''
        Method provides sample rmax distribution @ rt and @ zeros bias.
        rmax_smp is generated in correlation with rmin_smp using their
        respective ra_smp that are carefullly desiged to be correlated
        
        Returns:
            - Sample rmax distribution
        '''
        ra_max_sig_ratio  = self._calc_rmin()[2]
        ra_max           = ( 1 + self._tmr)*self._calc_rmin()[0] 
        ra_max_sigma     = ra_max_sig_ratio*ra_max
        rnd_crlted       = MRAM._rho_ramin_ramax * self._set_rnd_variables()[2] + self.np.sqrt(1 - MRAM._rho_ramin_ramax**2)*self._set_rnd_variables()[3]
        ra_max_smp       = ra_max_sigma * rnd_crlted + ra_max       
        rmax_smp_rt_v0   = ra_max_smp/self._calc_geom()[1]

        return rmax_smp_rt_v0
    #   ===============================================================
    
    # =====
    '''
    Method calculates vmtj if imtj is the input excitation.  
    
    '''
    
    def _get_v_if_i(self): 
        if any(self._vmtj):
            vmtj = self._vmtj
        else:
            c3p , c2p , c1p = - 5e-6*self._temp_c + 0.0638 , 2e-5*self._temp_c - 0.3612 , 6e-5*self._temp_c + 0.956
            c3n , c2n , c1n =   4e-5*self._temp_c + 0.059 ,  5e-5*self._temp_c + 0.3557 , 2e-4*self._temp_c + 0.9303
            
            tmr_attemp_0bias  = self._tmr*(1 - MRAM._tmr_t_cfc*(self._temp_c - 25))
            rmax_attemp_0bias = (1 + tmr_attemp_0bias) *self._calc_rmin()[1]
            
            # vmtj as a function of imtj
            # imtj is multiplied by rmax because the v-dep of rmax is based on normalized rmax
            vmtj = (c3p * (self._imtj*rmax_attemp_0bias) **3 + c2p * (self._imtj*rmax_attemp_0bias) **2 + c1p*(self._imtj*rmax_attemp_0bias) ) * (self.np.sum(self._imtj) >= 0) + \
                   (c3n * (self._imtj*rmax_attemp_0bias) **3 + c2n * (self._imtj*rmax_attemp_0bias) **2 + c1n*(self._imtj*rmax_attemp_0bias) ) * (self.np.sum(self._imtj) <  0)
        return vmtj
    # =====
    
    # =====
    '''
    Method calculates imtj if vmtj is the input excitation.  
    
    '''
    
    def _get_i_if_v(self): 
        if any(self._imtj):
            imtj = self._imtj
        else:
            tmr_attemp_0bias  = self._tmr*(1 - MRAM._tmr_t_cfc*(self._temp_c - 25))
            rmax_attemp_0bias = (1 + tmr_attemp_0bias) *self._calc_rmin()[1]

            imtj = self._vmtj / rmax_attemp_0bias     
        return imtj
    # =====
    
    # Normalized voltage dependence of Rmax at operating temperature  ===========
    def _rmaxnorm_vdep_attemp(self, temp_c):
        """
        Method provides normalized voltage dependence of rmax at T. 
        at this time, only rmax has voltage and temperature dependence. 
        Ouput: an n by 1 numpy array  
        
        Returns:
            - Normalized V dependence of rmax @ T
        """
        
        
        
        # temperature dependent rmax bias dependence ----------
        rmax_slope_tdep_pve = - 0.00021 *self._temp_c + 0.52525
        rmax_slope_tdep_nve = - 0.000525*self._temp_c + 0.592625
        # -----------------------------------------------------
        #
        # ----------------------------------------------------------------------------------------------------------------
        FIT = (rmax_slope_tdep_nve*self._get_v_if_i() + 1) * (self._get_v_if_i()<0) + (- rmax_slope_tdep_pve*self._get_v_if_i() + 1) * (self._get_v_if_i()>=0)
        # Normalized rmax(vmtj,T). This function is later multiplied by the rmax value @ (vmtj=0 ,T=25) to 
        # ouput rmax.  Note that rmax value @ (V=0 ,T=25) can be an instance value in the sample rmax distribution
        # or the entire rmax distribution : see the method rmax_at_atV
        # ----------------------------------------------------------------------------------------------------------------
        # 
        return FIT.reshape(-1,1)
    #   ========================================================================
    
    
    def _rmax_attemp_at_bias(self):
        """
        Method provides rmax distribution by taking rmax's normalized voltage dependence and 
        rescaling it by its distribution @ (rt , 0Bias), then rescaling it again to obtain
        its value at operating T
            
        -   For reference purposes, nominal rmax value as a function of bias and 
            distributed rmax value at 0Bias has also been provided.
            
        Returns:  
            - Nominal rmax               : rmax_nom_atv_attemp            
            - Distributed rmax @ T,0Bias : rmax_smp_0Bias
            - Distributed rmax @ V,T     : rmax_smp
        """
        tmr_attemp_0bias = self._tmr*(1 - MRAM._tmr_t_cfc*(self._temp_c - 25))

        # Nominal rmax @ vmtj
        self.rmax_nom_atv_attemp = self._rmaxnorm_vdep_attemp(self._temp_c) *\
        ( self.np.array((1 + tmr_attemp_0bias) * (self._calc_rmin()[1]*(1+self._tmr) / (1 + self._tmr)))  ).reshape(1,-1)

        # Distributed rmax @T and @ V/I
        self.rmax_smp = self._rmaxnorm_vdep_attemp(self._temp_c) * ( (1 + tmr_attemp_0bias) * (self._rmax_atrt_0bias() \
                                                / (1 +self._tmr))          ).reshape(1,-1)

        return self.rmax_nom_atv_attemp , self.rmax_smp 

    #===============
    
    def rmax(self):
        '''
        Creates a dataframe from rmax data. This is for ease of use/access for diagnostics, plotting and illustration.
        
        Returns:
            - A dataframe of various Rmax parameters 
        '''
        # rmax distribution at user defined bias(es)
        rmax_nom_atv_attemp, rmax_smp  = self._rmax_attemp_at_bias()
        
        df            = self.pd.DataFrame()                 
        df['vmtj[V]'] = self._get_v_if_i()                                            # Store MTJ Voltage
        df['imtj[uA]'] = self.np.round(self._get_i_if_v()*1e6 , 1)                                         # Store MTJ current
        df['temp[C]'] = self._temp_c                                          # Store MTJ Voltage
        df['rmax_nom[$\Omega$]'] = self.np.round(rmax_nom_atv_attemp,1)                     # Store rmax nominal value @ (vmtj, T)
        df['rmax_smp_avg[$\Omega$]']  = self.np.round(np.mean  (rmax_smp, axis=1),1)     # Store rmax smp mean @ (vmtj, T)
        df['rmax_smp_med[$\Omega$]']  = self.np.round(np.median(rmax_smp, axis=1),1)     # Store rmax smp med  @ (vmtj, T)
        df['rmax_sig[$\Omega$]'] = self.np.round(np.std   (rmax_smp, axis=1),1)     # Store rmax smp std in Ohms  @ (vmtj, T)
        df['rmax_sig[%]']   = self.np.round(df['rmax_sig[$\Omega$]']/df['rmax_smp_avg[$\Omega$]']*100 ,1)    # Store rmax smp std in [%]  @ (vmtj, T)

        return df

    
    #===============

    
    def tmr_attemp_atbias(self):
        """
        Method provides tmr distribution by taking (rmin,rmax)'s distributions as inputs.  

        -   For reference purposes, nominal tmr value as a function of bias and 
            distributed tmr value at 0Bias has also been provided.
            
        Returns: 
            - Nominal tmr                : tmr_nom_atv_attemp            
            - Distributed tmr @ T,0Bias  : tmr_smp_0Bias
            - Distributed tmr @ T,V      : tmr_smp
        """
        rmax_nom_atv_attemp , rmax_smp  = self._rmax_attemp_at_bias() 
        
        # Nominal tmr @T and @bias
        tmr_nom_atv_attemp= (rmax_nom_atv_attemp - self._calc_rmin()[1]) / self._calc_rmin()[1] *100

        # Distributed tmr 
        tmr_smp =  (rmax_smp - self._calc_rmin()[3]) / self._calc_rmin()[3] *100
        
        # Report results 
        df         = self.pd.DataFrame()                 
        df['vmtj[V]']  = self._get_v_if_i()                                            # Store MTJ Voltage
        df['imtj[uA]'] = self.np.round(self._get_i_if_v()*1e6 , 1)                          # Store MTJ current
        df['temp[C]']  = self._temp_c                                                  # Store MTJ temperature 
        df['tmr_nom[%]']      = self.np.round(tmr_nom_atv_attemp, 1)                         # Store rmax nominal value @ (bias, T)
        df['tmr_smp_avg[%]']  = self.np.round(self.np.mean  (tmr_smp, axis=1), 1)                # Store rmax smp mean @ (bias, T)
        df['tmr_smp_med[%]']  = self.np.round(self.np.median(tmr_smp, axis=1), 1)                # Store rmax smp med  @ (bias, T)
        df['tmr_sig[$\Omega$]'] = self.np.round(self.np.std (tmr_smp, axis=1), 1)                   # Store rmax smp std in Ohms  @ (bias, T)
        df['tmr_sig[%]']   = self.np.round(df['tmr_sig[$\Omega$]']/df['tmr_smp_avg[%]']*100 ,1)  # Store rmax smp std in [%]  @ (bias, T)       

        return df
    
    #===============
 
    
    #===============
    
    def rminrmax_win(self, tail_prob, empirical=False):
        '''
        - Methed provides the difference between high rmin value (defined by tail_prob as the upper tail)
          and low rmax value (defined by tail_prob as the lower tail). 
      
        - This is @ user-defined tail_prob , @ user-defined voltages and @ user-defined temperature
        - The method calculates the window by assuming a lognormal distribution for rmin and rmax. 
        - If empirical is set to True, then the window is calculated using data 
          without assuming any particular distribution. 
        '''
   
        # Define a data frame to hold theoretical and empirical rminrmax window 
        df                 = self.pd.DataFrame()
        df['vmtj[V]']      = self._get_v_if_i()
        df['imtj[uA]']     = self.np.round(self._get_i_if_v()*1e6 , 1)    
        df['temp[C]']      = self._temp_c  
        df['ecd[nm]']      = self._ecdnm 
        df['rmin[$\Omega$]']   = self._calc_rmin()[1]    
        df['rmin_sig [%]'] = self.specs()['rmin_sig_smp [%]'][0] 
        df['rmax[$\Omega$]']   = self.rmax()['rmax_nom[$\Omega$]']    
        df['rmax_sig [%]']     = self.rmax()['rmax_sig[%]']
        df['tmr[%]']        = self.tmr_attemp_atbias()['tmr_smp_avg[%]']   


        # Theoretical approach (assumes lognormal distribution)    -----------
        # rmin...
        loc   = self.np.log(self._calc_rmin()[3]).mean()
        scale = self.np.log(self._calc_rmin()[3]).std()
        rmin_high = self.np.exp(self.stats.norm.ppf(1-tail_prob, loc , scale))
        # rmax...
        _ , rmax_smp = self._rmax_attemp_at_bias()
        
        loc   = self.np.mean(np.log(rmax_smp), axis = 1)
        scale = self.np.std(np.log(rmax_smp), axis = 1)
        rmax_low = self.np.exp(self.stats.norm.ppf(tail_prob, loc , scale))
        #
        '''
        # Test the equivalance of Sort-and-% method vs ppf method for finding the tail values
        if len(self._get_v_if_i()>1):
            rmaxSorted = np.sort(np.exp(np.random.randn(self._n_smp).reshape(1,-1) * scale.reshape(-1,1) + loc.reshape(-1,1)))
            print(rmaxSorted[0][int(tail_prob*self._n_smp)] , rmaxSorted[1][int(tail_prob*self._n_smp)])
            print(rmax_low)
            print(10*'==')
        else: 
            rmaxSorted = np.sort(np.exp(np.random.randn(self._n_smp).reshape(1,-1) * scale.reshape(-1,1) + loc.reshape(-1,1)))
            print(rmaxSorted[0][int(tail_prob*self._n_smp)] , rmaxSorted[1][int(tail_prob*self._n_smp)])
            print(rmax_low)
            print(10*'==')
        '''
        
        df['rmin_rmax_win[Ohm] : theoretical']  = np.round((rmax_low - rmin_high), 1)
        
        # --------------------------------------------------------------------
        
        if empirical:
            # empirical approach (provides confidence intervals using bootstrap)
            win = self.np.zeros((self._get_v_if_i().shape[0], MRAM._n_bs )) 
            for ii in range(MRAM._n_bs):

                # rmin tail
                rmin_high = self.np.quantile(self.resample(self._calc_rmin()[3]), 1-tail_prob)
                # rmax tail
                rmax_low  = self.np.quantile(self.np.array( [self.resample(vec) for vec in self._rmax_attemp_at_bias()[1]] ), tail_prob, axis=1)
                # rmin-rmax win
                win[:, ii] = (rmax_low - rmin_high)


            win_avg = self.np.mean(win, axis=1).reshape(-1,1)
            win_std = self.np.std(win, axis=1).reshape(-1,1)

            ci_low  = win_avg - 1.96*win_std
            ci_high = win_avg + 1.96*win_std

            df['rmin_rmax_win[Ohm]: empirical'] = self.np.round(win_avg, 1)
            df['empirical 95% CI Low']  = self.np.round(ci_low,1)
            df['empirical 95% CI High'] = self.np.round(ci_high,1)

        return df
    
    
    #===============
    
    def rref(self, poly=False):  
        '''
        This method defines the reference distribution and returns the distribution and the reference band. 
        
        - Reference band depends on both the standard deviation of the devices used as refrence as well as 
          the number of refrence bits per 1e6 databits (error is reported in ppm) 
        - Both MTJ and Poly reference are covered here with the defualt being MTJ reference for which keyword poly=False
        
        Returns:
            - Reference band at values of vmtj array
            - The reference sample distribution
            - The center of the reference distribution : can be tuned via Sa trim
        '''

        
        rmax_nom_atv_attemp , rmax_smp = self._rmax_attemp_at_bias()
        rref_loc   = MRAM._rref_loc_calib * (self._calc_rmin()[1] * rmax_nom_atv_attemp) / (self._calc_rmin()[1] + rmax_nom_atv_attemp) 
        
        if poly == False:     
            rref_sig   = MRAM._rref_sig_calib * self.np.sqrt( (self.np.std(self._calc_rmin()[3])**2 + self.np.std(rmax_smp,axis=1)**2) /2 )

            ref_bits_per_mil      = int(1e6/MRAM._numdb_per_mtjrb)           # The number 64 quantifies the number of reference bits per 1e6 data bits. It is very much design dependent 
            ref_error_rate_1ppm   = 1 / ref_bits_per_mil  # On average, this is the most extreme probability we can have in the Pdf of ref bits
            ref_error_1ppm_zscore = abs(self.stats.norm.ppf(ref_error_rate_1ppm))
            
            rref_band  = 2 * ref_error_1ppm_zscore * rref_sig  
            rref_smp   = rref_sig.reshape(-1,1) * np.random.randn(self._n_smp).reshape(1,-1) +  rref_loc
        else: 
            ref_bits_per_mil      = int(1e6/MRAM._numdb_per_polyrb)    # The number 160 quantifies the number of reference bits per 1e6 data bits. It is very much design dependent 
            ref_error_rate_1ppm   = 1 / ref_bits_per_mil  # On average, this is the most extreme probability we can have in the Pdf of ref bits
            ref_error_1ppm_zscore = abs(self.stats.norm.ppf(ref_error_rate_1ppm))
            
            rref_band  = 2 * ref_error_1ppm_zscore * MRAM._rref_sig_poly  
            rref_smp   = MRAM._rref_sig_poly * np.random.randn(self._n_smp) +  rref_loc

        return self.np.round(rref_band,1) , rref_smp ,  rref_loc
        
    #===============
    
    def read_margin(self, target_error_rate=1e-6, ac_gb = 50 , empirical = False , poly = False):  
        '''
        This method calls the following methods: rminrmax_win & rref.
        
        It then uses the outputs from the previous methods along with the aC gaurd band to calculate the read margin
        @ user-specified error rate
        
        Returns: 
            - a dataframe containing relevant MraM specs and Read Margin for those specs 
        '''
        
        win              = self.rminrmax_win(tail_prob=target_error_rate, empirical=empirical)['rmin_rmax_win[Ohm] : theoretical'].values
        rref_band, _ , _ = self.rref(poly=poly)
        rm               = win - rref_band - 2*ac_gb
        
        rm_df = self.pd.DataFrame()
        rm_df['vmtj[V]']      = self._get_v_if_i()
        rm_df['imtj[uA]']     = self.np.round(self._get_i_if_v()*1e6 , 1)    
        rm_df['temp[C]']      = self._temp_c  
        rm_df['ecd [nm]']     = self._ecdnm 
        #rm_df['rmin [Ohm]']   = self._calc_rmin()[1]    
        rm_df['rmin_sig [%]'] = self.specs()['rmin_sig_smp [%]'][0] 
        rm_df['tmr[%]']        = self.tmr_attemp_atbias()['tmr_smp_avg[%]']  
        rm_df['ac g.b.[$\Omega$]']      = ac_gb 
        rm_df[f'rm@{target_error_rate}']  = rm
        
        return rm_df
    
    #===============
    
    
    def rer(self, ac_gb = 50 , empirical = False , poly = False):  
        '''
        Method calculates Read Margin for given AC gaurd-band for the technology under consideration.
        
        It supports poly-reference as well as MTJ reference 
        Methods depends on rmin,rmax distribution as well as reference distribution
        
        Returns: 
            - Dataframe of R0 failures
            - Dataframe of R1 failures
            - Dataframe of read error rate
        '''
        _ , rmax_smp = self._rmax_attemp_at_bias()
        rref_band , _ , ref_loc_default = self.rref(poly=poly)
        
        # Theoretical approach (assumes lognormal distribution)    ----------------
        # rmin...
        loc_min   = np.mean(np.log(self._calc_rmin()[3]))
        scale_min = np.std(np.log(self._calc_rmin()[3]))
        # rmax...
        loc_max   = np.mean(np.log(rmax_smp), axis =1).reshape(-1,1)
        scale_max = np.std(np.log(rmax_smp), axis = 1).reshape(-1,1)
        # -------------------------------------------------------------------------
        
        grid_res = 15  # grid resolution
        Grid_Ref_loc = np.linspace(-5 , 9 , grid_res)
        sa_trim  = (Grid_Ref_loc*MRAM._sa_trim_step).reshape(1,-1)  +   ref_loc_default.reshape(-1,1) 
        sa_trim = sa_trim.reshape(len(self._get_v_if_i()), -1)       

        # -------------------------------------------------------------------------
        r0_er = np.zeros((len(self._get_v_if_i()) , grid_res ))
        r1_er = np.zeros((len(self._get_v_if_i()) , grid_res ))
        rer   = np.zeros((len(self._get_v_if_i()) , grid_res ))
        for i in range(sa_trim.shape[0]):      # Over Voltage
            print(i)
            for j in range(sa_trim.shape[1]):  # Over Sa trim grid 
                
                cliff_low  =  sa_trim[i,j] - rref_band[i]/2 - ac_gb 
                cliff_high =  sa_trim[i,j] + rref_band[i]/2 + ac_gb
                
                r0_er[i,j] = 1 - self.stats.norm.cdf(np.log(cliff_low), loc_min , scale_min)
                r1_er[i,j] =     self.stats.norm.cdf(np.log(cliff_high), loc_max[i] , scale_max[i])
                rer[i,j]   = r0_er[i,j] + r1_er[i,j]
  
                # Test  Read Margin vs RER  ------------------------------------
                '''
                rmin_high = np.exp(self.stats.norm.ppf(1-1e-6, loc_min , scale_min))
                rmax_low = np.exp(self.stats.norm.ppf(1e-6, loc_max , scale_max))
                if np.isclose(r0_er[i,j], 1e-6, rtol=0.3, atol=1e-8):
                    print("rmin H' : ", cliff_low)
                    print("rmin H  : ", rmin_high)
                if np.isclose(r1_er[i,j], 1e-6, rtol=0.3, atol=1e-8):
                    print("rmax H' : ", cliff_high)
                    print("rmax H  : ", rmax_low[i])
                '''    
                #---------------------------------------------------------------
                
        r0_er_df = r1_er_df = rer_df = self.pd.DataFrame()         
        r0_er_df['vmtj[V]'] = r1_er_df['vmtj[V]']  = rer_df['vmtj[V]']  = self._get_v_if_i()  
        r0_er_df['imtj[uA]'] = r1_er_df['imtj[uA]']  = rer_df['imtj[uA]']  = np.round(self._get_i_if_v()*1e6 , 1)  
        r0_er_df['temp[C]'] = r1_er_df['temp[C]']  = rer_df['temp[C]']  = self._temp_c
        r0_er_df['ecd[nm]'] = r1_er_df['ecd[nm]']  = rer_df['ecd[nm]']  = self._ecdnm 
        r0_er_df['rmin[$\Omega$]']   = r1_er_df['rmin[$\Omega$]']    = rer_df['rmin[$\Omega$]']   = self._calc_rmin()[1]
        r0_er_df['rmin_sig[%]'] = r1_er_df['rmin_sig[%]']  = rer_df['rmin_sig[%]'] = self.specs()['rmin_sig_smp [%]'][0]
        r0_er_df['tmr[%]'] = r1_er_df['tmr[%]']  = rer_df['tmr[%]'] = self.tmr_attemp_atbias()['tmr_smp_avg[%]']

        r0_er_data = self.pd.DataFrame(r0_er , columns=[str(i) for i in Grid_Ref_loc])
        r0_er_df   = self.pd.concat([r0_er_df, r0_er_data], axis=1)
        
        r1_er_data = self.pd.DataFrame(r1_er , columns=[str(i) for i in Grid_Ref_loc])
        r1_er_df   = self.pd.concat([r1_er_df,r1_er_data], axis=1)

        rer_data   = self.pd.DataFrame(rer   , columns=[str(i) for i in Grid_Ref_loc])
        rer_df     = self.pd.concat([rer_df , rer_data], axis=1)
        
        return r0_er_df, r1_er_df, rer_df
    
    #===============
             
    def specs(self):
        '''
        Method stores key MraM specs in a Pandas dataframe
        '''
        df = {}
        # Input parameters 
        #df['temp']        = [self._temp_c]
        df['tmr[%]']      = [self._tmr*100]
        df['ecd[nm]']     = [self._ecdnm]
        
        # rmin Section
        df['rmin_pop']    = [self._calc_rmin()[1]]
        df['rmin_smp']    = [self.np.round(np.mean(self._calc_rmin()[3]),1)]
        df['rmin_sig_smp [Ohm]']  = [self.np.round(self.np.std(self._calc_rmin()[3]),1)]
        df['rmin_sig_smp [%]']    = self.np.round(self.np.std(self._calc_rmin()[3])/ np.mean(self._calc_rmin()[3]) *100, 2)
        
        # rmax section 
        df['rmax_pop_0Bias25C']      = [self.np.round(self._calc_rmin()[1] *(1+self._tmr),1)]
        df['rmax_smp_0Bias25C']      = [self.np.round(self.np.mean(self._rmax_atrt_0bias()),1)]
        df['rmax_sig_smp_0bias25C [Ohm]']  = [self.np.round(self.np.std(self._rmax_atrt_0bias()),1)]
        df['rmax_sig_smp_0bias25C [%]']    = self.np.round(df['rmax_sig_smp_0bias25C [Ohm]'][0] / df['rmax_smp_0Bias25C']*100, 2)[0]
                                                
        return self.pd.DataFrame(df)
        
    #===============
        
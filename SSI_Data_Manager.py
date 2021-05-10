#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:51:40 2021

@author: jedmond
"""



"""
TO DO:
    1.) Fix to_csv such that it takes in units and converts approp when
        it gets read back in
    2.) Add set_start_date func
"""




import pandas as pd
import numpy as np
import datetime, time, os, warnings, math
import matplotlib.pyplot as plt





# ==================== SUPPORTING CLASS: CUSTOMDATE ====================
    
    
# Define date class for ease of moving between dates expressed as strings
# or series of numbers (e.g. "3 Aug 2012, 12:35:50" --> equivalent epoch time)

class CustomDate:
    
    default_date = [1,1,1,0,0,0]
    
    # do args and kwargs thing here so that they get passed to
    # datetime
    # engine can be: "pandas", "python_datetime"
    def __init__(self,date_list=None,epoch=None,from_str=None,engine=None):
        
        # Default arg handling
        if from_str is None: from_str = False
        self.using_pandas, self.using_python_datetime = False, False
        if engine is None: self.using_pandas = True
        else:
            if engine == "pandas": self.using_pandas = True
            if engine == "python_datetime": self.using_python_datetime = True
        
        # Can't provide both epoch and date_list; has to be one or the other
        both_args_set = date_list is None and epoch is None
        neither_arg_set = date_list is not None and epoch is not None
        if both_args_set or neither_arg_set:
            excep_str = "Error! When instantiating Date, must either...\n   " + \
                "(1) provide a list of strings or numbers as date info to " + \
                "date_list, or \n   " + \
                "(2) provide the epoch time (in seconds) to epoch"
            raise Exception(excep_str)
        
        # Set class variables to None val for intialization
        self._proper_date_list = None
        self._epoch = None
        
        # Compute proper date and epoch is date_list provided
        if date_list is not None:
            self._proper_date_list = self._determine_proper_date(date_list,
                                                            from_str=from_str)
            self._epoch = self._compute_epoch_from_proper_date()
        
        # Compute proper date if epoch given
        if epoch is not None:
            self._epoch = epoch
            self._proper_date_list = self._compute_date_from_epoch()
        
        
        
    # ----- Date: private functions -----
    
    
    # get date from epoch
    def _compute_date_from_epoch(self):
        
        date_info = None
        
        # ----- VERSIONS -----
        
        # PANDAS
        if self.using_pandas:
            date_info = pd.Timestamp( self._epoch, unit="s" )
        
        # PYTHON DATETIME
        if self.using_python_datetime:
            date_info = datetime.datetime.fromtimestamp( self._epoch )
        
        # --------------------
        
        the_date_list = [date_info.year,date_info.month,date_info.day,
                         date_info.hour,date_info.minute,date_info.second]
        
        return the_date_list



    
    
    
    # return epoch
    def _compute_epoch_from_proper_date(self):
        year = self._proper_date_list[0]
        month = self._proper_date_list[1]
        day = self._proper_date_list[2]
        hour = self._proper_date_list[3]
        minute = self._proper_date_list[4]
        second = self._proper_date_list[5]
        
        the_epoch = None
        
        # ----- VERSIONS -----
        
        # PANDAS
        if self.using_pandas:
            pd_timestamp = pd.Timestamp(year=year,month=month,day=day,
                                    hour=hour,minute=minute,second=second)
            pd_stamp_to_daterange = pd.date_range(pd_timestamp,periods=1,freq="d")
            the_epoch = ((pd_stamp_to_daterange - pd.Timestamp("1970-01-01")) //\
                                         pd.Timedelta("1s"))[0]
        
        # PYTHON DATETIME
        if self.using_python_datetime:
            current_time_dt = datetime.datetime(year,month,day,hour,minute)
            the_epoch = time.mktime(current_time_dt.timetuple()) + second
        
        # --------------------
        
        return the_epoch
    
    
    
    # Extract date info from date_list
    # INPUT:
    #   (1) date_list - A variable length list of nums representing a date. Ranges
    #         in length from 1 to 6 elements where successive elements add more
    #         time resolution. Any values not given (besides the year) assume
    #         default values. The syntax is as follows:
    #           1 elem  - Year
    #           2 elems - Year, month
    #           3 elems - Year, month, day
    #           4 elems - Year, month, day, hour
    #           5 elems - Year, month, day, hour, minute
    #           6 elems - Year, month, day, hour, minute, second
    #         Default Vals: Year (0), Month (1), Day (1), Hour (0), Minute (0)
    #           Second (0)
    # Returns a 6-element list denoting the date, from year to second.
    def _determine_proper_date(self,date_list,from_str=None):

        # Handle default arg        
        if from_str is None: from_str = False        
        
        # Deepcopy default date
        date = []
        for i in range(len(self.default_date)):
            date.append( self.default_date[i] )
        max_size = len(date)
        
        # See if too many args given
        if (len(date_list) > max_size):
            error_mssg = "Error! Too many nums provided to get_date_from_list!" + \
                "Should not get more than "+str(max_size)+" but received : "
            raise ValueError(error_mssg,len(date_list))
        
        # Prepare proper date based on default date vals
        for i in range(len(date_list)):
            date[i] = date_list[i]
            
        # If given date as string, then convert to approp numbers
        if from_str:
            date[0] = int( date[0] )
            date[1] = int( date[2] )
            date[2] = int( date[2] )
            date[3] = int( date[3] )
            date[4] = int( date[4] )
            date[5] = float( date[5] )
            
        return date
    
    
    
    # ----- CustomDate: public functions -----
    
    
        
    # return date as single string
    def get_date_single_string(self):        
        datetime_timestamp = self.get_date_as_timestamp()
        return datetime_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    
    
    
    # return date as list of numbers
    def get_date_as_num_list(self):
        return self._proper_date_list
    


    
    
    # returns epoch
    def get_epoch(self):
        return self._epoch
    
    
    
    
    
    # returns date as timestamp (from python's datetime.datetime)
    def get_date_as_timestamp(self):
        
        the_stamp = None
        
        # ----- VERSIONS -----
        
        # PANDAS
        if self.using_pandas:
            the_stamp = pd.to_datetime( self._epoch, unit="s" )
        
        # PYTHON'S DATETIME
        if self.using_python_datetime:
            the_stamp = datetime.datetime.fromtimestamp( self._epoch )
        
        # --------------------
        
        return the_stamp
    
    
    
    
    
    @staticmethod
    def generate_date_range(CustomDateObj_Start,CustomDateObj_End,frequency):
        
        #start_timestamp = datetime.datetime.fromtimestamp( start_epoch )
        #end_timestamp = datetime.datetime.fromtimestamp( end_epoch )
        ###start_timestamp = pd.Timestamp( start_epoch, unit="s" )
        ###end_timestamp = pd.Timestamp( end_epoch, unit="s" )
        
        # Warn user if Python's datetime was used for either CustomDate object
        using_datetime = CustomDateObj_Start.using_python_datetime or \
                            CustomDateObj_End.using_python_datetime
        if using_datetime:
            warn_mssg = "A date range will be generated using pandas libraries. " + \
                        "Sometimes, epoch calculations can differ between Pandas " + \
                        "and Python's Datetime Module (by ~5 hours). Datetime" +  \
                        "has been selected as the engine for at least one of the " + \
                        "provided CustomDate arguments , so there is a possibility " + \
                        "that any such time calculations will contain noticeable " + \
                        "errors."
            warnings.warn(warn_mssg)
                        
        # Make timestamps and compute date range
        start_timestamp = CustomDateObj_Start.get_date_as_timestamp()
        end_timestamp = CustomDateObj_End.get_date_as_timestamp()
        date_ranges = pd.date_range(start=start_timestamp,
                                    end=end_timestamp,freq=frequency)
        
        return date_ranges
        
        
# ================================================================




class SSI_Data_Manager:
        
    
    
    
    # ########## CLASS / "STATIC" VARIABLES ##########
    
    # input files based on variable
    input_files_dict = {
        "X": "th#.xgse", "Y": "th#.ygse", "Z": "th#.zgse",
        "B_X": "th#.fgs_bxgse", "B_Y": "th#.fgs_bygse", "B_Z": "th#.fgs_bzgse",
        "V_X": "th#.peir_vxgse", "V_Y": "th#.peir_vygse", "V_Z": "th#.peir_vzgse",
        "density": "th#.peir_N", "temp": "th#.peir_TeV"
                        }
    
    # instrument names
    legal_inst_names = ["tha","thb","thc","thd","the"]
    
    # columns for data in input files
    raw_data_index_dict = {"year": 0, "month": 1, "day": 2, "hour": 3,
                               "minute": 4, "second": 5, "var": 6, "epoch": 7}
    
    # ################################################
    
    
    # ========== CONSTRUCTOR ==========
    
    def __init__(self,original_data_folders=None,instruments=None,enable_warnings=None,
                 start_date=None,prepared_data_folder=None):
        
        
        # ########## INSTANCE VARIABLES ##########
        
        self.var_names = {
                "time": "time", "X": "X", "Y": "Y", "Z": "Z", "R": "R",
                "B_X": "B_X", "B_Y": "B_Y", "B_Z": "B_Z", "B": "B",
                "V_X": "V_X", "V_Y": "V_Y", "V_Z": "V_Z", "V": "V",
                "density": "density", "temp": "temp"
                            }
                
        self._vector_comps = {
                "R": ["X", "Y", "Z"],
                "B": ["B_X", "B_Y", "B_Z"],
                "V": ["V_X", "V_Y", "V_Z"]
                              }
        
        self._dependent_variables = {
            "planar_angle": "degrees"
                                    }
        
        self.units_dict = {
            **dict.fromkeys(["time"], "secs"),
            **dict.fromkeys(["X","Y","Z","R"], "R_E"),
            **dict.fromkeys(["B_X","B_Y","B_Z","B"], "nT"),
            **dict.fromkeys(["V_X","V_Y","V_Z","V"], "km/s"),
            **dict.fromkeys(["density"], "#/cc"),
            **dict.fromkeys(["temp"], "eV")
                            }
        
        # Initialize instance variables to None (or empty dicts if they're dicts)
        self.instruments = None
        self._enable_warnings = None
        self.start_date = None
        self._raw_data = {}
        self._clean_data = {}
        
        # ###################################################################
        
        
        
        # ----- Handle simple default args issues -----
        
        # Enable warning
        if enable_warnings is None: self.enable_warnings = True
        
        # Check that either original_data_folders or prepared_data are given
        both_orig_and_prepared_given = original_data_folders is not None and \
                                        prepared_data_folder is not None
        neither_orig_or_prepared_given = original_data_folders is None and \
                                        prepared_data_folder is None
        if both_orig_and_prepared_given or neither_orig_or_prepared_given:
            excep_str = "Error! When instantiating SSI_Data_Manager, you must either...\n" + \
                        "(1) supply the location(s) of original data to " + \
                        "original_data_folders, or\n" + \
                        "(2) provide the location of a .csv file containing " + \
                        "cleaned data to prepared_data_file.\n" + \
                        "You cannot do both!!!"
            raise Exception(excep_str)
            
        # If single string given for input original_dat_folders, turn it into list
        # of just that string
        if type(original_data_folders) == str:
            original_data_folders = [ original_data_folders ]
        
        # If only one str given for inst, convert it to single-element list of said str
        if type(instruments) == str: instruments = [ instruments ]
        
        # If instruments provided, confirm that they're legal
        if instruments is not None:
            if self._confirm_legal_inst_names(instruments):
                self.instruments = instruments
        
        # If no instruments provided, just use all
        if instruments is None: instruments = SSI_Data_Manager.legal_inst_names
        
        # ----------------------------------------------



        # ----- DATA READING / ORGANIZING -----
        
        # Here, either a prepared_data_file is supplied from a .csv file, or
        # raw data is read in and converted to organized data

        # Read cleaned data from prepared csvs if provided by constructor
        if prepared_data_folder is not None:
            self._read_clean_data_from_files(prepared_data_folder)
            self._raw_data = None

        # Read raw data from source folders if specified
        if original_data_folders is not None:
            self._raw_data = self._read_raw_data(original_data_folders)
            self.start_date = self._determine_start_date_from_raw_data()
            self._clean_data = self._prepare_clean_data_from_raw_data()
            self._raw_data = None
            
        # Beyond this point, _raw_data is set back to None because it will not
        # be referenced again!
            
        # ---------------------------------------------
        
        
        # Finally, save start_data as list of nums by using Supporting class
        if start_date is not None:
            date_obj = self.CustomDate(start_date)
            self.start_date = date_obj.get_date_as_num_list()

        
        # Notify user of success
        #[start_time,end_time] = self._get_start_and_end_epoch()
        #print("Successfully read in data for instruments ",self.instruments,
        #      " from ",SSI_Data_Manager._convert_epoch_to_date(start_time)," to ",
        #      SSI_Data_Manager._convert_epoch_to_date(end_time))
        
        
        
        
        
        
        
    # ----- Private functions -----
    
    
    
    
    
    
    
    
    # Confirm that instruments assigned to the instruments class variable
    # are all legal
    def _confirm_legal_inst_names(self,possible_instruments):
        
        # Check to make sure no repeated elements are used
        if (len(set(possible_instruments)) != len(possible_instruments)):
            excep_str = "Error! Repeated instrument names used! The given " + \
                        "instrument names should be unique. You provided: " + \
                        " ".join(possible_instruments)
            raise Exception(excep_str)
        
        # Check that all elements are legal
        for elem in possible_instruments:
            if elem not in self.legal_inst_names:
                excep_str = "Error! Illegal name provided for instrument! " + \
                            "Legal names are: " + \
                            " ".join(SSI_Data_Manager.legal_inst_names) + "\n" + \
                            "You provided: " + elem
                raise Exception(excep_str)
                
        # If all unique and legal elements, then provided instrument names are legal
        return True
    
    
    
    
    
    
    # raw_data is a ...
    #  (1)  Dict of ...  (keys are original_data_folders elements [e.g. "files/found/here"])
    #  (2)      Dicts of ...  (instruments used [e.g. "tha" or "thc"])
    #  (3)          Dicts of ...  (variables read in [e.g. "X", "B_Z", or "temp"])
    #                   Dataframes,
    #   where the dataframes contain all the original data read in for a...
    #   *particular* variable (level 3) from a *particular* instrument (level 2)
    #   from a *particular* folder (level 1).
    # For example, raw data could contain the entry:
    # raw_data = { "my/files/are/here": inst_dict }, where
    #     inst_dict = { "tha": variables_dict }, where
    #        variables_dict = { "X": [dataframe1], "Y": [dataframe2], and so on...}.
    def _read_raw_data(self,list_of_source_folders):
        print("===== READING RAW DATA =====\n")
        
        raw_data = {}
        for i in range(len(list_of_source_folders)):
            source_folder = list_of_source_folders[i]
            print("*** (" + str(i+1) + ") ***")
            print("Reading files from folder:\n------->",source_folder)
            raw_data[source_folder] = self._read_raw_data_from_folder(source_folder)
            print("")
        
        print("==============================")
        return raw_data
    
    
    
    
    
    
    
    
    # For source_folder provided, read all data for all instruments in
    # self.instruments from that location
    # returns dict of dicts of dataframes
    def _read_raw_data_from_folder(self,source_folder):
        
        data_in_folder_dict = {}
        for inst in self.instruments:
            print("... for instrument \"" + inst + "\":",end=" ")
            data_in_folder_dict[inst] = \
                self._read_raw_data_from_folder_and_instrument(source_folder,inst)
            print("")
        return data_in_folder_dict
            
            
            
            
            
            
            
    # Given both source folder and instrument, read in all relevant data
    # returns dict of dataframes
    @staticmethod
    def _read_raw_data_from_folder_and_instrument(source_folder,inst):
        
        data_from_folder_and_inst_dict = {}
        for var in list(SSI_Data_Manager.input_files_dict.keys()):
            print(var,end=" ")
            
            # determine filename for var
            filename = SSI_Data_Manager._determine_filename(inst,var)
            
            # The  "\s+" means one or more whitespaces is the delimiter between vals
            source_file = source_folder + "/" + filename
            col_labels = list(SSI_Data_Manager.raw_data_index_dict.keys())
            data_from_folder_and_inst_dict[var] = \
                pd.read_csv(source_file, header=None, 
                            names=col_labels, sep='\s+')
            
        return data_from_folder_and_inst_dict
            
            
            
            
            
            
    
    # determines actual filename for a variable based on instrument
    # e.g. "th#.xgse" --> "tha.xgse" for instrument "tha"
    @staticmethod
    def _determine_filename(inst,var):
        filename = SSI_Data_Manager.input_files_dict[var]
        replacement_letter = inst[-1]    # last letter of instrument name
        return filename.replace("#",replacement_letter)
    
    
    
    
    
    
    # Determines the start date (in the format of a list of numbers) from _raw_data.
    # The earliest date among all entries in raw_data is considered to be the
    # start date.
    # This function assumes that the data read into _raw_data is already sorted
    # with respect to time!
    def _determine_start_date_from_raw_data(self):
        
        # Collect all date info from first rows of all dataframes
        all_first_dates = []
        for folder in list(self._raw_data.keys()):
            for inst in list(self._raw_data[folder].keys()):
                for var in list(self._raw_data[folder][inst].keys()):
                    first_row = self._raw_data[folder][inst][var].iloc[0,:].values
                    a_start_date = self._get_date_data_from_raw_data_row(first_row)
                    all_first_dates.append( CustomDate(date_list=a_start_date) )
                    
        # Convert those dates to epochs and see which one is smallest
        index_of_min_epoch = -1
        min_epoch = np.inf
        for i in range(len(all_first_dates)):
            date = all_first_dates[i]
            new_epoch = date.get_epoch()
            if new_epoch < min_epoch:
                index_of_min_epoch = i
                min_epoch = new_epoch
                
        # Now return date as list of the earliest time
        return all_first_dates[index_of_min_epoch].get_date_as_num_list()
        
        
        
    
    
    
    
    # Get date data from raw_data_row
    # INPUT:
    #   (1) data_row - A 6-element list of nums where the order of the nums
    #         is [year, month, day, hour, minute, second] (which is specified
    #         in the raw_data_index_dict dictionary); quantities year
    #         through minute are integers, but seconds is a float.
    # RETURNS:
    #   Date as numbers (not string!) in format [year,month,day,hour,minute,second].
    def _get_date_data_from_raw_data_row(self,data_row):
        year = int( data_row[SSI_Data_Manager.raw_data_index_dict["year"]] )
        month = int( data_row[SSI_Data_Manager.raw_data_index_dict["month"]] )
        day = int( data_row[SSI_Data_Manager.raw_data_index_dict["day"]] )
        hour = int( data_row[SSI_Data_Manager.raw_data_index_dict["hour"]] )
        minute = int( data_row[SSI_Data_Manager.raw_data_index_dict["minute"]] )
        second = data_row[SSI_Data_Manager.raw_data_index_dict["second"]]
        return [year,month,day,hour,minute,second]
    
    
    
    
    
    
    
    
    # The faster version of converting a date_list to epoch
    # This is faster because there is NO safety-checks against date_list; it is
    # assumed to be exactly 6 elements, or this will crash
    @staticmethod
    def _fast_convert_date_to_epoch(date_list):
        year = date_list[0]
        month = date_list[1]
        day = date_list[2]
        hour = date_list[3]
        minute = date_list[4]
        second = date_list[5]
        current_time_dt = datetime.datetime(year,month,day,hour,minute)
        return time.mktime(current_time_dt.timetuple()) + second
                
        
    
    
    
    
    
    # Converts _raw_data to _clean_data
    # incomplete_points default arg dictates what to do with data points
    # where not all measurements are available (e.g. you want to make |B| but
    # only have B_X and B_Y and no B_Z for that time). Default option sets
    # it to "ignore", which means that such data points are discarded
    def _prepare_clean_data_from_raw_data(self,incomplete_points=None):
        
        if incomplete_points is None: incomplete_points = "ignore"
        
        print("===== PREPARING CLEANED DATA =====\n")
        
        cleaned_data = {}
        for inst in self.instruments:
            print("*** \"" + inst + "\" ***")
            
            # Gather all data at source folder for single instrument...
            single_inst_data = []
            for source_folder in list(self._raw_data.keys()):
                print("From folder",source_folder)
                
                dict_of_data = self._raw_data[source_folder][inst]
                single_inst_data.append(
                    self._prepare_clean_data_from_folder_and_inst(dict_of_data,
                                                            incomplete_points)
                                        )
                
            # ... Then turn those into individual pandas dataframes into
            # single frame
            col_labels = list(single_inst_data[0])
            single_inst_data_as_frame = pd.DataFrame(columns=col_labels)
            for a_frame in single_inst_data:
                single_inst_data_as_frame = pd.concat(
                    [single_inst_data_as_frame, a_frame], ignore_index=True
                    )
            
            # And assign that single frame as entry into dict with key being
            # the instrument
            cleaned_data[inst] = single_inst_data_as_frame
            
            print("")
            
        print("==============================")
            
        return cleaned_data
                
                
                
                        
                        
                        
                        
                        
                        
                        
    # Given a data_dict (dict of dataframes) and string denoting how to treat
    # data points lacking measurements (incomplete_points), build a single
    # dataframe whose columns are the keys of data_dict.
    # --- currently, only ignore incomplete points is supported (i.e. no
    # --- interpolation).
    def _prepare_clean_data_from_folder_and_inst(self,data_dict,incomplete_points,
                                                 report_progress=None):
        
        if report_progress is None: report_progress = 4
        
        # Create dict with key-values of var names and last saved indices
        # Since the files are sorted relative to time, then can just look at
        # indices beyond the last saved index for a variable to see where
        # the similar time is at
        last_saved_indices = { key:-1 for key in list(data_dict.keys()) }
        num_rows_per_var = { key:data_dict[key].values.shape[0] \
                            for key in list(data_dict.keys()) }
        
        # All times / dates will have to be compared to some reference time / date,
        # so let's just pick the first one in the dict to be the reference
        ref_var_for_time = list(data_dict.keys())[0]
        all_other_vars = [ elem for elem in list(data_dict.keys()) \
                          if elem != ref_var_for_time]
        var_col_index = self.raw_data_index_dict["var"]
        
        # initialize dict of empty lists; these lists will contains measurements
        # that have been confirmed to have matching times / dates
        vals_at_matching_times = { key:[] for key in list(data_dict.keys()) }
        saved_times = []
        
        if incomplete_points == "ignore":

            # Iterate over rows of ref_var, checking against times of other vars
            num_rows_ref = data_dict[ref_var_for_time].values.shape[0]
            printed_times = 1
            end_of_file_reached = False
            for i in range(num_rows_ref):
                if report_progress > 0:
                    if i==0: print("--[progress]-->: ",end="")
                    frac_processed = i / num_rows_ref
                    if ( frac_processed > printed_times * (1/report_progress) ):
                        print("{:.2f}".format(frac_processed*100)+"%.. ",end="")
                        printed_times = printed_times + 1
                
                # save date info and var value for ref_var...
                data_row = data_dict[ref_var_for_time].iloc[i,:].values
                ref_epoch = CustomDate(
                            date_list=self._get_date_data_from_raw_data_row(
                                                     data_row
                                                                         )
                                     ).get_epoch()
                #ref_epoch =  date_obj.get_epoch()
                var_value = data_row[var_col_index]
                
                # initialize dict for row and save measurement from ref var
                vals_at_date = {}
                vals_at_date[ref_var_for_time] = var_value
                
                
                
                # ----- Grabbing data from other vars and comparing -----
                # ----- times against the ref_var time ------------------
                
                # Then compare against other vars, saving those with same dates
                num_vals_found_for_date = 1
                for var in all_other_vars:
                    
                    # grab index from last_saved_indices to shorten where to look
                    start_row_index = last_saved_indices[var]+1
                    end_of_file_reached = start_row_index == num_rows_per_var[var]\
                                            or end_of_file_reached
                    matching_time_found = False
                    for row_index in range(start_row_index,num_rows_per_var[var]):
                    
                        # save date info and var value from var
                        other_var_data_row = data_dict[var].iloc[row_index,:].values
                        #other_var_date_data = \
                        #    self._get_date_data_from_raw_data_row(other_var_data_row)
                        #other_var_epoch = SSI_Data_Manager.\
                        #    _fast_convert_date_to_epoch(other_var_date_data)
                        other_var_epoch = CustomDate(
                            date_list=self._get_date_data_from_raw_data_row(
                                                     other_var_data_row
                                                                           )
                                                     ).get_epoch()
                    
                        # If dates match, then save var value and index                            
                        if ref_epoch == other_var_epoch:
                            last_saved_indices[var] = row_index
                            vals_at_date[var] = other_var_data_row[var_col_index]
                            matching_time_found = True
                            num_vals_found_for_date = num_vals_found_for_date + 1
                            break
                            
                        # Times are sorted, so if other_var_epoch starts to
                        # be larger than ref_epoch, just break out
                        if ref_epoch < other_var_epoch:
                            matching_time_found = False
                            break
                        
                    # If no match, then no need to check rest of indices
                    if not matching_time_found: break
                
                # --------------------------------------------------------
                
                
                
                # If a data point was found where a measurement was not available
                # for all variables, then skip onto the next time / date
                #if not matching_time_found:
                if num_vals_found_for_date != len(list(data_dict.keys())):
                    continue
                
                
                
                # ----- Save data from rows if measurements found for -----
                # ----- all vars ------------------------------------------
                
                # save time info as epoch
                saved_times.append( ref_epoch )
                
                # save measurements found on row
                if len(list(data_dict.keys())) != len(list(vals_at_date.keys())):
                    print("Vals missing at ref row",i)
                    print("vals with measurements only",list(vals_at_date.keys()))
                for var in list(data_dict.keys()):
                    vals_at_matching_times[var].append( vals_at_date[var] )
                
                # ---------------------------------------------------------
            if report_progress > 0: print("100%")
                
            
            # Turn all saved lists into arrays
            saved_times = np.array( saved_times )
            for var in list(data_dict.keys()):
                vals_at_matching_times[var] = np.array( vals_at_matching_times[var] )
            
            # Build any vector magnitudes once all measurements with matching
            # times are compiled
            # This adds the vector magnitudes to the supplied dict!
            self._build_vector_mags(vals_at_matching_times)
                
            # Turn all saved data into single numpy array
            num_cols = len(list(vals_at_matching_times.keys()))+1
            col_labels = list(vals_at_matching_times.keys())
            
            # save times as first column, but with times being counted relative
            # to first time.
            shifted_times = saved_times - CustomDate(date_list=self.start_date).get_epoch()
            #saved_times[0]
            col_labels.insert(0,"time")    # put time first
            out_arr = np.zeros(( len(saved_times) , num_cols ))
            out_arr[:,0] = shifted_times
            
            # Then sequentially save the other vars by whatever ordering the
            # list of keys of data_dict provides
            i = 0
            for var in list(vals_at_matching_times.keys()):
                i = i + 1
                out_arr[:,i] = vals_at_matching_times[var]
                
            return pd.DataFrame(out_arr,columns=col_labels)
        
        
        
        
        
        
        
        
        
        
    # Given a dict with keys being variable names and values being arrays
    # (where similar indices across them means those measurements occur at
    # the same time), compute the vector magnitudes as prescribed in
    # the class var _vector_comps
    # THIS WILL MODIFY THE GIVEN DICT! THE CONSTRUCTED LIST OF VECTOR
    # MAGNITUDES WILL BE ADDED TO IT!
    def _build_vector_mags(self,dict_of_vectors):
        
        # For each vector mag, get components and build it
        for vector_mag_var in list(self._vector_comps.keys()):
            comp_vars = self._vector_comps[vector_mag_var]
            
            # get comps
            x_vals = dict_of_vectors[comp_vars[0]]
            y_vals = dict_of_vectors[comp_vars[1]]
            z_vals = dict_of_vectors[comp_vars[2]]
            vec_mags = np.zeros( len(x_vals) )
            
            # compute magnitude
            vec_mags = np.sqrt( x_vals**2 + y_vals**2 + z_vals**2 )
            
            # add to dict
            dict_of_vectors[vector_mag_var] = vec_mags
            
    
    
    
    
    
    
    
    
    
    # If data is being read in from .csv files that have already been prepared
    # on a previous run, then the data processing is much simpler and faster
    # than processing it all from scratch again. Simply provide a string
    # denoting the global location of the file containing the .csvs, and a
    # .csv for each instrument will be read in.
    def _read_clean_data_from_files(self,folder_loc):
        print("Reading cleaned data from files for instruments: ",end="")
        
        # confirm that file exists
        if not os.path.exists(folder_loc):
            excep_mssg = "Error! Folder " + folder_loc + "does not exist!"
            raise Exception(excep_mssg)
        
        start_date_header_index = 0
        instrument_header_index = 1
        num_header_lines = 2
        start_dates = []
        
        # get files based on inst names
        inst_files = [folder_loc + "/" + elem + ".csv" for elem in self.instruments]
        
        cleaned_data = {}
        for file_loc in inst_files:
            
            # Collect header info
            with open(file_loc) as myfile:
                header_info = [next(myfile) for x in range(num_header_lines)]
                
            # Read start date
            read_start_date_line = header_info[start_date_header_index]
            read_start_date_str = read_start_date_line\
                            [read_start_date_line.find("=")+1:].strip()
            read_start_date_str = read_start_date_str\
                [1:len(read_start_date_str)-1].split(",")
            start_dates.append( read_start_date_str )
            
            # Read instrument
            read_inst_line = header_info[instrument_header_index]
            read_inst_str = read_inst_line[read_inst_line.find("=")+1:].strip()
            self._confirm_legal_inst_names( [read_inst_str] )            
            print(read_inst_str,end=" ")
            
            # now read data for dataframe if inst is among instruments user specifies
            cleaned_data[ read_inst_str ] = \
                pd.read_csv(file_loc,skiprows=num_header_lines)
            
            # remove first col from csv (which is just for counting rows)
            cleaned_data[read_inst_str].drop\
                (cleaned_data[read_inst_str].columns[0],axis=1,inplace=True)
                
        print("")
        
        # Save data into dict
        self._clean_data = cleaned_data
        
        # Check that date is consistent; if it isn't, then throw exception...
        same_start_date = True
        for elem in start_dates:
            same_start_date = same_start_date and elem == start_dates[0]
        if not same_start_date:
            excep_mssg = "Error! When reading clean data from file, found " + \
                         "different dates! Dates should be consistent."
            raise Exception(excep_mssg)
        
        # ... Otherwise, save date
        else: 
            date_obj = CustomDate(date_list = start_dates[0],from_str=True)
            self.start_date = date_obj.get_date_as_num_list()
    
    
        
            
    
    
    
    
    
    
    
    
    # Create list of dates based on start and end (given as epochs) and a
    # frequency (which is week ("w"), day ("d"), year ("y"), etc)
    def _create_list_of_dates_from_date_range(self,start_epoch,end_epoch,frequency):
        
        if frequency == "month": frequency = "MS"
        if frequency == "year": frequency = "YS"
        if frequency == "day": frequency = "D"
        if frequency == "week": frequency = "7D"
        
        # --- So there seems to be a discrepancy between datetime's epoch ---
        # --- calculator and pandas's version; giving them both the epoch ---
        # --- 10**9 seconds yields [2001,9,8,21,46,40] for datetime and -----
        # --- [2001,9,9,1,46,40] for pandas; about 5 hours difference!!! ----
        # --- Let's just be consistent and only use one...
        #start_timestamp = datetime.datetime.fromtimestamp( start_epoch )
        #end_timestamp = datetime.datetime.fromtimestamp( end_epoch )
        ###start_timestamp = pd.Timestamp( start_epoch, unit="s" )
        ###end_timestamp = pd.Timestamp( end_epoch, unit="s" )
        
        # Turn epochs into timestamp and create range of dates        
        CustomDate_Start = CustomDate(epoch=start_epoch)
        CustomDate_End = CustomDate(epoch=end_epoch)
        date_ranges = CustomDate.generate_date_range(CustomDate_Start,
                                                     CustomDate_End,
                                                     frequency)
        
        # Then convert the data in the range into date_lists
        list_of_dates = []
        for i in range(len(date_ranges)):
            list_of_dates.append(
                    [date_ranges[i].year,date_ranges[i].month,date_ranges[i].day,
                     date_ranges[i].hour,date_ranges[i].minute,date_ranges[i].second]
                )
            
        # If no bounds were generated, then forcibly insert start date
        if len(date_ranges) == 0:
            list_of_dates.append( CustomDate_Start.get_date_as_num_list() )
            
        # check if last date in list_of_dates matches end_epoch; if not,
        # manually insert the corresponding date at the end.
        final_date = CustomDate(epoch=end_epoch).get_date_as_num_list()
        if list_of_dates[-1] != final_date: list_of_dates.append(final_date)
        
        return list_of_dates
    
    
    
    
    
    
    
    
    
    # Determine position of subplot based on variable being plotted
    def _make_subplot_params(self,var_str):
        
        # common vector columns
        posit_col = 0
        B_col = 1
        v_col = 2
        else_col = 3
        
        # intializer for axes variable
        axes = (-1,-1)
        
        # Position data
        if (var_str == "X"):
            axes = (0,posit_col)
        elif (var_str == "Y"):
            axes = (1,posit_col)
        elif (var_str == "Z"):
            axes = (2,posit_col)
        elif (var_str == "R"):
            axes = (3,posit_col)
        
        # Mag Field data
        if (var_str == "B_X"):
            axes = (0,B_col)
        elif (var_str == "B_Y"):
            axes = (1,B_col)
        elif (var_str == "B_Z"):
            axes = (2,B_col)
        elif (var_str == "B"):
            axes = (3,B_col)
            
        # Velocity data
        if (var_str =="V_X"):
            axes = (0,v_col)
        elif (var_str == "V_Y"):
            axes = (1,v_col)
        elif (var_str == "V_Z"):
            axes = (2,v_col)
        elif (var_str == "V"):
            axes = (3,v_col)
            
        # All other data
        if (var_str == "density"):
            axes = (0,else_col)
        elif (var_str == "temp"):
            axes = (1,else_col)
        """elif (var_str == self.var_names["perp-temp"]):
            axes = (2,else_col)"""
            
        return axes
    
    
    
    
    
    
    
    # Returns the "formal" name for a variable, which is just the variable name
    # followed by the units in brackets (e.g. "X" --> "X [R_E]")
    def _get_formal_var_name(self,var_str):
        return self.var_names[var_str] + " (" + self.units_dict[var_str] + ")"
    
    
    
    
    
    
    
    
    
    # Returns the function that would convert a time value in old_units to 
    # new_units (but naively so - i.e., does NOT do appropriate epoch conversion;
    # assumes uniform 24 exact hours in a day, etc.)
    # Current units supported are:
    #   secs, mins, hours, days, and weeks (naively calculated, e.g. 1 week = 
    #       7 * 86400 secs)
    def _naive_time_calculator(old_units,new_units):
        
        
        # --- Convert from old_units to seconds ---
        # old_units --> seconds
        old_units_to_seconds = {
            "weeks": lambda x: x * 86400 * 7,
            "days": lambda x: x * 86400,
            "hours": lambda x: x * 3600,
            "mins": lambda x: x * 60,
            "secs": lambda x: x
                            }
        # Confirm that units for units_to_seconds exists
        if (old_units not in old_units_to_seconds.keys()):
            error_mssg = "Error! Units provided \"" + old_units + \
                "\" is not an option. Possible options include: "
            raise ValueError(error_mssg,list(old_units_to_seconds.keys()))
        # -----------------------------------------
        
        
        # --- Then convert from seconds to new_units ---
        # seconds --> new_units
        seconds_to_new_units = {
            "weeks": lambda x: x / (86400 * 7),
            "days": lambda x: x / 86400,
            "hours": lambda x: x / 3600,
            "mins": lambda x: x / 60,
            "secs": lambda x: x
                            }
        
        # Confirm that new_units for seconds_to_new_units exists
        if (new_units not in seconds_to_new_units.keys()):
            error_mssg = "Error! Units provided \"" + new_units + \
                "\" is not an option. Possible options include: "
            raise ValueError(error_mssg,list(seconds_to_new_units.keys()))
        # -------------------------------------------------
        
        
        # Combine the lambda functions into single lambda
        def function_composition(outer_func,inner_func):
            return lambda x: outer_func(inner_func(x))        
        return_func = function_composition(seconds_to_new_units[new_units],
                                           old_units_to_seconds[old_units])
        
        return return_func
    
    
    
    
    
    
    
    
    # which_inst is either single inst name, list of inst names, or None
    # Returns the earliest and latest epochs found for the instrument provided
    # to which_inst (and if none is given, then it will search across all instruments)
    # INPUT:
    #   which_inst (string or list of strings): Either a string denoting the name
    #                                            of an individual instrument or a
    #                                            list of instrument names.
    def _get_earliest_and_latest_epochs(self,which_inst=None):
        
        start_epoch = CustomDate(date_list=self.start_date).get_epoch()
        
        # If no instrument specified, then find it globally, i.e. across all
        # instruments
        if which_inst is None: instruments_to_check = self.instruments
        
        # Gather the earliest and latest times from each instrument specified
        earliest_times, latest_times = [], []
        for inst in instruments_to_check:
            earliest_times.append( self._clean_data[inst]["time"].values[0] )
            latest_times.append( self._clean_data[inst]["time"].values[-1] )
            
        # And be sure to account for current time units (in case they're not
        # already in seconds)
        time_conv_current_units_to_secs = SSI_Data_Manager.\
                    _naive_time_calculator(self.units_dict["time"], "secs")
        earliest_times_in_seconds = [ time_conv_current_units_to_secs(elem) \
                                     for elem in earliest_times ]
        latest_times_in_seconds = [ time_conv_current_units_to_secs(elem) \
                                     for elem in latest_times ]
        
        return [min(earliest_times_in_seconds) + start_epoch,
                max(latest_times_in_seconds) + start_epoch]
    
    
    
    
    
    
    
    
    # converts given time val (which is in the units of self.units_dict["time"])
    # and converts it to epoch time    
    def _convert_epoch_to_current_units(self,epoch_val):        
        start_epoch = CustomDate(date_list=self.start_date).get_epoch()
        time_conv_secs_to_current_units = SSI_Data_Manager.\
                    _naive_time_calculator("secs",self.units_dict["time"])
        return time_conv_secs_to_current_units(epoch_val - start_epoch)
    
    
    
    
    
    
    
    
    
    
    # Computes all values according to var_name (from the class dictionary
    # _dependent_variables, which details the variables required to compute
    # var_name) from the dataframe provided (the_df).
    def _compute_dependent_variables(self,var_name,the_df):
        
        depend_vars = list(self._dependent_variables.keys())
        
        # Check that var_name provided is supported
        if var_name not in depend_vars:
            excep_mssg = "Error! The supplied variable name " + var_name + \
                         " is not supported. The list of supported dependent " + \
                         "variables is [" + ",".join(depend_vars) + "]."
            raise Exception(excep_mssg)
            
        # Compute var_name according to function definition
        
        # the X-Y angle in the plane (e.g. MLT in GSM)
        if var_name == "planar_angle":

            return_vals = np.arctan2(the_df["Y"],the_df["X"]) * 180/np.pi
            for i in range(len(return_vals)):
                if return_vals[i] < 0: return_vals[i] = return_vals[i] + 360
        
        return return_vals
    
    
    
    
    
    
    
    
    # ----- PUBLIC FUNCTIONS -----
    
    
    
    
    
    
    # Returns all data computed by SSI_Data_Manager in the format of either
    # a dictionary of dataframes (the keys of which are the instrument names) or
    # a single dataframe.
    # INPUT:
    #   (1) single_frame (default arg = False):
    #           Boolean that dictates if the user wants all cleaned data returned
    #           as a single dataframe or a dictionary of dataframes
    #   (2) add_instrument_col (default arg = False):
    #           Boolean that, if True, will add a column to each dataframe
    #           indicating from which instrument each data point came from.
    #           This works even regardless of the value of single_frame.
    # RETURNS:
    #   Either dict of dataframes or single dataframe (see single_frame default arg).
    def get_data(self,single_frame=None,add_instrument_col=None):
        
        # Handle default args
        if single_frame is None: single_frame = False
        if add_instrument_col is None: add_instrument_col = False
        
        # Make deep copy of each dataframe in dict
        dict_of_frames = {}
        for key in list(self._clean_data.keys()):
            dict_of_frames[key] = self._clean_data[key].copy()
        
        # Turn each column label (e.g. "X") into proper name (e.g. "X (R_E)")
        for inst in self.instruments:
            for var_name in list(self.var_names.keys()):
                proper_var_name = self._get_formal_var_name(var_name)
                dict_of_frames[inst].rename(columns={var_name:proper_var_name},inplace=True)
        
        # Add columns showing instruments, if specified
        if add_instrument_col:
            for inst in list(dict_of_frames.keys()):
                (dict_of_frames[inst])["instrument"] = inst
        
        # If user just wants dict of frames, then return it
        if not single_frame: return dict_of_frames
        
        # Otherwise, stack them
        else:
            col_labels = list(dict_of_frames[ list(dict_of_frames.keys())[0] ])
            entire_frame = pd.DataFrame(columns=col_labels)
            for inst in list(dict_of_frames.keys()):
                entire_frame = pd.concat( [entire_frame,dict_of_frames[inst]],
                                          ignore_index=True )
            return entire_frame
        
    
    
    
    
    
    
    # Saves the data processed to .csv files (per instrument) at the global file
    # location given by the string path_to_folder. These .csvs can then be read
    # in to this same program later if the folder location is given to the
    # constructor by using the prepared_data_folder argument.
    # NOTE: If the given path does not exist, it will be created.
    def save_data_to_csv(self,path_to_folder):
        print("Writing data to files in",path_to_folder)
        print("--Created files-->: ",end="")
        
        # make folder if path_to_folder does not exist
        if not os.path.exists(path_to_folder): os.makedirs(path_to_folder)
        
        # Then save each inst frame to its own file, with comments as headers
        for inst in list(self._clean_data.keys()):
            
            # Each data for an instrument has its own file
            filename = inst + ".csv"
            full_filename = path_to_folder + "/" + filename
            the_file = open(full_filename, 'w')
            print(filename,end=" ")
            
            # write header info to file
            header_first_line = "# start_date = [" + \
                    ",".join([str(elem) for elem in self.start_date]) + "]" 
            header_second_line = "# instrument = " + inst
            header_info = header_first_line + "\n" + header_second_line + "\n"
            the_file.write(header_info)
            the_file.close()
            
            # write dataframe into to file
            self._clean_data[inst].to_csv(full_filename,mode="a")
        print("")
        
    
    
    
    
    
    
    
    
    
    # Create a 4x4 series of plots for each instrument;
    # The subplots are of the quantities measured and their vector magnitudes.
    # Exact positioning of which variables occupy which subplot can be changed
    # in the _get_subplot_params function.
    # The figures are saved to the fig_folder string argument.
    # INPUT:
    #  plot_duration (default arg, string) - String denoting the length of time over
    #     which separate plots are to created; can be any of the following:
    #       "day", "week", "month", or "year"
    #  add_to_figname (default arg, string) - Additional string to append to
    #     created filenames; handy for when you plan on trimming data and want
    #     to save the trimmed data in separate plots from the originals.
    # NO RETURNS
    def plot_data(self,fig_folder,plot_duration=None,add_to_figname=None):
        
        # make fig folder if DNE
        if not os.path.exists(fig_folder): os.makedirs(fig_folder)
        
        # set default args
        if plot_duration is None: plot_duration="month"    # meaning month
        if add_to_figname is None: add_to_figname = ""
        
        # make functions to convert between seconds (for epoch) and current
        # time units
        #time_conv_current_units_to_secs = SSI_Data_Manager.\
        #            _naive_time_calculator(self.units_dict["time"], "secs")
        time_conv_secs_to_current_units = SSI_Data_Manager.\
                    _naive_time_calculator("secs", self.units_dict["time"])
        
        # determine dates to plot over
        """start_epoch = CustomDate(date_list=self.start_date).get_epoch()
        final_times = []
        for inst in list(self._clean_data.keys()):
            final_times.append( self._clean_data[inst]["time"].values[-1] )
        final_seconds = [ time_conv_current_units_to_secs(elem) \
                         for elem in final_times ]
        final_epochs = np.array(final_seconds) + start_epoch"""
        _, final_epoch = self._get_earliest_and_latest_epochs()
        start_epoch = CustomDate(date_list=self.start_date).get_epoch()
        list_of_dates = self._create_list_of_dates_from_date_range\
                (start_epoch,final_epoch,plot_duration)
        #print(list_of_dates)
        
        print("Creating",len(list_of_dates)-1,"plots, separated by " + \
              plot_duration + ", per instrument: ",end="")
        
        # For each instrument...
        for inst in self.instruments:
            print(inst,end=" ")
            
            # Create large macro-plot for each period of time plot_duration
            for i in range(len(list_of_dates)-1):
                inst_data = self._clean_data[inst]
                
                # gather start date info
                plot_start_date = list_of_dates[i]
                plot_start_seconds_since_start_date = \
                    CustomDate(date_list=plot_start_date).get_epoch() - start_epoch
                plot_start_in_current_units = \
                        time_conv_secs_to_current_units(
                            plot_start_seconds_since_start_date
                                                         )
                # gather end date info
                plot_end_date = list_of_dates[i+1]
                plot_end_seconds_since_start_date = \
                    CustomDate(date_list=plot_end_date).get_epoch() - start_epoch
                plot_end_in_current_units = \
                        time_conv_secs_to_current_units(
                            plot_end_seconds_since_start_date
                                                         )
                
                # make plot and assign empty subplots
                fig, axes = plt.subplots(4, 4, figsize=(12,12), dpi=120)
                axes[3,3].axis("off")    # Not using the last subplot
                axes[2,3].axis("off")
                
                
                
                # ----- Create subplot for each variable -----
                
                # grab indices from numpy array from dataframe that have the
                # correct times
                more_than_indices = np.where(inst_data["time"].values >= \
                                        plot_start_in_current_units)
                less_than_indices = np.where(inst_data["time"].values <= \
                                        plot_end_in_current_units)
                right_time_indices = np.intersect1d(more_than_indices,less_than_indices)
                #print("more than:",more_than_indices)
                #print("less than:",less_than_indices)
                #print("the indices:",right_time_indices)
                
                
                # Knowing indices, grab the times
                x_vals = [ inst_data["time"].values[elem] for elem in right_time_indices ]
                x_vals = [ elem - plot_start_in_current_units for elem in x_vals ]

                # Knowing the indices of the dataframe's numpy array where the
                # times match up, make the plots                
                for the_var in list(inst_data.keys()):
                    
                    if the_var == "time": continue
                    
                    # Get axes and variable type
                    axis_posit = self._make_subplot_params(the_var)
                    ax = axes[ axis_posit[0], axis_posit[1] ]
                    
                    # Get and plot data
                    y_vals = [ inst_data[the_var].values[elem] for elem in right_time_indices ]
                    ax.scatter(x_vals,y_vals)
                    
                    # Change x label and title based on units
                    ax.set_xlabel(self._get_formal_var_name("time") + " since start time")
                    ax.set_title(self._get_formal_var_name(the_var))
                # --------------------------------------------
                
                
                
                # Build title from date info and compress layout
                plot_start_date_str = CustomDate(date_list=plot_start_date)\
                                    .get_date_single_string()
                plot_end_date_str = CustomDate(date_list=plot_end_date)\
                                    .get_date_single_string()
                #start_and_end_times = self._get_start_and_end_epoch()
                #start_date = SSI_Data_Prepper._convert_epoch_to_date(start_and_end_times[0])
                #end_date = SSI_Data_Prepper._convert_epoch_to_date(start_and_end_times[1])
                title_str = inst.upper() + " Data from \n" + plot_start_date_str + \
                    " to " + plot_end_date_str
                fig.suptitle(title_str, fontsize=24)
                fig.tight_layout()
                
                # Build base / original filename
                plot_start_date_str = plot_start_date_str.split()[0].replace("-","_")
                plot_end_date_str = plot_end_date_str.split()[0].replace("-","_")
                figname = inst.upper() + "_FROM_" + plot_start_date_str + \
                            "_TO_" + plot_end_date_str
                    
                # Check if modifications are made to filename; then save fig and close
                figname = figname + add_to_figname + ".png"
                fig.savefig(fig_folder + "/" + figname)
                plt.close()
        print("")
                
                
    
    
    
    
    
    
    
    
    
    # Trims clean_data based on variable name var_name according to the min/max given
    # by min_val and max_val. This works for any variable in the dataframe, but
    # trimming according to time has a unique input. To run, either min_val or
    # max_val must be set (but both can be provided if you want to select an
    # interval).
    # And note for the future: Assumes all dataframes have same length!
    # TRIMMING BY TIME:
    #   For time, the min_val and max_val default args are to be the dates
    #   expressed as lists of numbers. Say I wanted to trim data from 12 Dec
    #   2019, 01:00 hours to 19 Dec 2019, 17:50:37 (that is, hour 17, minute 50,
    #   and the 37th second). The function call would look like this:
    #     example: trim_data("time",min_val=[2019,12,12,1],
    #                             max_val=[2019,12,19,17,50,37])
    # TRIMMING BY ANY OTHER VARIABLE:
    #   For any other variable, the values to provide to min/max val are simply
    #   numbers in the units of that variable. Here are some examples:
    #     Keeping all R's > 7 R_E:  trim_data("R",min_val=7)
    #     Keeping all B's < 100 nT:  trim_data("B",max_val=100)
    #     Keeping all n's between 10 and 2000 #/cc:  trim_data("density",min_val=10,max_val=2000)
    # INPUT:
    #   (1) var_name - The name of the variable as seen in clean_data_names_dict
    #   (default arg): min_val - data < this val will be removed
    #   (default arg): max_val - data > this val will be removed
    #   (default arg, Boolean): remove_interval - If true, then data found WITHIN
    #        the interval min_val < data < max_val will be removed. If not set,
    #        then data OUTSIDE of the interval will be removed!
    # NO RETURNS
    def trim_data(self,var_name,min_val=None,max_val=None,remove_interval=None):
        
        if remove_interval is None: remove_interval = False
        
        # Confirm that user provided min_val, max_val, or both
        if min_val is None and max_val is None:
            error_mssg = "Error! Neither min_val nor max_val have been specified." + \
                  "When trimming data, one of the 3 options must be met: " + \
                  "(1) a min_val is provided, (2) a max_val is provided, or " + \
                  "(3) both min_val and max_val are provided."
            raise Exception(error_mssg)
            
        # Determine trimming conditions based on min_val / max_val
        only_min = max_val is None
        only_max = min_val is None
        min_and_max = not only_min and not only_max
            
        # Can only set remove_interval if interval is given
        if remove_interval and not min_and_max:
            error_mssg = "Error! When specifying removal of interval, you need " + \
                        "to provide an entire interval! Not just min_val or max_val, " + \
                        "but both!"
            raise Exception(error_mssg)
            
        # See if date provided (only works for time)
        date_provided = type(min_val) == list or type(max_val) == list
        
        # Check here that min_val is indeed < max_val!!!
        
        
        
        # ----- PRINTING TRIMMING ACTIVITY TO TERMINAL -----
        
        # If actual numbers instead of lists (i.e. dates) given, this is easy
        mssg_to_term = ""
        if not date_provided:
            
            # determine units
            current_units = None
            if var_name in list(self.var_names.keys()):
                current_units = self.units_dict[var_name]
            if var_name in list(self._dependent_variables.keys()):
                current_units = self._dependent_variables[var_name]            
            
            # print bounds and units for non-dates
            if only_max:
                mssg_to_term = "Removing data where " + var_name + " > " + str(max_val) + \
                            " " + current_units + " for instruments: "
            elif only_min:
                mssg_to_term = "Removing data where " + var_name + " < " + str(min_val) + \
                            " " + current_units + " for instruments: "
            elif min_and_max and not remove_interval:
                mssg_to_term = "Removing data outside of interval " + str(min_val) + " < " + \
                                var_name + " < " + str(max_val) + " " + current_units + \
                                " for instruments: "
            elif min_and_max and remove_interval:
                mssg_to_term = "Removing data INSIDE of interval " + str(min_val) + " < " + \
                                var_name + " < " + str(max_val) + " " + current_units + \
                                " for instruments: "
                
        # If dates, then have to handle the printing a little differently
        else:
            if only_max:
                mssg_to_term = "Removing data after date [" + ",".join(map(str,max_val)) + \
                                "] for instruments: "
            if only_min:
                mssg_to_term = "Removing data before date [" + ",".join(map(str,min_val)) + \
                                "] for instruments: "
            if min_and_max and not remove_interval:
                mssg_to_term = "Removing data outside of dates [" + ",".join(map(str,min_val)) + \
                                "] and [" + ",".join(map(str,max_val)) + "] for instruments: "
            elif min_and_max and remove_interval:
                mssg_to_term = "Removing data BETWEEN dates [" + ",".join(map(str,min_val)) + \
                                "] and [" + ",".join(map(str,max_val)) + "] for instruments: "
                                
        print(mssg_to_term,end="")
        
        # --------------------------------------------------------
        
        
        
        # ----- CONVERTING TIME DATES TO CURRENT TIME UNITS -----
        
        # check to see if date was provided for time; can tell it's a date b/c
        # it would be a list
        if var_name == "time":
            
            # if dates given...
            if date_provided:
                
                # ... then need to convert those dates to epochs ...
                dates_to_convert = [min_val, max_val]
                dates_as_epochs = [None,None]
                for i in range(len(dates_to_convert)):
                    if dates_to_convert[i] is None: continue
                    dates_as_epochs[i] = \
                        CustomDate(date_list=dates_to_convert[i]).get_epoch()
                        
                # ... and those epochs to the current time units
                dates_in_current_units = [None,None]
                for i in range(len(dates_to_convert)):
                    if dates_as_epochs[i] is None: continue
                    dates_in_current_units[i] = \
                        self._convert_epoch_to_current_units(dates_as_epochs[i])
                
                # ... then finally reassign those converted dates into their
                # proper units
                min_val, max_val = dates_in_current_units
                
        # ----------------------------------------------------------
            



        # ----- Trim data for each instrument -----
        
        for inst in self.instruments:
            print(inst + " ",end="")
            inst_data = self._clean_data[inst]
            indices_to_remove = None
            data_to_trim = None
            
            # if var_name is explicitly one of the variables, then this is pretty simple;
            # just check whenever vals in that column exceed max / fall-below min
            if var_name in list(self.var_names.keys()):
                data_to_trim = self._clean_data[inst][var_name]
                
            # However, if var_name is a variable that is DERIVED from the
            # variables present in the dataframes, we'll have to manually compare
            elif var_name in list(self._dependent_variables.keys()):
                data_to_trim = self._compute_dependent_variables(
                                                            var_name,
                                                            self._clean_data[inst]
                                                                )
        
            # Determine which row indices have data points that are beyond
            # the bounds of data we want to keep
            if only_max:
                indices_to_remove = np.where( data_to_trim > max_val )
            elif only_min:
                indices_to_remove = np.where( data_to_trim < min_val )
            elif min_and_max and not remove_interval:
                indices_lower_than_min_val = np.where( data_to_trim < min_val )
                indices_higher_than_max_val = np.where( data_to_trim > max_val )
                indices_to_remove = np.union1d(indices_lower_than_min_val,
                                                 indices_higher_than_max_val)
            elif min_and_max and remove_interval:
                indices_greater_than_min_val = np.where( data_to_trim > min_val )
                indices_lower_than_max_val = np.where( data_to_trim < max_val )
                indices_to_remove = np.intersect1d(indices_greater_than_min_val,
                                                 indices_lower_than_max_val)
            
            """# However, if var_name is a variable that is DERIVED from the
            # variables present in the dataframes, we'll have to manually compare
            else:
                
                # compute values of dependent variable
                vals = self._compute_dependent_variables(var_name,self._clean_data[inst])
                
                # Then determine indices that contain undesired values like above
                if min_val is None: indices_to_remove = np.where( vals > max_val )
                elif max_val is None: indices_to_remove = np.where( vals < min_val )
                else:
                    indices_lower_than_min_val = np.where( vals < min_val )
                    indices_higher_than_max_val = np.where( vals > max_val )
                    indices_to_remove = np.union1d(indices_lower_than_min_val,
                                                     indices_higher_than_max_val)"""
                
            
            # Knowing the indices to remove, hand that list to Pandas for removal
            inst_data.drop(inst_data.index[indices_to_remove], inplace=True)
            
            # resets row count for frame; not needed when trimming data once,
            # but trimming consecutive times will lead to errors and mis-indexing
            # without this!
            inst_data.reset_index(drop=True,inplace=True)    
        print("")
                
                
    
    
    
    
    
    
    # Modifies names and units for variables. The optional name change will only
    # change the "formal" name (the name is for any printing or plotting), but
    # it will not change the name required to call certain data (e.g. mydata["X"]).
    # INPUT:
    #   (1) var_name (str): Name of variable to modify units of
    #   (2) new_units (str):  Name of new units var_name will be in
    #   (3) new_units_func (function): Function dictating how a value is converted
    #         to the new_units.
    #   (4) new_var_name (str,default arg): The new name of the variable
    # EXAMPLE:
    #   modify_var('R','LOG_R_E',lambda x: math.log(x),new_var_name='LOG_R')
    def modify_var(self,var_name,new_units,new_units_func,new_var_name=None):
        
        # Modify each dataframe in _clean_data dict
        for inst in self.instruments:
            #df = (self._clean_data[inst])
            #df[var_name] = df[var_name].apply(new_units_func)
            #(self._clean_data[inst])[var_name] = df[var_name]
            (self._clean_data[inst])[var_name] = \
                (self._clean_data[inst])[var_name].apply(new_units_func)
            
        # Modify class dictionaries to reflect new units and new_var_name (if given)
        self.units_dict[var_name] = new_units
        if new_var_name is not None: self.var_names[var_name] = new_var_name
    
    
    
    
    
    
    
    
    
    
    
    # Shortcut function for changing time units as specified by user.
    # Supported units are as follows (where the first word is what needs to
    # be provided as arg for new_time_units):
    #   (1) secs (seconds), (2) mins (minutes), (3) hours, (4) days, (5) weeks
    # INPUT:
    #   new_time_units - String that describes the units to convert to. See above
    #     for supported units.
    # NO RETURNS
    def convert_time_to(self,new_time_units):
        current_time_units = self.units_dict["time"]
        
        # Check that user isn't converting to same units - just return if so
        if (current_time_units == new_time_units):
            warn_mssg = "Warning! Specified time units " + new_time_units + \
                " already matches current time units. Instead of needlessly " + \
                "multiplying all time values by 1, let's just do nothing " + \
                "instead. Returning..."
            warnings.warn(warn_mssg)
            return

        # Otherwise, determine function to convert to new_time_units
        # and notify user of units change        
        print("Coverting time from "+current_time_units+" to "+new_time_units)
        new_time_func = SSI_Data_Manager._naive_time_calculator(
                                                    current_time_units,
                                                    new_time_units
                                                    )
        self.modify_var("time",new_time_units,new_time_func)
        
    
    
    
    
    
    
    
    
    # Shortcut func for making certain variables base-10 Log.
    # In general, function applied is Log( abs(x[i]) ) for all i; makes no
    # difference for vector magnitudes, but is required in order for it to work
    # with vector components.
    def make_log(self,vars_to_modify):
        print("Coverting variables to Log: ",end="")
        for elem in vars_to_modify:
            print(elem," ",end="")
            self.modify_var(elem,"LOG_"+self.units_dict[elem],
                        lambda x: math.log10(abs(x)),new_var_name="LOG_"+elem)
        print("")
    
    
    
    
    
    
    
    
    
    # Returns a dict detailing the variable / column names in the returned
    # dataframes
    def get_names_dict(self):
        
        # Build the dict such that dict["X"] --> "X (R_E)"
        returned_dict = {}
        for item in self.var_names.keys():
            returned_dict[item] = self._get_formal_var_name(item)
        
        return returned_dict
            
            
            
            
            
            
            
            
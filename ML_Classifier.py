#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applies KMeans ML method to datasets to look for clusters of data.


SOM Example taken from the following links:
    
(1): https://rubikscode.net/2018/09/24/credit-card-fraud-detection-using-self-organizing-maps-and-python/

(2): https://rubikscode.net/2018/08/27/implementing-self-organizing-maps-with-python-and-tensorflow/

(3): https://www.rpubs.com/loveb/som

(4): https://github.com/JustGlowing/minisom/blob/master/examples/BasicUsage.ipynb

(5): https://github.com/JustGlowing/minisom/blob/master/minisom.py
    



IDEAS TO IMPLEMENT:
  1.)  Originally, I was going to create an SOM and feed the nodes to KMeans, like
    the OpenGGCM ML paper, but I don't think that will help here. The goal of
    SOM in that respect is essentially dimensionality reduction, but if KMeans
    can run on the raw data in a reasonable amount of time, then there's no
    need for reduction? In case I change my mind, it looks like it can be
    easily done by accessing the MiniSom class variable _weights (seen in the
    constructor of the code at link #5).
    
  2.) Using PCA. Again, I don't need dimensionality reduction here, but some
    visualization of how much variance certain variables contribute would be
    beneficial.
    
  3.) Could make a function that allows saving clustered date to csvs





Created on Wed Jan 20 14:19:10 2021

@author: jamesedmond
"""


# ========== IMPORTS ==========


# Graphical imports
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import cv2

# General non-graphical imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math, os, shutil, warnings, time
from scipy.optimize import minimize_scalar
from scipy.spatial import KDTree

# Machine Learning imports
from minisom import MiniSom
from sklearn.cluster import KMeans


# ========== CLASS ==========


class ML_Classifier:
    
    
    # ========== STATIC VARIABLES ==========
    
    # Graphical Params
    DISCRETE_LIMIT = 10
    ROW_TABLE_LIMIT = 8
    NUM_HIST_BINS = 300
    MULTIPLOT_SIZE = (16,8)
    DEFAULT_3D_VIEWING_ANGLES = [30, 270+30]
    
    # SOM Params
    DEFAULT_SOM_LEARNING_RATE = 0.5
    # No static default SOM sigma, but a dynamic default one is used;
    # see prepare_SOM function for details
    DEFAULT_SOM_MAX_NUM_ITERS = 500
    
    # KMeans Params
    DEFAULT_MAX_NUM_CLUSTERS = 5
    DEFAULT_KMEANS_MAX_NUM_ITERS = 300
    CLUSTER_FOLDER = "#_clusters"
    DEFAULT_MARKERS = ["o", "v", "^", "<", ">",
                       "s", "*", "X", "d", "p"]
    DEFAULT_COLORS = ["C0", "C1", "C2", "C3", "C4",
                      "C5", "C6", "C7", "C8", "C9"]
    
    # Phys params needed for consistent bow shock plotting
    DEFAULT_BZ = 0.151     #0.07688    # nT
    DEFAULT_NUM_DENS = 12.982       #5.622    # #/cc
    DEFAULT_SPEED = 407.2       #436.6    # km/s
    
    
    # ========== CONSTRUCTOR ==========
    
    
    def __init__(self,input_dataframe,output_loc):
        
        self.output_folder = output_loc
        self.orig_data = input_dataframe
        self._plot_original_data()
        self.norm_data = self._numericize_and_normalize_data()
        self.som = None
        self.kmeans = None
        self.num_clusters = None
        self.indices_by_cluster = None    # list[i] = n means data point at index i is in nth cluster
        self.orig_data_by_cluster = None
        self.norm_data_by_cluster = None
        self.kmeans_ignored_cols = None
        self.reduced_data_by_cluster = None
        self.downsample_res = None
        
    
    
    # ========== FUNCTIONS ==========
    
    
    
    
    
    # --- Assisting Functions (not to be ---
    # --- invoked by user) -----------------
    
    
    
    
    
    
    
    # The number of plots per row in an overall figure changes depending upon the
    # total number of plots
    # INPUT:
    #   (1) num_plots - Int, the total number of plots
    # RETURNS:
    #   (1) plots_per_row - Int, the number of plots per row for an overall figure
    def _get_plots_per_row(num_plots):
        plots_per_row = num_plots
        if ( (num_plots == 3) or (num_plots == 4) ): plots_per_row = 2
        elif ( (num_plots > 4) and (num_plots <= 9) ): plots_per_row = 3
        elif (num_plots > 9): plots_per_row = 4
        return plots_per_row
    
    
    
    
    
    
    
    
    
    
    # Takes an axes object and plots the average and standard deviation of data_arr
    # on top of it; differences in color and alpha can be specified
    def _plot_avg_and_stddev(axes_obj,data_arr,branch_point=0.7,alpha=1.0,color="red"):
        style = "--"
        # There's a strange error using np.average() for some data sets (NOT ALL!)
        # even when the np.std() function works... So I'm cheating by using more
        # fundamental array tools to get the average.
        avg = sum(data_arr)/len(data_arr) #np.average( data_arr )
        std_dev = np.std( data_arr )
        std_dev_left = avg - std_dev
        std_dev_right = avg + std_dev
        # Note for the axhline and axvlines below: the main arg (x for vline and
        # y for hline) are in DATA coords and the limiters (ymin,ymax for vline and
        # xmin,xmax for hline) are in AXES coords (i.e. between 0 and 1).
        # Vertical line for average
        axes_obj.axvline(x=avg,color=color,linestyle=style)
        # Vertical line for avg - std_dev
        axes_obj.axvline(x=avg - std_dev,ymin=0,ymax=branch_point,color=color,\
                        linestyle=style)
        # Vertical line for avg + std_dev
        axes_obj.axvline(x=avg + std_dev,ymin=0,ymax=branch_point,color=color,\
                        linestyle=style)
        # ... btw, it's too much of a struggle to provide axh/vline the data coords
        # sometimes, so I just used a blended transform below (such that x is in data
        # coords and y is in axes coords). Read more about it under "Blended Transforms"
        # at https://matplotlib.org/3.3.3/tutorials/advanced/transforms_tutorial.html
        xdata_yaxes = transforms.blended_transform_factory(axes_obj.transData, axes_obj.transAxes)
        # Horizontal line connecting avg - std_dev --> avg
        axes_obj.plot([std_dev_left,avg],[branch_point,branch_point],color=color,\
                      linestyle=style,transform=xdata_yaxes)
        # Horizontal line connecting avg --> avg + std_dev
        axes_obj.plot([avg,std_dev_right],[branch_point,branch_point],color=color,\
                      linestyle=style,transform=xdata_yaxes)
            
            
            
            
            
            
            
            
            
    # Computes the histogram bins for numpy_arr according to the max and min values
    # of that array; the number of bins desired is assumed to be num_hist_bins
    # (a global scope variable) but can be specified otherwise when calling.
    # NOTE: The returned array will have length N+1 where N = bins; this is because
    # matplotlib's hist function wants the BIN EDGES as an argument.
    def _compute_hist_bins(self,numpy_arr,bins=None):
        if bins is None:
            bins = self.NUM_HIST_BINS
        min_val = min( numpy_arr )
        max_val = max( numpy_arr )
        # The hist bins need n+1 bin edges, so we increment max according to the
        # resolution specified by bins and max_val and min_val.
        max_val_for_hist = (max_val - min_val) / bins + max_val
        return np.linspace(min_val,max_val_for_hist,num=bins)
    
    
    
    
    
    
    
    
    
    
    
    
    # Returns a list of the unused axes for a plot if given; if every subplot is
    # used then return an empty list.
    # INPUT
    #   (1) last_coord - The last axis used in a series of subplots
    #   (2) plot_dim - 2-elem list where first elem is num rows of subplots and
    #         second is num columns of subplots
    # E.G.
    #   If creating a 3x3 series of subplots and the last coord is [1,0] (such that
    #   there are seven plots used) then the list returned will consist only of the
    #   last two unused coords (being [ [1,1],[1,2] ] ).
    def _get_unused_axes_coords(last_coord, plot_dim):
        [num_rows,num_cols] = plot_dim
        [x_slices,y_slices] = np.mgrid[0:num_rows:1,0:num_cols:1]
        all_axes = np.vstack(( x_slices.flatten(), y_slices.flatten() )).T
        all_axes_list = [list(elem) for elem in all_axes]
        index = all_axes_list.index(last_coord) if last_coord in all_axes_list else None
        if (index == None): return []
        else: return all_axes_list[all_axes_list.index(last_coord):]
    
    
    
    
    
    
    
    
    
    
    
    # Plots all the variables in self.orig_data in multiple ways
    #   (1): A freqeuncy plot for each variable
    #   (2): A heatmap for the correlation between variables
    def _plot_original_data(self):
        print("---------- PLOTTING ORIGINAL DATA ----------")
        print("Creating Histograms and Tables...")
        
        
        # --- Histogram Plots ---
        
        # Calculate total number of subplots and create overall fig
        num_plots = len(list(self.orig_data))
        plots_per_row = ML_Classifier._get_plots_per_row(num_plots)
        subplots_x = math.ceil( num_plots/plots_per_row )
        subplots_y = plots_per_row
        fig, axes = plt.subplots(subplots_x,subplots_y,
                                 figsize=self.MULTIPLOT_SIZE)
        
        
        # Create each subplot
        for i in range(num_plots):
            [unique_vals,val_count] = np.unique(self.orig_data.iloc[:,i].values,
                                                return_counts=True)
            num_unique_points = len(unique_vals)            
                
            
            # Determine fig location
            [x_plot,y_plot] = ML_Classifier._get_axes_posit(i,plots_per_row)
            axes[x_plot,y_plot].set_title( list(self.orig_data)[i] )
            axes[x_plot,y_plot].grid(True)
            
            
            # Find a way to search for non-numerics across a whole col later -
            # make it simple for now
            col_sample = self.orig_data.iloc[0,i]   #.values
            is_continuous = False
            is_discrete = False
            if ML_Classifier._is_number(col_sample):
                is_continuous = num_unique_points > ML_Classifier.DISCRETE_LIMIT
                is_discrete = True if not is_continuous else False
            else:
                is_continuous = False
                is_discrete = True
            
            
            # - "Continuous" Data -
            if is_continuous:
                hist_bins = self._compute_hist_bins( self.orig_data.iloc[:,i].values )
                axes[x_plot,y_plot].hist(self.orig_data.iloc[:,i].values,\
                                     alpha=0.5,color="red",bins=hist_bins)
                ML_Classifier._plot_avg_and_stddev(axes[x_plot,y_plot],
                                    self.orig_data.iloc[:,i].values,
                                    alpha=1.0,color="green")
                    
                    
            # - Discrete Data -
            elif is_discrete:
                label_var = list(self.orig_data)[i]
                sn.countplot(x=label_var, ax=axes[x_plot,y_plot], data=self.orig_data)
    
                    
        # Remove unused subplots
        last_coord = [ int(num_plots/plots_per_row), num_plots % plots_per_row ]
        for elem in ML_Classifier._get_unused_axes_coords(last_coord, [subplots_x,subplots_y]):
            [x_coord,y_coord] = elem
            axes[x_coord,y_coord].axis("off")
    
            
        # Finish overall fig and save
        fig.suptitle("Histograms of Every Variable", fontsize=24)
        fig.tight_layout()
        fig.savefig(self.output_folder + "/" + "Original_Data_Histogram.png")
        plt.close()
        # --------------------------
        
        
        # --- Heatmap Plot ---
        print("Creating Heatmap of original data...")
        self._plot_heatmap(self.orig_data,
                title_str="Correlation Heatmap Between All Numerical Variables",
                fig_name="Original_Data_Heatmap.png")
        # --------------------
        
        
        print("---------------------------------------")
        
        
        
        
        
        
        
        
        
        
        
    # Takes in the original dataframe and numericizes the non-numerical variables.
    # The result is normalized and returned as a dataframe (with the same original labels).
    def _numericize_and_normalize_data(self):
        print("---------- PREPARING DATA ----------")
    
    
        # First transform nonumerical values into ints and create boxplot
        # of numericized data
        data_all_numerical = ML_Classifier\
            ._transform_nonnumeric_vars_into_ints(self.orig_data)
        print("Creating box-and-whisker plots of all variables (numericized)...")
        self._plot_dataframe_boxplot(data_all_numerical,
                title_str="Box-and-Whisker Plots of Numericized and NON-normalized Data",
                fig_name="Numericized_and_Nonnormalized_data.png")
            
        
        # Then normalize date and create boxplot of that
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_raw_nums = sc.fit_transform(data_all_numerical.values)
        scaled_dataframe = pd.DataFrame(scaled_raw_nums,
                                        columns=list(data_all_numerical))
        print("Creating box-and-whisker plots of all variables " + \
              "(numericized and normalized)...")
        self._plot_dataframe_boxplot(scaled_dataframe,
                title_str="Box-and-Whisker Plots of Numericized and Normalized Data",
                fig_name="Normalized_data.png")
        
        
        # Create Heatmap of Numericized and normalized data
        # (If there was no non-numerical data, then this plot will
        # be the exact same as the heatmap of the original data)
        print("Creating Heatmap of all variables (numericized and normalized)")
        self._plot_heatmap(scaled_dataframe,
                title_str="Correlation Heatmap Between All Numericized and Normalized Variables",
                fig_name="Numericized_Data_Heatmap.png")
        
        
        # Conclude func and return dataframe of numericized and normalized data
        print("---------------------------------------")
        return scaled_dataframe
    
    
    
    
    
    
    
    
    
    
    
    
    # Plots the various variables in the dataframe; optionally can specify a title in the
    # figure and the name of the figure (if it's to be saved).
    def _plot_dataframe_boxplot(self,dataframe_in,title_str="",fig_name=""):
        plt.figure(figsize=(9, 9))
        sn_ref = sn.boxplot(data=dataframe_in,orient="v")
        if (title_str != ""):  sn_ref.set_title(title_str)
        if (fig_name != ""):
            plt.savefig(self.output_folder + "/" + fig_name)
        else:
            plt.show()
        plt.close()
        
        
        
        
        
        
        
        
        
        
        
    # Creates a heatmap of all variables in dataframe_in (if there are
    # non-numerics, those variables will be ignored)
    def _plot_heatmap(self,dataframe_in,title_str="",fig_name=""):
        # This is done to establish a figure size
        _ , _ = plt.subplots(figsize=self.MULTIPLOT_SIZE)
        # Now build the heatmap
        corrMatt = dataframe_in.corr()
        mask = np.array(corrMatt)
        mask[np.tril_indices_from(mask)] = False
        sn_ref = sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
        # Apply title and fig names if relevant
        if (title_str != ""):  sn_ref.set_title(title_str)
        if (fig_name != ""):
            plt.savefig(self.output_folder + "/" + fig_name)
        else:
            plt.show()
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    # Converts discrete nonnumeric variables into integer equivalents
    # e.g. ['a','b','f','q2'] --> [0,1,2,3]
    # data_in is standard numpy array
    def _transform_nonnumeric_vars_into_ints( pd_dataframe ):
        print("Transforming discrete non-numerical data into integers...")
        
        
        # Extract numerics
        data_in = pd_dataframe.values
        
        
        # Determine which columns require changing
        indices_to_change = []
        for col_index in range( data_in.shape[1] ):
            val = data_in[0,col_index]
            # if num, skip
            if ML_Classifier._is_number(val): continue
            # otherwise, it's a discrete nonnumeric var
            indices_to_change.append(col_index)
            
            
        # If no indices to change, just return original dataframe
        if (len(indices_to_change) == 0):
            print("No non-numerical data found! Returning original dataframe...")
            return pd_dataframe
            
            
        # knowing which indices to look out, count the discrete vars to see how many
        # unique values there are
        unique_vals = []
        for col_index in indices_to_change:
            unique_vals_in_one_col = []
            for row_index in range( data_in.shape[0] ):
                val = data_in[row_index,col_index]
                if (val not in unique_vals_in_one_col): unique_vals_in_one_col.append( val )
            unique_vals.append( unique_vals_in_one_col )
            
            
        # So the indices of indices_to_change agree with the values in unique_vals
        # i.e. if indices_to_change = [1,5,7] and unique_vals = [list1,list2,list3],
        # then list1 contains all the unique vals for the data in column 1, list2 for
        # column 5, and list 3 for column 7.
        # Knowing this, change the chars in data_in found in unique_vals into their
        # integer equivalent i.e ['a','b','c'] --> [0,1,2]
        for meta_index in range(len(indices_to_change)):
            col_index = indices_to_change[meta_index]
            unique_vals_in_col = unique_vals[meta_index]
            for row_index in range( data_in.shape[0] ):
                val = data_in[row_index,col_index]
                data_in[row_index,col_index] = unique_vals_in_col.index( val )
        
        
        # Recreate pandas dataframe
        new_dataframe = pd.DataFrame( data_in, columns = list(pd_dataframe) )
        return new_dataframe
    
    
    
    
    
    
    
    
    
    
    # Checks to see if input_val is a number (really, a int or a float); if neither,
    # then not a number
    def _is_number(input_val):
        try:
            float(input_val)
            return True
        except ValueError:
            return False
        
        
        
        
        
        
        
        
    # Given an array showing which cluster a data point belongs to and the numpy
    # array of data points, return a list with the data organized by cluster
    # INPUT:
    #   (1) cluster_indarr - 1D list / array with length equal to length of data_arr.
    #         If n clusters were computed by KMeans, then the values of this array
    #         will range from 0 to n-1, denoting which cluster the data point in
    #         data_arr at that same index belongs to.
    #   (2) dataframe_in - Dataframe for the data supplied to KMeans; need not be
    #         normalized as it's just used for relative indexing purposes.
    # RETURNS:
    #  (1) data_by_cluster - A list of 1D numpy arrays where each array corresponds to the
    #        data for the cluster. E.g. data_by_cluster[0] possesses the data for
    #        all data in the 0th cluster.
    def _organize_data_by_cluster(cluster_indarr,dataframe_in):
        
        
        # Confirm each row has a cluster affiliation
        if not (len(cluster_indarr) == dataframe_in.shape[0]):
            print("Error! Sizes for inputs in organize_data_by_cluster do not agree!")
            return []
        
        
        # Count num clusters and prepare lists
        [cluster_types,cluster_sizes] = np.unique(cluster_indarr,return_counts=True)
        data_by_cluster_arrs = [[] for i in range(len(cluster_types))]
        data_by_cluster_dataframes = [[] for i in range(len(cluster_types))]
        
        
        # First save all rows in their right list based on cluster affiliation
        for i in range(dataframe_in.shape[0]):
            data_by_cluster_arrs[cluster_indarr[i]].append( dataframe_in.iloc[i,:].values )
            
            
        # Then convert those arrays into dataframes
        for i in range(len(cluster_types)):
            data_by_cluster_dataframes[i] = pd.DataFrame\
                (data_by_cluster_arrs[i],columns=list(dataframe_in))
        return data_by_cluster_dataframes
    
    
    
    
    
    
    
    
    
    
    
    def _get_axes_posit(nth_plot,plots_per_row):
        col_num = nth_plot % plots_per_row
        row_num = int(nth_plot/plots_per_row)
        return [row_num,col_num]
    
    
    
    
    
    
    
    
    
    
    
    # return elems of list if index is in desired_indices; can optionally
    # return a 2-elem list where first element just described and second
    # element is all elements of list whose index is NOT in desired_indices
    def _get_list_elems_from_indices(the_list,desired_indices,return_ignored=False):
        
        
        # Confirm desired_indices are nums; if strings, then they are already
        # the desired elements
        all_strs = True
        for i in range(len(desired_indices)):
            all_strs = not ML_Classifier._is_number(desired_indices[i]) and all_strs
        if all_strs: return desired_indices
            
        
        # If all nums, then get each elem in the_list with index
        # in desired_indices
        wanted_elems = []
        for i in range(len(desired_indices)):
            wanted_elems.append( the_list[desired_indices[i]] )
        if not return_ignored: return wanted_elems
        
        
        # If all nums and ignored elements are wanted too, return them
        unwanted_elems = []
        for elem in the_list:
            if elem not in wanted_elems: unwanted_elems.append(elem)
        return [wanted_elems,unwanted_elems]
    
    
    
    
    
    
    
    
    
    # Like get_list_elems_from_indices but inverted (i.e. return indices of
    # elems in desired_elems from the_list and can optionally return indices
    # of elems NOT in desired_elems that ARE in the_list)
    def _get_indices_of_list_elems(the_list,desired_elems,return_ignored=False):
        
        
        # Confirm desired elems are strings; if nums, then they are already
        # the desired indices
        all_nums = True
        for i in range(len(desired_elems)):
            all_nums = ML_Classifier._is_number(desired_elems[i]) and all_nums
        if all_nums: return desired_elems
        
        
        # If not all nums, then is all strings; determine indices of elems
        wanted_indices = []
        for i in range(len(desired_elems)):
            wanted_indices.append( the_list.index(desired_elems[i]) )
        if not return_ignored: return wanted_indices
        
        
        # If not all nums and the ignored elements are wanted too, return them
        unwanted_indices = []
        for i in range(len(the_list)):
            if i not in wanted_indices: unwanted_indices.append(i)
        return [wanted_indices,unwanted_indices]
    
    
    
    
    
    
    
    
    
    
    
    # unwanted cols can be all ints (indices) or col labels (strings)
    def _filter_cols_from_dataframe(dataframe_in,unwanted_cols):
        
        
        # If unwanted_col is empty list, just return given data
        if len(unwanted_cols) == 0: return dataframe_in
        
        
        # Get indices of unwanted_cols
        unwanted_col_indices = ML_Classifier\
            ._get_indices_of_list_elems(list(dataframe_in), unwanted_cols)
         
            
        # If repeated elems in unwanted_cols, return null and notify user
        if ( len(set(unwanted_col_indices)) != len(unwanted_col_indices) ):
            print("Error! Repeated indices / column labels give to ",
                  "filter_cols_from_dataframe. Quitting...")
            return None
            
            
        # Remove cols based on either index or col label
        return dataframe_in.drop(unwanted_cols,axis=1)
    
    
    
    
    
    
    
    
    
    # get nth element from collection where the coll_type is the type
    # collection being used (defaulted to list, but could be other types)
    # Indexing starts from 0, so if you want the 5th item (starting from 1),
    # then your input for n is n = 4.
    def _get_nth_item_from_arb_collection(n,collection,coll_type=list):
        flattened_coll = collection
        continue_unpacking = False
        
        
        # Check to see if contained type is coll_type (and if so,
        # we need to unpack further)
        for cnt,item in enumerate(flattened_coll):
            continue_unpacking = isinstance(item,coll_type)
            break
        
        
        # Keep unpacking collection until type NOT of coll_type is found
        while continue_unpacking:
            flattened_coll = \
                [ item for subcoll in flattened_coll for item in subcoll ]
            for cnt,item in enumerate(flattened_coll):
                continue_unpacking = isinstance(item,coll_type)
                break
            
        
        # With collection fully flattened, retrieve nth element
        for cnt,item in enumerate(flattened_coll):
            if (cnt == n): return item
            
            
            
            
            
    
    
    
    # Try drawing a little earth?    
    def _plot_earth(the_axis):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        the_axis.plot_wireframe(x, y, z, color="black")
    
    
    
    
            
            
            
            
            
            
    # Plots a magnetopause contour (according to 7.4.2 of Space Physics by
    # Russel, Luhmann, and Strangeway)
    # If dims is None / "2d", then plots a magpause contour
    # If dims is "3d", then plots 4 magpause contours (each of which are rotated)
    def _plot_magpause(the_axis,dims=None,xmin=-10):
        
        # Assign dims if default arg not used
        if dims is None: dims = "2d"
        
        # Physics params
        avg_bz = ML_Classifier.DEFAULT_BZ   #  nT
        avg_num_dens = ML_Classifier.DEFAULT_NUM_DENS * 10**6    #  #/m^3
        mass = 1.67 * 10**-27     #  kg
        avg_speed = ML_Classifier.DEFAULT_SPEED    #  km/s
        avg_dyn_press = avg_num_dens * mass * (avg_speed * 1000)**2 * 10**9    # nPascal
        alpha = (0.58 - 0.007*avg_bz) * (1 + 0.024 * math.log(avg_dyn_press))
        r_0 = (10.22 + 1.29 * math.tanh( 0.184 * (avg_bz + 8.14) ) )*avg_dyn_press**(-1/6)
        
        # Quick-define magpause contour for root-finding
        def _magpause_r(ang_rad):
            return (2**alpha) * r_0 * (1 + np.cos(ang_rad)) ** (-alpha)
        
        # root find to determine what theta is required to get out to xmin
        root_find_me = lambda angle: abs(_magpause_r(angle)*np.cos(angle) - xmin)
        root_find_result = \
            minimize_scalar(root_find_me,bounds=(0, math.radians(180)), method='bounded')
        
        # Set plotting params once max theta is known
        num_points = 1000
        min_theta, max_theta = -root_find_result.x, root_find_result.x
        theta_data = np.linspace(min_theta,max_theta,num=num_points)
        
        # compute bow shock r based on params and theta and
        # then compute the y values from those
        r_data = (2**alpha) * r_0 * (1 + np.cos(theta_data)) ** (-alpha)
        x_data = r_data * np.cos(theta_data)
        y_data = r_data * np.sin(theta_data)
            
        
        # ----- Plots magpause -----
        
        # In 2D (easy)
        if dims == "2d":
            
            the_axis.plot(x_data,y_data,linestyle="dashdot",color="black",
                          linewidth=1.0)
            #the_axis.plot(np.arange(20)-10,np.arange(20)-10)
            
        # In 3D (harder, requires rotations)
        if dims == "3d":
            
            # Generate contour for each of these angles (rotated about x axis)
            rotations = [0,math.radians(45),math.radians(90),math.radians(135)]
            linestyles = ["solid", "dashed", "dashdot", "dashed"]
            #supported linestyles are...
            # -', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            
            # Quick-define a rotation matrix for rotating about x axis
            def _trans_mat(ang_rad):
                yz_rotation = [ [1,0,0] , [0, np.cos(ang_rad),-np.sin(ang_rad)],
                                [0,np.sin(ang_rad),np.cos(ang_rad)] ]
                return np.array(yz_rotation)
            
            # Turn init x-y data into x-y-z array for mat mult
            init_data = np.vstack((x_data,y_data))
            init_data = np.vstack(( init_data,np.zeros(len(theta_data)) ))
            
            # Generate rotated data and plot it
            for i in range(len(rotations)):
                angle = rotations[i]
                new_data = np.matmul(_trans_mat(angle),init_data)
                the_axis.plot(new_data[0,:],new_data[1,:],new_data[2,:],
                              linestyle=linestyles[i],color="black",linewidth=0.5)
        
    
    
    
    
    
    
    
    
    
    # Given a base angle pair (elev,azim) and extra_angle_pairs, return list
    # of angles. The extra_angle_pairs may contains "+" or "-" operators,
    # referencing how much of a shift from the base_angle_pair is desired.
    # When wanting to keep the same angle for one and not the other, the string
    # "same" should be used for the angle to be kept E.G. ("same","+90"), or
    # ("same",235)
    def _determine_angles(base_angle_pair,extra_angles):
        
        # Determine if extra_angles is just one pair or a LIST OF PAIRS
        # If extra_angles is list of pairs, then just need assignment
        extra_angle_pairs = extra_angles
        # If extra_angles is just a pair (i.e. list of two nums/strings), then
        # need to make covert to list of pairs
        if not isinstance(extra_angles[0],list): extra_angle_pairs = [extra_angles]
        
        # Now create a numerical pair of angles for each pair given in
        # extra_angle_pairs
        angles = [base_angle_pair]
        for pair in extra_angle_pairs:
            # Initially assign pair to new_pair (all that's needed if its just nums)
            new_pair = pair
            for i in range(len(pair)):
                # If elements of pair are strings, then compute them in function
                if isinstance(pair[i],str):
                    if (pair[i].lower() == "same"): new_pair[i] = base_angle_pair[i]
                    else:
                        new_pair[i] = ML_Classifier._compute_angles_from_str\
                                    (base_angle_pair[i],pair[i])
            # Append pair of pure nums
            angles.append( new_pair )
        
        return angles
            
            
            
            
            
            
            
            
            
    # Assists _determine_angles by applying operator found in str of angle_str
    # E.G. func(45,"+90") --> 135
    def _compute_angles_from_str(base_angle,angle_str):
        supported_ops = ["+","-"]
        
        if "+" in angle_str: return base_angle + int(angle_str[angle_str.find("+")+1:])
        if "-" in angle_str: return base_angle - int(angle_str[angle_str.find("-")+1:])
        
        # Should not get to this line!
        raise ValueError("Error! Unsupported operator given in angle_string. " + \
                         "Was given " + angle_str + "; supported operators are :" + \
                         ''.join([elem for elem in supported_ops]))
            
            
    
    
    
    
    
    
    
    
    
    # actually makes the movie
    # PRIOR TO CALLING THIS, the_axis THAT WAS CREATED TO PLOT DATA MUST
    # NOT BE CLOSED! OTHERWISE THIS FUNCTION WILL FAIL...
    # Closing the_axis is beyond the scope of this function.
    def _make_movie_from_axis_obj(self,temp_folder,the_axis,save_folder=None,
                                  framerate=10,rotate_horizontal=False,
                                  rotate_vertical=False,plot_per_degrees=5,
                                  movie_name=None,style=None):
        
        
        # Check that one of rotate_horizontal or rotate_vertical was set
        if not rotate_horizontal and not rotate_vertical:
            raise ValueError("Error! In function _make_movie_from_axis_obj, " + \
                             "at least one of the default args rotate_horizontal" + \
                             " or rotate_vertical has to be set as True! " + \
                             "Was given [rotate_horizonal,rotate_vertical] as: ",
                             [rotate_horizontal,rotate_vertical])
        
        # Determine default args
        if movie_name is None: movie_name = "MyMovie"
        if style is None: style = "scatter_3D"
        
        
        # If save folder not set, just set it to containing folder of temp_folder
        if save_folder is None: save_folder = temp_folder[:temp_folder.rfind("/")]
        
                
        # Delete temp folder if it's left over from prior run - then make new,
        # empty one
        if os.path.isdir(temp_folder): shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)
         
        
        # Rotate horizontal, if set
        if rotate_horizontal:
            num_plots_horizontal = math.ceil(360 / plot_per_degrees)
            orig_elev = self.DEFAULT_3D_VIEWING_ANGLES[0]
            for i in range(num_plots_horizontal):
                new_azim = self.DEFAULT_3D_VIEWING_ANGLES[1] + i * plot_per_degrees
                the_axis.view_init(orig_elev,new_azim)
                the_filename = "Elev" + str(orig_elev) + "_Azim" + str(new_azim) + ".png"
                plt.savefig(temp_folder + "/" + the_filename)
                
        
        # Rotate vertical, if set
        if rotate_vertical:
            num_plots_vertical = math.ceil(360 / plot_per_degrees)
            orig_azim = self.DEFAULT_3D_VIEWING_ANGLES[1]
            for i in range(num_plots_vertical):
                new_elev = self.DEFAULT_3D_VIEWING_ANGLES[0] + i * plot_per_degrees
                the_axis.view_init(new_elev,orig_azim)
                the_filename = "Elev" + str(new_elev) + "_Azim" + str(orig_azim) + ".png"
                plt.savefig(temp_folder + "/" + the_filename)
                
        
        # Make movie from temp folder contents
        _, _, filenames = next(os.walk(temp_folder))
    
    
        # Grab all image metadata from images in temp_folder and compile into list
        img_array = []
        for filename in filenames:
            file = temp_folder + "/" + filename
            img = cv2.imread(file)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
    
    
        # Create movie from image data / metadata
        out_path_to_file = save_folder + "/" + movie_name + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # for mp4
        out = cv2.VideoWriter(out_path_to_file,fourcc, framerate, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        
        
        # Delete temp folder
        shutil.rmtree(temp_folder)
        
        
    
    
    
    
    
    
    
    
    
    # Downsampling, but without averaging. Points get picked at random and any
    # points within distance res of it are instead treated as neighbors to the
    # central point
    # index i of base-points is central point of ball ---> index i of lists_of_neighbours
    # is list of w-element vectors that are within distance res of said central point
    # assumes that there are 3 components of a vector that compose the magnitude;
    # these components are related to the magnitude by the usual Pythagorean rule
    # cols_for_downsample is a 3-element list of indices which describe which
    # columns are the x_vals, y_val, and z_vals (in that order)
    def _downsample(cols_for_downsample,n_by_w_array,res,num_times_to_print=None):
    
        # Determine if printing is specified
        if num_times_to_print is None: num_times_to_print = 0
        if num_times_to_print > 0: print("downsampling data: ",end="")
        printed_times = 1
        
        # ----- FIRST: CLASSIFY POINTS BY SAVING INDICES -----
        
        # determine indices based on var_indices
        x_col, y_col, z_col = cols_for_downsample
        
        # When comparing distances, it saves time to compare against this instead
        # of the square root
        res_squared = res**2
        
        # Create numpy array of indices that we need check all other distances against
        remaining_points_by_index = np.arange(n_by_w_array.shape[0])
        
        # Iterate over all points, and check all distances they have with every
        # other point
        base_indices, lists_of_neighbours_indices = [], []
        while len(remaining_points_by_index) > 0:
            if num_times_to_print > 0:
                frac_processed = 1 - len(remaining_points_by_index) / n_by_w_array.shape[0]
                if ( frac_processed > printed_times * (1/num_times_to_print) ):
                    print("{:.2f}".format(frac_processed*100)+"%.. ",end="")
                    printed_times = printed_times + 1
            
            # Randomly determine which point we'll consider as a base point
            rand_index = int(np.random.random_sample() * len(remaining_points_by_index))
            main_index = remaining_points_by_index[rand_index]
            base_indices.append( main_index )
            center_point = [ n_by_w_array[main_index,x_col],
                             n_by_w_array[main_index,y_col],
                             n_by_w_array[main_index,z_col] ]
            
            # Seems deceptively expensive..., but it's faster to remove the index
            # from this array now, then to check against this index everytime in
            # the for loop
            remaining_points_by_index = \
                np.setdiff1d( remaining_points_by_index , [main_index] )
            
            # Iterate through remaining_points_by_index, checking which are within
            # res of center_point
            neighbour_indices = []
            for index in remaining_points_by_index:
                
                # Check individial coords first - odds are, most points will have
                # single-dimensional coords whose distance already exceeds the res.
                # In which case, no need to square root or check the other coords!!!
                too_far = False
                if abs(center_point[0] - n_by_w_array[index,x_col]) > res: too_far = True
                elif abs(center_point[1] - n_by_w_array[index,y_col]) > res: too_far = True
                elif abs(center_point[2] - n_by_w_array[index,z_col]) > res: too_far = True
                
                # now check euclidean distance if other options exhausted
                else: 
                    too_far = (
                        (center_point[0] - n_by_w_array[index,x_col]) * \
                        (center_point[0] - n_by_w_array[index,x_col])
                        + \
                        (center_point[1] - n_by_w_array[index,y_col]) * \
                        (center_point[1] - n_by_w_array[index,y_col])
                        + \
                        (center_point[2] - n_by_w_array[index,z_col]) * \
                        (center_point[2] - n_by_w_array[index,z_col]) 
                            )    >    res_squared
                
                # If within res, then save index as a neighbour of center_point
                if not too_far: neighbour_indices.append( index )
            
            # Delete indices of points found within res of center_point
            remaining_points_by_index = \
                np.setdiff1d(remaining_points_by_index,neighbour_indices)
                
            # Save list of indices that were neighbours for current center_point
            lists_of_neighbours_indices.append( neighbour_indices )
            
        # ----- SECOND: TURN INDICES INTO POINTS -----
            
        # Convert base_points from list of rows to single numpy array
        base_points = np.array( [n_by_w_array[index,:] for index in base_indices] )
        
        # Convert lists_of_neighbours_indices to list of 2D numpy arrays    
        lists_of_neighbours = []
        for i in range(len(lists_of_neighbours_indices)):
            lists_of_neighbours.append(
                    np.array(
                [n_by_w_array[index,:] for index in lists_of_neighbours_indices[i]]
                            )
                                        )
            
        return base_points, lists_of_neighbours
    
    
    
    
    
    
    
    
    
    
    
    # Determines the number of "contiguous" shapes in the points in n_by_3_array.
    # Two points are "contiguous" if their euclidean distance is <= disconnected_distance.
    # RECOMMENED:
    # Before calling this, you should probably call _downsample. Then feed
    # base_points and TWICE the distance used for the _downsample resolution
    # to this function. The bulk of the numerical hard work here is done by
    # Scipy's KDTree's.
    def _compute_contiguous_shapes(n_by_3_array,disconnected_distance):
        
        # Use KDTree to determine all point pairs that have distances <= the
        # distance specified.
        kd_tree = KDTree(n_by_3_array)
        # pairs is a set of unique 2-element tuples of ints (where the ints
        # represent indices of rows in n_by_3_array)
        # ALSO: Lists are unhashable! The elements of pairs are 2-element TUPLES,
        # not lists!!!
        pairs = kd_tree.query_pairs(r=disconnected_distance)
    
            
        
        # so this is a weird combo of bfs AND dfs??? not entirely certain... but it works!
        # ints pairs are (i.j) where i is index of point we're leaving and j is
        # index of point we're arriving at... so i ---> j
        # This function is ONLY used here, so I just define it inside the function.
        def _build_shape_dfs_nonrecursive(int_pairs):
            
            # Get first connection by popping pair out and saving both "departing"
            # and "arriving" ints into encountered_ints
            first_pair = int_pairs.pop()
            encountered_ints = {first_pair[0],first_pair[1]}
            int_pairs.add( first_pair )
            
            # Begin search of int_pairs so long as connections
            # continue to be found (meaning a pair in int_pairs exists that has
            # an int in common with the set of departing_ints).
            remaining_pairs = {elem for elem in int_pairs}    # deep-copy
            int_pairs_in_shape = set()
            found_another_connection = True
            while found_another_connection:
                found_another_connection = False
                for index_pair in remaining_pairs:
                    
                    # Check to see if EITHER the departing int or arriving int
                    # is in the set of encountered_ints; if it is, then add both
                    # ints (as it's a set, the redundent one will just be ignored).
                    # And save the pair as well for removal later.
                    int_inside_pair = (index_pair[0] in encountered_ints) or \
                                        (index_pair[1] in encountered_ints)
                    if int_inside_pair:
                        found_another_connection = True
                        for the_int in index_pair: encountered_ints.add( the_int )
                        int_pairs_in_shape.add( index_pair )
                        
                # Remove from the set of pairs we're iterating over any pairs we've
                # already encountered.
                remaining_pairs = remaining_pairs - int_pairs_in_shape
            
            # Now create to return vals: first being the set of ints reached (which
            # will constitute the shape) and the second being the set of tuples of
            # their pairings.
            reached_ints = set()
            for elem in int_pairs_in_shape:
                for the_int in elem: reached_ints.add( the_int )
                
            return reached_ints, int_pairs_in_shape
    
    
    
        # Save all shapes as lists of lists of ints (where the ints represent their
        # indices in the rows of n_by_3_array)
        all_shapes = []
        lonesome_points = {index for index in range(n_by_3_array.shape[0])}
        remaining_pairs = {elem for elem in pairs}    # make deep-copy of pairs
        while len(remaining_pairs) > 0:
            reached_ints, tuples_used = _build_shape_dfs_nonrecursive(remaining_pairs)
            all_shapes.append( reached_ints )
            lonesome_points = lonesome_points - reached_ints
            remaining_pairs = remaining_pairs - tuples_used
            
        # When remaining_pairs has been exhaused, any points left in lonesome_points
        # are shapes in of themselves! (meaning, no adjacent point in the distance
        # specified to kd_tree.query_pairs exists)
        for point in lonesome_points:
            all_shapes.append( [point] )
            
        # Finally, construct list of (variable row size) x 3 arrays where the elements
        # of the list are the actual points instead of their index representation
        all_shapes_as_points = []
        for list_of_indices in all_shapes:         
            stackme = [ n_by_3_array[index,:] for index in list_of_indices ] 
            all_shapes_as_points.append( np.vstack(tuple( stackme )) )
            
        return all_shapes_as_points














    # Once both _downsample AND _compute_contiguous_shapes have been called,
    # you can optionally add back in the neighbours for shapes.
    # separate_shapes: list of N x 3 arrays
    # downsample_points: Single N x 3 array (is composed of all points in
    #                    separate_shapes, just without the list)
    # downsample_neighbours: List of N x 3 arrays. The array at index i contains
    #                       all the neighbour points to the main point located at
    #                       downsample_points[i,:].
    # small_shape_limit: Int. If a shape has this many points or less, it's
    #                   neighbours will be added back in to that shape.
    # NOTE: Running this with large small_shape_limit essentially undoes
    # the entire downsample / shape-build process and will be computationally
    # intense! But you will still know what points constitute what shapes.
    def _recombine_neighbours_into_shapes(separate_shapes,downsample_points,
                                    downsample_neighbours,small_shape_limit=None):
        
        if small_shape_limit is None: small_shape_limit=5
        
        # --- There's an issue in this function where sometimes points ---
        # --- are added redundantly! Trying to figure out what happens...
        
        # Deep-copy separate_shapes
        # ^^^ this is not a deep copy!!!!!!! Arrays still ahve same pointers!!
        denser_shapes = [arr for arr in separate_shapes]
        
        # Iterate through shapes and check points in shape ...
        for i in range(len(denser_shapes)):
            shape_points = denser_shapes[i]
            
            # If the number of points in the shape exceeds small_shape_limit, don't
            # add any neighbours! Just skip.
            if shape_points.shape[0] > small_shape_limit: continue
        
            # If the size is small enough, identify where the points of shape_points
            # are in downsample_points (i.e. find their indices for downsample_points)
            downsample_indices = []
            for row in shape_points:
                
                # Should only be one matching index! Printout warning if so!
                the_indices = np.where( (downsample_points==row).all(axis=1) )[0]
                if len(the_indices) > 1:
                    warnings.warn("Warning! Multiple points have been found to be " + \
                        "identical in function _recombine_neighbours_into_shapes! " + \
                        "The repeated data point:",row,"\nIndices of downsample_points " + \
                        "array:",the_indices)
                downsample_indices.append( int(the_indices[0]) )
            
            #print("the points",shape_points,"have the neighbours:")
            
            # Having found the indices, go to downsample_neighbours, grab the
            # neighbours located at index, and append them to shape_points
            vstackme = np.zeros(( 0 , downsample_points.shape[1] ))
            for index in downsample_indices:
                
                # if just empty arr for neighbours at index, then break out ...
                if downsample_neighbours[index].shape[0] == 0: break
            
                # ... otherwise, stack non-empty array
                #print(downsample_neighbours[index])
                vstackme = np.vstack(( vstackme , downsample_neighbours[index] ))
            
            #denser_shapes[i] = np.vstack(( shape_points , np.vstack(tuple(stackme)) ))
            #print("with all neighbours for said points stacked together, we have:",
            #      vstackme)
            denser_shapes[i] = np.vstack(( shape_points , vstackme ))
            
        return denser_shapes
    
    
    
    
    
    
    
    
    
    
    
    # implements ball-pivoting algorithm for a list of shapes
    # Returns list of dictionaries, where the dict at some index describes where
    # points in the shapes_list point to non-transitively. I.e.
    # For index A,....
    #     shapes_list[A] - [Variable rows] x 3 array of points, all in same shape
    #     returned_dict_list[A] - Dict with variable number of keys, and each key
    #                              has a list as a value. A key is a row index
    #                              for a data point in shapes_list[A] (call it
    #                              primary point), and the values are row indices
    #                              of points in shapes_list[A] that are within
    #                              max_distance of primary_point. These distances
    #                              are determined non-transitively, so if both
    #                               points X and Y are within max_distance of
    #                               each other and the distance FROM X TO Y is
    #                               found first, then the dict will show that,
    #                               BUT IT WILL NOT SHOW Y -> X!.
    def _ball_pivot(shapes_list,max_distance):
        
        # ----- FIRST: GENERATE DICT SHOWING WHICH POINTS GO WHERE -----
        
        # For speedier numerical calculation embedded in the inner-most for loop
        max_dist_squared = max_distance**2
        
        # Create list of dicts where each dict will have ...
        # 1.) The row index of a point as a key
        # 2.) The value for said key will be a list of row indices of all points
        #      that are within a distance max_distance of the key point.
        # 3.) These distances are NOT transitive within the dict! I.e., if point A
        #      is within max_distance of point B, then a dict will indicate that
        #      either A goes to B or B goes to A ... BUT NOT BOTH!!!
        list_of_line_dicts = []
        for shape_arr in shapes_list:
            
            # For each shape array, need to create dict described in above comment
            line_dict = {}
            
            # Start checking points by counting from LAST ROW of shape_arr,
            # and slowly work way up to 0th row.
            for highest_index in range(shape_arr.shape[0]-1,-1,-1):
                center_point = shape_arr[highest_index,:]
            
                # We don't want to double-count points, so just count from 0 to
                # highest_index!
                reached_points_by_index = []
                for i in range(highest_index):
                    if highest_index == i: continue    # don't check point against itself!
                    considered_point = shape_arr[i,:]
                    
                    # ---------- DISTANCE CHECKING ----------
                    
                    # Check individial coords first - odds are, most points will have
                    # single-dimensional coords whose distance already exceeds the res.
                    # In which case, no need to square root or check the other coords!!!
                    too_far = False
                    if abs(center_point[0] - considered_point[0]) > max_distance: too_far = True
                    elif abs(center_point[1] - considered_point[1]) > max_distance: too_far = True
                    elif abs(center_point[2] - considered_point[2]) > max_distance: too_far = True
                    
                    # If coords-checking fails, have to actually check euclidean distance
                    else: 
                        net_vector = center_point - considered_point
                        too_far = np.dot(net_vector,net_vector) > max_dist_squared
                    # ---------------------------------------
                    
                    # Add row index of point if within max_distance
                    if not too_far: reached_points_by_index.append(i)
                    
                # Now populate line_dict with row indices of points found close
                # to center_point -- if there were any such points.
                # If there were NO SUCH POINTS, then DO NOT create a dict entry!
                if len(reached_points_by_index) > 0:
                    line_dict[highest_index] = reached_points_by_index
                
            # After creating an entire dictionary showing which points are within
            # max_distance of each other (non-transitively!), add said dict to the
            # list_of_line_dicts
            list_of_line_dicts.append( line_dict )
            
        
        # ----- SECOND: BUILD ARRAY OF LINE SEGMENT BASED ON DICT -----
        
        list_of_line_coords = []
        for i in range(len(shapes_list)):
            shape_arr = shapes_list[i]
            line_dict = list_of_line_dicts[i]
            for start_index in list(line_dict.keys()):
                for end_index in line_dict[start_index]:
                    list_of_line_coords.append(
                                np.vstack(( 
                                shape_arr[start_index,:],
                                shape_arr[end_index,:] 
                                        ))
                                                )
            
        #return list_of_line_dicts
        return list_of_line_coords
    
    
    
    
    
    
    
    
    # Give list of shapes and dictionary showing which points go where, then
    # plot lines for said points for each shape
    # call compute_contiguous_shapes before calling this!
    def _ball_pivot_plotting(axis,shape_list,list_of_line_coords,linecolor="C0"):
        
        # Determine bounds of 3d plot
        buffer = 1.1
        all_xmins = [min(arr[:,0]) for arr in shape_list]
        all_ymins = [min(arr[:,1]) for arr in shape_list]
        all_zmins = [min(arr[:,2]) for arr in shape_list]
        xmin,ymin,zmin = min(all_xmins),min(all_ymins),min(all_zmins)
        all_xmaxs = [max(arr[:,0]) for arr in shape_list]
        all_ymaxs = [max(arr[:,1]) for arr in shape_list]
        all_zmaxs = [max(arr[:,2]) for arr in shape_list]
        xmax,ymax,zmax = max(all_xmaxs),max(all_ymaxs),max(all_zmaxs)
        axis.set_xlim(xmin*buffer,xmax*buffer)
        axis.set_ylim(ymin*buffer,ymax*buffer)
        axis.set_zlim(zmin*buffer,zmax*buffer)
        
        # Turn line coords into collection object for axis
        axis.add_collection(Line3DCollection(list_of_line_coords,
                                             colors=linecolor))
        
        # Okay.... so SOME POINT has to be plotted in order for this to work...
        # let's make it as small as possible and just put it at the origin!
        axis.scatter(0,0,zs=0,s=0.01,c=linecolor)
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    # Performs data reduction
    # Not intended for statistical simpliciation! Just for graphical simulation!
    def _perform_data_reduction(n_by_3_array,res=None,small_shape_limit=5,
                                return_shape_list=False):
        
        # Dynamically determine res if not given
        # Currently, res = 1/3 of avg of std_dev's
        # probably not a good idea......
        if res is None:
            std_dev_x = np.std(n_by_3_array[:,0])
            std_dev_y = np.std(n_by_3_array[:,1])
            std_dev_z = np.std(n_by_3_array[:,2])
            res = (1/3) * (std_dev_x + std_dev_y + std_dev_z) / 3
        
        # Decrease data size using _downsample
        #start_time = time.time()
        primary_points, list_of_neighbours = ML_Classifier._downsample(n_by_3_array,res)
        #print("time to downsample",n_by_3_array.shape[0],"points to",
        #      primary_points.shape[0],":",time.time()-start_time,"seconds.")
        
        # If only a downsample is desired, then just return primary_points
        if not return_shape_list: return [primary_points,list_of_neighbours]
        
        # Compute_contiguous_shapes
        #start_time = time.time()
        list_of_separate_shapes = ML_Classifier.\
            _compute_contiguous_shapes(primary_points,2*res)
        #print("time to compute",len(list_of_separate_shapes),"shapes from downsample:",
        #      time.time()-start_time,"seconds.")
            
        # If small_shape_limit was set to 0, then neighbour points are not to be
        # added to anything. In which case, return current list of shapes
        if small_shape_limit == 0: return list_of_separate_shapes
        
        # Recombine neighbours into outliers if specified
        #start_time = time.time()
        denser_shapes = ML_Classifier._recombine_neighbours_into_shapes\
            (list_of_separate_shapes,primary_points,list_of_neighbours,
             small_shape_limit=small_shape_limit)
        #print("Time to recombine neighbours into small shapes:",time.time()-start_time,
        #      "seconds")
            
        #print("After adding neighbours back in for \"small\" shapes, are all",
        #      "the points within unique?")
        #print("num rows in after making denser shapes:",sum([arr.shape[0] \
        #                                    for arr in denser_shapes]))
        #print("num unique rows in denser_shapes:",
        #      sum([np.unique(arr,axis=0).shape[0] for arr in denser_shapes]))
        
        # Return list of dense shape
        return denser_shapes
    
    
    
    
    
    
    
    # compute heatmap val from array based on scaling specification
    def _compute_heatmap_val_based_on_scaling(central_heatmap_val,
                                              distances,distant_heatmap_vals,
                                              scaling):
        
        # If no scaling wanted, just return the central value
        if scaling is None:
            
            return central_heatmap_val
        
        # Just your basic simple average (not weighted!)
        elif scaling == "average":
            
            sum_of_heatmap_vals = sum(distant_heatmap_vals) + central_heatmap_val
            return sum_of_heatmap_vals / (len(distant_heatmap_vals) + 1)
    
    
    
    
    
    
    
    
    
    
    
    # current method of "averaging" a variable across main and neighbour points
    # is to give greater consideration towards closer points by a linear factor
    # of distance
    def _build_heatmap_var_from_reduced_data(self,main_points,list_of_neighbours,
                                             heatmap_var,downsample_vars,
                                             scaling=None):
        
        # determine column index corresponding to heatmap_var
        heatmap_var_col_index = list(self.orig_data).index(heatmap_var)
        x_col = list(self.orig_data).index(downsample_vars[0])
        y_col = list(self.orig_data).index(downsample_vars[1])
        z_col = list(self.orig_data).index(downsample_vars[2])
        
        # compute some average of heatmap values to assign to base_point
        # (adding in that base point's original heatmap value)
        averaged_heatmap_vals = []
        for i in range(len(main_points)):
            
            # get base point and its neighbours
            base_point = main_points[i,:]
            neighbour_points = list_of_neighbours[i]
                
            # If no neighbour points exist, then no scaling to be done.
            if len(neighbour_points) == 0:
                averaged_heatmap_vals.append( base_point[heatmap_var_col_index] )
                continue
            
            # computes distances as 1d array
            # so for some fucking stupid goddamn reason, np.sqrt() doesn't want
            # to work on arrays here even though I've done it dozens of times
            # before; guess I'll do it manually, element-by-element, like
            # a fucking tool then...
            #x_distances = base_point[x_col] - neighbour_points[:,x_col]
            #y_distances = base_point[y_col] - neighbour_points[:,y_col]
            #z_distances = base_point[z_col] - neighbour_points[:,z_col]
            #sum_of_dist_squared = x_distances**2 + y_distances**2 + z_distances**2
            distances = np.zeros(len(neighbour_points))
            for q in range(len(distances)): 
                #distances[q] = np.sqrt( sum_of_dist_squared[q] )
                distances[q] = np.sqrt(
                        (base_point[x_col] - neighbour_points[q,x_col])**2 + \
                        (base_point[y_col] - neighbour_points[q,y_col])**2 + \
                        (base_point[z_col] - neighbour_points[q,z_col])**2
                                    )
            #print(sum_of_dist_squared,type(sum_of_dist_squared))
            #print( np.sqrt(sum_of_dist_squared[0]) )
            #distances = np.sqrt( sum_of_dist_squared )
            #distances = np.sqrt(
            #    (base_point[x_col] - neighbour_points[:,x_col])**2 + \
            #    (base_point[y_col] - neighbour_points[:,y_col])**2 + \
            #    (base_point[z_col] - neighbour_points[:,z_col])**2
            #    )
            
            computed_heatmap_val = ML_Classifier._compute_heatmap_val_based_on_scaling(
                                        base_point[heatmap_var_col_index],
                                        distances,
                                        neighbour_points[:,heatmap_var_col_index],
                                        scaling
                                                                        )
            
            averaged_heatmap_vals.append( computed_heatmap_val )
                
        return averaged_heatmap_vals
            
        
    
        
                
                
        
        
        
            
            
        
    
    
    
        
        
    
    
    
    
    
    
    
    # --- Primary Functions (to be ---
    # --- called by user) ------------
        
        
        
        
        
        
    
    
    
    # Creates a square of nodes where node_length is sqrt(num_nodes)
    # Default sigma
    def run_SOM(self,node_length,sigma=None,learning_rate=None,
                    num_iters=None,plot_inertia=False):
        print("---------- PREPARING AND RUNNING SOM ----------")
        
        
        # Get defaults params if none provided
        if learning_rate is None:
            learning_rate = self.DEFAULT_SOM_LEARNING_RATE
        if sigma is None:
            # linear prop such that sigma = 1.0 for node_len = 10
            sigma = (1/10) * node_length
        if num_iters is None:
            num_iters = self.DEFAULT_SOM_MAX_NUM_ITERS
            
            
        # --- Build SOM ---
        # Create (but not yet train!) the SOM
        print("Preparing SOM with",node_length,"x",node_length,
              "nodes, sigma = ",sigma,"learning rate = ",learning_rate,
              "and",num_iters,"iterations.")
        self.som = MiniSom(node_length, node_length, self.norm_data.values.shape[1],
                    sigma=sigma, learning_rate=learning_rate)
        # ------------------
        
        
        # --- Train SOM ---
        # If not inertia plotting, training is simple
        print("Training SOM...   ",end="")
        if not plot_inertia: self.som.train(self.norm_data.values,num_iters)#,verbose=True)
        else:
        
            
            # Run SOM and save weight sum (inertia) vs iteration
            #q_error = []
            weight_sums = []
            for i in range(num_iters):
                rand_i = np.random.randint(len(self.norm_data.values))
                self.som.update(self.norm_data.values[rand_i],
                           self.som.winner(self.norm_data.values[rand_i]), i, num_iters)
                weight_sums.append(
                    sum( [sum(metarow) for metarow in \
                              [sum(row**2) for row in self.som.get_weights()]  ] )
                                                )
                #q_error.append( som.quantization_error(scaled_raw_nums ))
    
    
            # Plot inertia
            #plt.plot(np.arange(num_iters), q_error, label='quantization error')
            plt.scatter(np.arange(num_iters), weight_sums)
            plt.title("SOM Inertia vs number of iterations")
            plt.ylabel("SOM Inertia")
            plt.xlabel("Number of iterations")
            plt.savefig(self.output_folder + "/" + "SOM_Inertia.png")
            plt.close()
        # ------------------
        
        
        # Save node map
        plt.figure(figsize=(9, 9))
        plt.pcolor(self.som.distance_map().T, cmap='bone')  # plotting the distance map as background
        plt.colorbar()
        plt.title(str(node_length) + " x " + str(node_length) + \
                  " Node map after " + str(num_iters) + " updates")
        plt.savefig(self.output_folder + "/" + "SOM_nodes.png")
        plt.close()
        print("Done!")
        print("---------------------------------------------")
        
        
        
        
        
        
        
        
        
        
    # Run KMeans algorithm and plots inertia (i.e. sum of squares within cluster)
    # Generally, KMeans has a tendency to fall into local minima for their inertias.
    # Fortunately, the scikitlearn implementation accounts for this and a call to
    # KM has a default parameter n_init (default value is 10) the specifies how
    # many times KMeans is ran (and by the end, the return the run with
    # minimal inertia).
    def plot_kmeans_inertia_vs_num_clusters(self,max_clusters=None,ignore_cols=[]):
        print("Plotting KMeans inertia vs number clusters...")
        if max_clusters is None:
            max_clusters = self.DEFAULT_MAX_NUM_CLUSTERS
            
            
        # Apply KMeans to data that does NOT use columns from ignore_cols
        data_for_kmeans = ML_Classifier._filter_cols_from_dataframe(self.norm_data,ignore_cols)
        
        
        # Collect total inertias for each cluster *and* each time!
        inertias = []    # the sum of squares within clusters
        for i in range(1,max_clusters+1):
            kmeans = KMeans(n_clusters=i, random_state=i).fit(data_for_kmeans.values)
            inertias.append( kmeans.inertia_ )
        
        
        # Plot the total inertia over each
        plt.figure(figsize=(9, 9))
        plt.plot(np.arange(max_clusters+1)[1:],inertias,marker="o",markersize=15)

    
        # Plotting the inertias
        plt.title("KMeans Inertia vs Cluster Size")
        plt.ylabel("KMeans Inertia")
        plt.xlabel("Number of KMeans clusters used")
        plt.savefig(self.output_folder + "/" + "KMeans_Inertia.png")
        plt.close()
        
        
        
        
        
        
        
        
        
        
        
    # Plots each cluster on some variables as specified in var_lists, so
    # 1 subplot per cluster.
    # var_lists is either a list of strings (e.g. ["X", "Y"] or ["X", "Y", "Z"])
    # or a LIST of list of strings ([ ["X","Y"] , ["X","Z"] ], and similar for
    # 3d variable plotting). These variable names must match what are in the
    # original provided pandas dataframe.
    # The extra angles are list-pairs you can provide so that additional 3d plots
    # can be provided at that orientation.
    def plot_separate_clusters_on_vars(self,var_lists,extra_angles=None):
        print("Plotting",len(self.orig_data_by_cluster),
              "SEPARATED clusters across variable pairs: ")
        
        
        # Partition var_lists into two lists: one of pairs (for 2D plots)
        # and one of triplets (for 3D plots)
        var_lists_2D = [elem for elem in var_lists if len(elem) == 2]
        var_lists_3D = [elem for elem in var_lists if len(elem) == 3]
        
        
        # ======= PLOTTING 2D =======
        
        if (len(var_lists_2D) > 0):
        
            # Fig params
            num_total_figs = len(var_lists_2D)
            num_plots_per_fig = len(self.orig_data_by_cluster)
            num_clusters = num_plots_per_fig
            plots_per_row = ML_Classifier._get_plots_per_row(num_plots_per_fig)
            subplots_x = math.ceil( num_plots_per_fig/plots_per_row )
            subplots_y = plots_per_row  
            last_axis_coord = [ int(num_plots_per_fig/plots_per_row),
                                num_plots_per_fig % plots_per_row ]
            
            
            # Create an overall fig for each variable pair
            print("    ",end="")
            for i in range(num_total_figs):
                pair = var_lists_2D[i]
                xlabel, ylabel = pair[0], pair[1]
                fig, axes = plt.subplots(subplots_x, subplots_y,
                                         figsize=self.MULTIPLOT_SIZE, dpi=120)
                print("[",xlabel,"x",ylabel,"] ",end="")
                
                
                # For consistency, get min/max vals for each var so that
                # similar plotting ranges can be enforced
                extra_x_space, extra_y_space = 0.1, 0.1
                min_x_val, min_y_val = np.inf, np.inf
                max_x_val, max_y_val = -1 * np.inf, -1 * np.inf
                for a in range(num_clusters):
                    possible_new_x_max = max( (self.orig_data_by_cluster[a])[pair[0]].values )
                    possible_new_x_min = min( (self.orig_data_by_cluster[a])[pair[0]].values )
                    possible_new_y_max = max( (self.orig_data_by_cluster[a])[pair[1]].values )
                    possible_new_y_min = min( (self.orig_data_by_cluster[a])[pair[1]].values )
                    max_x_val = possible_new_x_max if possible_new_x_max > max_x_val else max_x_val
                    min_x_val = possible_new_x_min if possible_new_x_min < min_x_val else min_x_val
                    max_y_val = possible_new_y_max if possible_new_y_max > max_y_val else max_y_val
                    min_y_val = possible_new_y_min if possible_new_y_min < min_y_val else min_y_val
                y_lower_bound = min_y_val - (max_y_val - min_y_val) * (extra_y_space/2)
                y_upper_bound = max_y_val + (max_y_val - min_y_val) * (extra_y_space/2) 
                x_lower_bound = min_x_val - (max_x_val - min_x_val) * (extra_x_space/2)
                x_upper_bound = max_x_val + (max_x_val - min_x_val) * (extra_x_space/2)               
                
                
                # --- Plot each cluster separately ---
                
                for a in range(num_plots_per_fig):
                
                    # get axis to be used for current plot
                    the_axis = ML_Classifier\
                        ._get_nth_item_from_arb_collection(a,axes,coll_type=np.ndarray)
                
                    # Create each subplot
                    x_data = (self.orig_data_by_cluster[a])[pair[0]].values
                    y_data = (self.orig_data_by_cluster[a])[pair[1]].values
                    the_axis.scatter(
                        x_data,y_data,s=60,
                        facecolors="None",edgecolors=self.DEFAULT_COLORS[a],
                        linewidth=2,marker=self.DEFAULT_MARKERS[a]
                        )
                    the_axis.set_xlabel( xlabel )
                    the_axis.set_ylabel( ylabel )
                    the_axis.set_title("Cluster "+str(a+1),fontweight="bold")
                    
                    # If plotting x,y or x,z, want magpause included in plot --
                    # which requires some manipulation of plot bounds
                    is_X_vs_Y = (xlabel == "X (R_E)") and (ylabel == "Y (R_E)")
                    is_X_vs_Z = (xlabel == "X (R_E)") and (ylabel == "Z (R_E)")
                    if not is_X_vs_Y or not is_X_vs_Z:
                        the_axis.set_xlim([x_lower_bound,x_upper_bound])
                        the_axis.set_ylim([y_lower_bound,y_upper_bound])
                    
                    # These are plotted but not visible most of the time!
                    # Fix this in the future for when the magpause is outside
                    # of the data's range...
                    if is_X_vs_Y or is_X_vs_Z:
                        ML_Classifier._plot_magpause(the_axis,xmin=1.1*x_lower_bound)
                # -------------------------------------
                    
                    
                # Delete unused subplots
                for elem in ML_Classifier._get_unused_axes_coords\
                            (last_axis_coord,[subplots_x,subplots_y]):
                    [x_coord,y_coord] = elem
                    axes[x_coord,y_coord].axis("off")
                
                
                # Create folder to place in if it doesn't already exist
                folder_loc = self.output_folder + "/" + \
                    ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                    "/" + "2D_Clusters"
                if not os.path.exists(folder_loc):
                    os.makedirs(folder_loc)
                    
                    
                # Adjust fig dims
                fig.tight_layout(pad=1.0)
                plt.subplots_adjust(top=0.9,bottom=0.1,hspace=0.4)
                
                # Make title
                title_str = "Distribution of " + str(num_clusters) + \
                    " clusters Across " + xlabel + " VS " + ylabel
                fig.suptitle(title_str,fontsize=24)
                
                # Get proper labels
                legal_xlabel = xlabel.replace(" ","_").replace("/","_per_")\
                    .replace("#","num")
                legal_ylabel = ylabel.replace(" ","_").replace("/","_per_")\
                    .replace("#","num")
                    
                # Save fig and close
                fig_name = str(num_clusters) + "_clusters_" + legal_xlabel + \
                    "_" + legal_ylabel + ".png"
                plt.savefig(folder_loc + "/" + fig_name) 
                plt.close()
        # ========== END PLOTTING 2D ==========
            
            
        # ========== PLOTTING 3D ==========
        
        if (len(var_lists_3D) > 0):
            
            # elev --> angle above x-y plane  ;  azim --> angle along azimuth
            [orig_elev, orig_azim] = self.DEFAULT_3D_VIEWING_ANGLES
            
            angles = [(orig_elev,orig_azim)]
            if extra_angles is not None:
                angles = ML_Classifier._determine_angles(angles[0],extra_angles)
            
            for angle_pair in angles:
        
                # Fig params
                num_total_figs = len(var_lists_3D)
                num_plots_per_fig = len(self.orig_data_by_cluster)
                num_clusters = num_plots_per_fig
                plots_per_row = ML_Classifier._get_plots_per_row(num_plots_per_fig)
                subplots_x = math.ceil( num_plots_per_fig/plots_per_row )
                subplots_y = plots_per_row  
                last_axis_coord = [ int(num_plots_per_fig/plots_per_row),
                                    num_plots_per_fig % plots_per_row ]
                
                
                # Create an overall fig for each variable pair
                print("    ",end="")
                for i in range(num_total_figs):
                    triplet = var_lists_3D[i]
                    xlabel, ylabel, zlabel = triplet[0], triplet[1], triplet[2]
                    fig = plt.figure(figsize=self.MULTIPLOT_SIZE, dpi=120)
                    #fig, axes = plt.subplots(subplots_x, subplots_y,
                    #                         figsize=self.MULTIPLOT_SIZE, dpi=120)
                    print("[",xlabel,"x",ylabel,"x",zlabel,"] ",end="")
                    
                    
                    # For consistency, get min/max vals for each var so that
                    # similar plotting ranges can be enforced
                    extra_x_space, extra_y_space, extra_z_space = 0.1, 0.1, 0.1
                    min_x_val, min_y_val, min_z_val = np.inf, np.inf, np.inf
                    max_x_val, max_y_val, max_z_val = -1 * np.inf, -1 * np.inf, -1 * np.inf
                    for a in range(num_clusters):
                        possible_new_x_max = max( (self.orig_data_by_cluster[a])[triplet[0]].values )
                        possible_new_x_min = min( (self.orig_data_by_cluster[a])[triplet[0]].values )
                        possible_new_y_max = max( (self.orig_data_by_cluster[a])[triplet[1]].values )
                        possible_new_y_min = min( (self.orig_data_by_cluster[a])[triplet[1]].values )
                        possible_new_z_max = max( (self.orig_data_by_cluster[a])[triplet[2]].values )
                        possible_new_z_min = min( (self.orig_data_by_cluster[a])[triplet[2]].values )
                        max_x_val = possible_new_x_max if possible_new_x_max > max_x_val else max_x_val
                        min_x_val = possible_new_x_min if possible_new_x_min < min_x_val else min_x_val
                        max_y_val = possible_new_y_max if possible_new_y_max > max_y_val else max_y_val
                        min_y_val = possible_new_y_min if possible_new_y_min < min_y_val else min_y_val
                        max_z_val = possible_new_z_max if possible_new_z_max > max_z_val else max_z_val
                        min_z_val = possible_new_z_min if possible_new_z_min < min_z_val else min_z_val
                    x_lower_bound = min_x_val - (max_x_val - min_x_val) * (extra_x_space/2)
                    x_upper_bound = max_x_val + (max_x_val - min_x_val) * (extra_x_space/2)
                    y_lower_bound = min_y_val - (max_y_val - min_y_val) * (extra_y_space/2)
                    y_upper_bound = max_y_val + (max_y_val - min_y_val) * (extra_y_space/2) 
                    z_lower_bound = min_z_val - (max_z_val - min_z_val) * (extra_z_space/2)
                    z_upper_bound = max_z_val + (max_z_val - min_z_val) * (extra_z_space/2)
                    
                    
                    # --- Plot each cluster separately ---
                    
                    for a in range(num_plots_per_fig):
                    
                        # get axis to be used for current plot
                        #the_axis = ML_Classifier\
                        #    ._get_nth_item_from_arb_collection(a,axes,coll_type=np.ndarray)
                        #the_axis = fig.add_subplot(subplots_y,subplots_x,a+1, projection='3d')
                        the_axis = fig.add_subplot(subplots_x,subplots_y,a+1, projection='3d')
                    
                        # Create each subplot
                        x_data = (self.orig_data_by_cluster[a])[triplet[0]].values
                        y_data = (self.orig_data_by_cluster[a])[triplet[1]].values
                        z_data = (self.orig_data_by_cluster[a])[triplet[2]].values
                        """the_axis.scatter(
                            x_data,y_data,z_data,s=60,
                            facecolors="None",edgecolors=self.DEFAULT_COLORS[a],
                            linewidth=2,marker=self.DEFAULT_MARKERS[a]
                            )"""
                        the_axis.scatter(
                            x_data,y_data,z_data, s=60, facecolor=(0,0,0,0),
                            edgecolors=self.DEFAULT_COLORS[a],linewidth=2,
                            marker=self.DEFAULT_MARKERS[a]
                            )
                        the_axis.set_xlabel( xlabel )
                        the_axis.set_ylabel( ylabel )
                        the_axis.set_zlabel( zlabel )
                        the_axis.set_xlim([x_lower_bound,x_upper_bound])
                        the_axis.set_ylim([y_lower_bound,y_upper_bound])
                        the_axis.set_zlim([z_lower_bound,z_upper_bound])
                        the_axis.view_init(angle_pair[0],    # elev first
                                           angle_pair[1])    # then azim
                        the_axis.set_title("Cluster "+str(a+1),fontweight="bold")
                        
                        # Plot magpause if x,y,z
                        using_xyz_posit = (xlabel == "X (R_E)") and \
                            (ylabel == "Y (R_E)") and (zlabel == "Z (R_E)")
                        if using_xyz_posit:
                            ML_Classifier._plot_magpause(the_axis,dims="3d")
                    # -------------------------------------
                        
                        
                    # Delete unused subplots
                    for elem in ML_Classifier._get_unused_axes_coords\
                                (last_axis_coord,[subplots_x,subplots_y]):
                        [x_coord,y_coord] = elem
                        axes[x_coord,y_coord].axis("off")
                    
                    
                    # Create folder to place in if it doesn't already exist
                    folder_loc = self.output_folder + "/" + \
                        ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                        "/" + "3D_Clusters"
                    if not os.path.exists(folder_loc):
                        os.makedirs(folder_loc)
                        
                        
                    # Adjust fig dims
                    fig.tight_layout(pad=1.0)
                    plt.subplots_adjust(top=0.9,bottom=0.1,hspace=0.4)
                    
                    # Make title
                    title_str = "Distribution of " + str(num_clusters) + \
                        " clusters Across " + xlabel + " VS " + ylabel + \
                        " VS " + zlabel
                    fig.suptitle(title_str,fontsize=24)
                    
                    # Get proper labels
                    legal_xlabel = xlabel.replace(" ","_").replace("/","_per_")\
                        .replace("#","num")
                    legal_ylabel = ylabel.replace(" ","_").replace("/","_per_")\
                        .replace("#","num")
                    legal_zlabel = zlabel.replace(" ","_").replace("/","_per_")\
                        .replace("#","num")
                        
                    # Put angle info into fig name and save
                    angle_info = "elev" + str(angle_pair[0]) + "_azim" + \
                        str(angle_pair[1])
                    fig_name = str(num_clusters) + "_clusters_" + legal_xlabel + \
                        "_" + legal_ylabel + "_" + legal_zlabel + "_" + \
                        angle_info + ".png"
                    plt.savefig(folder_loc + "/" + fig_name) 
                    plt.close()
        # ========== END PLOTTING 3D ==========
        
        print("")
        
    
    
    
    
    
    
    
            
    
    
    # reduces data by cluster (which means some downsampling will occur!)
    # vars_for_downsample indicate what columns will be used for downsampling
    # (e.g. if downsampling on position space ("X","Y","Z") with vector mag "R",
    # then vars_for_downsample=["X","Y","Z"] )
    # run_kmeans() should already be called!
    def reduce_data_by_cluster(self,vars_for_downsample,downsample_resolution):
        print("Downsampling data across",self.num_clusters,"clusters in",
              vars_for_downsample,"space: ",end="")
        
        # Save resolution as class variable
        self.downsample_res = downsample_resolution
        
        # Determine which columns (numerically, i.e. by index) compose the
        # components of the vector that we're going to reduce by
        x_col = list(self.orig_data).index( vars_for_downsample[0] )
        y_col = list(self.orig_data).index( vars_for_downsample[1] )
        z_col = list(self.orig_data).index( vars_for_downsample[2] )
        cols_for_comps = [ x_col, y_col, z_col ]
        
        # Now reduce data for each cluster
        reduced_data_by_cluster = []
        for i in range(self.num_clusters):
            print(str(i+1)+" ",end="")
        
            # data reduction into main and neighbour points
            main_pts, neighbour_pts = \
                ML_Classifier._downsample(
                                        cols_for_comps,
                                        self.orig_data_by_cluster[i].values,
                                        downsample_resolution
                                        )
            
            # save main and neighbour points into dict, and that into class variable
            reduced_data_dict = {}
            reduced_data_dict["main"] = main_pts
            reduced_data_dict["neighbour"] = neighbour_pts
            reduced_data_by_cluster.append( reduced_data_dict )
        
        self.reduced_data_by_cluster = reduced_data_by_cluster
            
            
        
        
        
        
    
    
    
    
    
    
    # Makes movie of rotated images for 3-variable pairs provided.
    # Can specify vertical or horiontal rotation (or both!)
    # Can specify if clusters should be plotted together or separate (or both!)
    # 3D plotting is done using trisurfaces
    # Set enable_data_reduction to resolution used to say points are distinguishable
    #    e.g. enable_data_reduction = 0.1
    # ========== IDEAS FOR FUTURE ==========
    # 1.) I should rewrite the _perform_data_reduction function in cython
    #     since it's so numerically heavy and doesn't require a lot of
    #     outside knowledge other than an N x 3 array and a resolution
    # 2.) Better structure heatmap_vars to be a dict; that way, know when
    #     to apply heatmap to certain plots based on key
    # ======================================
    def make_movie_from_rotated_plots(self,var_lists,rotate_horizontal=False,
                                      rotate_vertical=False,plot_per_degrees=None,
                                      framerate=None,all_clusters=False,
                                      separate_clusters=False,
                                      use_reduced_data=None,
                                      heatmap_var=None):
                                      #style=None):
    
        # Determine framerate and plots_per_degree
        if framerate is None: framerate = 10
        if plot_per_degrees is None: plot_per_degrees = 5
        
        if use_reduced_data is None: use_reduced_data = False
        
        # Set style to using ball-pivot alg is none set
        #if style is None: style = "scatter"
        style="scatter" #fix this later
        
        # Check that all lists of vars to plot constitute 3D plotting -- no point
        # in rotating a 2D object...
        for elem in var_lists:
            if len(elem) == 2:
                raise ValueError("Error! A list of variables to plot that was " + \
                                 "passed to make_movie_from_rotated_plots is not" + \
                                 "3D! Was given: ",elem)

        # Confirm that either all_clusters or separate_clusters has been set to True                    
        if not all_clusters and not separate_clusters:
            raise ValueError("Error! At least one of the default args all_clusters" + \
                             " or separate_clusters must be set to True. Was given" + \
                             " [all_clusters,separate_clusters] as: ",
                             [all_clusters,separate_clusters])
                
        # Determine number of clusters, folder location for created movies, and
        # temp_folder for holdings images before they're compiled into gifs
        num_clusters = len(self.orig_data_by_cluster)
        folder_loc =  self.output_folder + "/" + \
                ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                "/" + "3D_Clusters"
        temp_folder = folder_loc + "/" + "MakingGif"
        
        
        # For each triplet of variables to plot, collect data and make a movie
        clustered_data = []
        for i in range(len(var_lists)):    
            triplet = var_lists[i]
            xlabel, ylabel, zlabel = triplet[0], triplet[1], triplet[2]
            x_col = list(self.orig_data).index(xlabel)
            y_col = list(self.orig_data).index(ylabel)
            z_col = list(self.orig_data).index(zlabel)
            
            # notify user
            print("Processing [" + xlabel + " x " + ylabel + \
                      " x " + zlabel +"] data from",num_clusters,"clusters to",
                      "make movie: ",end="")
        
        
        
            # ========== COLLECT CLUSTERED DATA ==========
        
            """# if performing data reduction, then routine is a little different here...
            if use_reduced_data is not None:
                
                for n in range(num_clusters):
                    print(str(n+1) + " ",end="")
                    
                    x_data = (self.orig_data_by_cluster[n])[xlabel].values
                    y_data = (self.orig_data_by_cluster[n])[ylabel].values
                    z_data = (self.orig_data_by_cluster[n])[zlabel].values
                    xyz_array = np.transpose( \
                                np.vstack(tuple( [x_data,y_data,z_data] ))
                                        )
                    
                    # classify data of cluster into shapes
                    shapes_list = ML_Classifier.\
                                _perform_data_reduction(xyz_array,
                                res=enable_data_reduction,return_shape_list=True,
                                small_shape_limit=0)
                                #min_points_per_shape=?)
                    # keep small-num outliers? Yes or no? Think about it...
    
                    # filter out any shapes lacking min number of points
                    # ^^^ actually, that job should be handled in _perform_data_reduction                            
                    clustered_data.append( shapes_list )
            
            # If NOT performing data reduction, then lets just put the data into list
            else:
                
                for n in range(num_clusters):
                    print(str(n+1) + " ",end="")
                    
                    x_data = (self.orig_data_by_cluster[n])[xlabel].values
                    y_data = (self.orig_data_by_cluster[n])[ylabel].values
                    z_data = (self.orig_data_by_cluster[n])[zlabel].values
                    xyz_array = np.transpose( \
                                np.vstack(tuple( [x_data,y_data,z_data] ))
                                        )
    
                    clustered_data.append( xyz_array )
                    
            print("")"""
            # ==============================================
        
        
                    
            # ========== PLOTTING ALL CLUSTERS TOGETHER ==========

            if all_clusters:    
            
                # Make fig and set labels and title
                fig = plt.figure(dpi=120)
                ax = fig.add_subplot(1,1,1,projection="3d")
                ax.set_xlabel( xlabel )
                ax.set_ylabel( ylabel )
                ax.set_zlabel( zlabel )
                #ax.set_title("Triangular-surface plotting of " + str(num_clusters) + \
                #             " Clusters",fontweight="bold")
                title_str = ""
                movie_name_str = ""
                    
                # notify user
                print("Plotting all clusters together! Making initial plot")
                
                
                # Collect data from each cluster and plot all trisurfaces in
                # same plot
                for n in range(num_clusters):
                    
                    
                    # ---------- GATHER POSIT DATA ----------
                    x_data, y_data, z_data = None, None, None
                     
                    # If using reduced data, then use that!
                    if use_reduced_data:
                        
                        # n --> which cluster, ...
                        #     "main" --> main points instead of neighbours, ...
                        #           [:,col] --> 1d array of data
                        x_data = self.reduced_data_by_cluster[n]["main"][:,x_col]
                        y_data = self.reduced_data_by_cluster[n]["main"][:,y_col]
                        z_data = self.reduced_data_by_cluster[n]["main"][:,z_col]
                        
                    # If data NOT reduced, then use original clustering
                    # (this might be pretty large!!!)
                    else:
                        
                        x_data = (self.orig_data_by_cluster[n])[xlabel].values
                        y_data = (self.orig_data_by_cluster[n])[ylabel].values
                        z_data = (self.orig_data_by_cluster[n])[zlabel].values
                    
                    # Stack x,y,z data into single n x 3 array
                    data_to_plot = np.transpose(
                            np.vstack( tuple( [x_data,y_data,z_data] ) )
                                                )
                    # ------------------------------------------
                    
                    
                    if heatmap_var is None:
                            
                        movie_name_str = "AllClusters"
                        title_str = "Reduced scatter-plotting of "
                        ax.scatter(data_to_plot[:,0],data_to_plot[:,1],
                                   data_to_plot[:,2],s=0.1,c=self.DEFAULT_COLORS[n],
                                   depthshade=False)
                            
                    """if heatmap_var is not None:
                        
                        movie_name_str = "AllClustersHeatmap"+heatmap_var
                        title_str = "Reduced scatter-plotting Heatmap of "
                        
                        heatmap_vals = \
                            self._build_heatmap_var_from_reduced_data(
                                self.reduced_data_by_cluster[n]["main"],
                                self.reduced_data_by_cluster[n]["neighbour"],
                                heatmap_var,
                                [xlabel,ylabel,zlabel]
                                                                )
                            
                        ax.scatter(data_to_plot[:,0],data_to_plot[:,1],
                                   data_to_plot[:,2],s=0.1,cmap="hot",
                                   c=heatmap_vals)"""
                    
                        
                    # -----------------------------------------
                    
                    
                # Plot bow shock and earth if x,y,z
                using_xyz_posit = (xlabel == "X (R_E)") and (ylabel == "Y (R_E)") \
                            and (zlabel == "Z (R_E)")
                if using_xyz_posit:
                    the_x_mins = [min(cluster_data[xlabel].values) for cluster_data \
                                    in self.orig_data_by_cluster]
                    global_x_min = min(the_x_mins) * 1.2
                    ML_Classifier._plot_magpause(ax,dims="3d",xmin=global_x_min)
                    ML_Classifier._plot_earth(ax)
                    
                
                # finish setting title
                title_str = title_str + str(num_clusters) + " Clusters"
                ax.set_title(title_str,fontweight="bold")
                
                # Make movie after all clusters have been plotted...
                print("... Making movie of rotated plots")
                self._make_movie_from_axis_obj(temp_folder,ax,
                                         framerate=framerate,
                                         rotate_horizontal=rotate_horizontal,
                                         rotate_vertical=rotate_vertical,
                                         plot_per_degrees=plot_per_degrees,
                                         movie_name=movie_name_str)
                
                # ... Then close figure
                plt.close()
            # ====================================================
        
        
        
            # ========== PLOTTING ALL CLUSTERS SEPARATELY ==========

            if separate_clusters: 
                
                # notify user
                print("Plotting clusters separately! Making movie for cluster: ",
                      end="")
                
                for n in range(num_clusters):
                    print(str(n+1) + " ",end="")
                    
                    # Make plot, labels, and title for each cluster
                    fig = plt.figure(dpi=120)
                    ax = fig.add_subplot(1,1,1,projection="3d")
                    ax.set_xlabel( xlabel )
                    ax.set_ylabel( ylabel )
                    ax.set_zlabel( zlabel )
                    #ax.set_title("Triangular-surface plotting of Cluster " + \
                    #             str(n+1),fontweight="bold")
                    title_str = ""
                    movie_name_str = ""
                    
                    
                    # ---------- GATHER POSIT DATA ----------
                    x_data, y_data, z_data = None, None, None
                     
                    # If using reduced data, then use that!
                    if use_reduced_data:
                        
                        # n --> which cluster, ...
                        #     "main" --> main points instead of neighbours, ...
                        #           [?]label --> data for component
                        x_data = self.reduced_data_by_cluster[n]["main"][:,x_col]
                        y_data = self.reduced_data_by_cluster[n]["main"][:,y_col]
                        z_data = self.reduced_data_by_cluster[n]["main"][:,z_col]
                        
                    # If data NOT reduced, then use original clustering
                    # (this might be pretty large!!!)
                    else:
                        
                        x_data = (self.orig_data_by_cluster[n])[xlabel].values
                        y_data = (self.orig_data_by_cluster[n])[ylabel].values
                        z_data = (self.orig_data_by_cluster[n])[zlabel].values
                    
                    # Stack x,y,z data into single n x 3 array
                    data_to_plot = np.transpose(
                            np.vstack( tuple( [x_data,y_data,z_data] ) )
                                            )
                    # ------------------------------------------
                    
                    if heatmap_var is None:
                        
                        movie_name_str = "Cluster"+str(n+1)
                        title_str = "Reduced scatter-plotting of "
                        ax.scatter(data_to_plot[:,0],data_to_plot[:,1],
                                   data_to_plot[:,2],s=0.1,c=self.DEFAULT_COLORS[n],
                                   depthshade=False)
                            
                    if heatmap_var is not None:
                        
                        movie_name_str = "Cluster"+str(n+1)+"_Heatmap_"+\
                            heatmap_var.split(" ")[0]
                        title_str = "Reduced scatter-plotting Heatmap of "
                        
                        # Get heatmap_vals based on heatmap_var
                        heatmap_vals = \
                            self._build_heatmap_var_from_reduced_data(
                                self.reduced_data_by_cluster[n]["main"],
                                self.reduced_data_by_cluster[n]["neighbour"],
                                heatmap_var,
                                [xlabel,ylabel,zlabel],
                                scaling="average"
                                                                )                        

                        # Then plot x,y,z scatter along with heatmap
                        the_scatter_thing = \
                            ax.scatter(data_to_plot[:,0],data_to_plot[:,1],
                                   data_to_plot[:,2],s=0.1,cmap="gist_rainbow",
                                   c=heatmap_vals, depthshade=False)
                        
                        # Add colorbar and set its label
                        the_colorbar = fig.colorbar(the_scatter_thing,
                                                    ax=ax)
                        the_colorbar.set_label(heatmap_var)
                        
                    
                    
                    
                    # Plot bow shock and earth if x,y,z
                    using_xyz_posit = (xlabel == "X (R_E)") and (ylabel == "Y (R_E)") \
                                        and (zlabel == "Z (R_E)")
                    if using_xyz_posit:
                        ML_Classifier._plot_magpause(ax,dims="3d",xmin=min(x_data)*1.2)
                        ML_Classifier._plot_earth(ax)
                        
                    
                    # Finish setting title
                    title_str = title_str + "Cluster " + str(n+1)
                    ax.set_title(title_str,fontweight="bold")
                    
                    # Make movie after a *single* cluster has been plotted ...
                    self._make_movie_from_axis_obj(temp_folder,ax,
                                             framerate=framerate,
                                             rotate_horizontal=rotate_horizontal,
                                             rotate_vertical=rotate_vertical,
                                             plot_per_degrees=plot_per_degrees,
                                             movie_name=movie_name_str)
                    
                    # ... Then close figure
                    plt.close()
                    
                print("")
            # ======================================================
            
            
    

    
    
    
    
        
        
        
        
        
        
        
    # Plots all clusters found on the variable plots specified
    # N clusters --> each subplot
    # vars_lists and extra_angles behave in the same way as described in
    # plot_separate_clusters_on_vars
    def plot_all_clusters_on_vars(self,var_lists,extra_angles=None):
        print("Plotting",len(self.orig_data_by_cluster),
              "clusters across variable pairs: ")
        

        # Partition var_lists into two lists: one of pairs (for 2D plots)
        # and one of triplets (for 3D plots)
        var_lists_2D = [elem for elem in var_lists if len(elem) == 2]
        var_lists_3D = [elem for elem in var_lists if len(elem) == 3]
        
        
        # ========== PLOTTING 2D ==========
        
        if (len(var_lists_2D) > 0):
            
            # Fig params
            num_plots = len(var_lists_2D)
            num_clusters = len(self.orig_data_by_cluster)
            plots_per_row = ML_Classifier._get_plots_per_row(num_plots)
            subplots_x = math.ceil( num_plots/plots_per_row )
            subplots_y = plots_per_row  
            last_axis_coord = [ int(num_plots/plots_per_row), num_plots % plots_per_row ]
            fig, axes = plt.subplots(subplots_x, subplots_y,
                                         figsize=self.MULTIPLOT_SIZE, dpi=120)
            
            
            # --- Create subplot for each pair of variables in var_lists_2D ---
            
            print("    ",end="")
            for i in range(num_plots):
                pair = var_lists_2D[i]
                xlabel, ylabel = pair[0], pair[1]
                print("[",xlabel,"x",ylabel,"] ",end="")
                
                
                # get axis to be used for current plot
                the_axis = ML_Classifier\
                    ._get_nth_item_from_arb_collection(i,axes,coll_type=np.ndarray)
                
                
                # If x,y or z,x, plot bow shock contour
                is_X_vs_Y = (xlabel == "X (R_E)") and (ylabel == "Y (R_E)")
                is_X_vs_Z = (xlabel == "X (R_E)") and (ylabel == "Z (R_E)")
                if is_X_vs_Y or is_X_vs_Z:
                    the_x_mins = [min(cluster_data[xlabel].values) for cluster_data \
                                    in self.orig_data_by_cluster]
                    global_x_min = min(the_x_mins)
                    ML_Classifier._plot_magpause(the_axis,xmin=global_x_min)
                
                
                # Retrieve data from each cluster and add it to current subplot
                for n in range(num_clusters):
                    x_data = (self.orig_data_by_cluster[n])[pair[0]].values
                    y_data = (self.orig_data_by_cluster[n])[pair[1]].values
                    the_axis.scatter(
                        x_data,y_data,label="Cluster " + str(n+1), s=60,
                        facecolors="None",edgecolors=self.DEFAULT_COLORS[n],
                        linewidth=2,marker=self.DEFAULT_MARKERS[n]
                        )
                #the_axis.legend(loc="upper right")#,bbox_to_anchor=(0.5,-0.1),ncol=5)
                
                # Set labels and title
                the_axis.set_xlabel( xlabel )
                the_axis.set_ylabel( ylabel )
                the_axis.set_title(xlabel + " vs " + ylabel,fontweight="bold")
                
                
                # Shift each subplot slightly upwards (to make room for overall legend)
                # NOTE: (0,0) is at top-left of plots for graphics! So a POSITIVE
                # vert_shit means all the plots are pushed DOWN
                #box = the_axis.get_position()
                #vert_shift = -0.05
                #box.y0, box.y1 = box.y0 + vert_shift, box.y1 + vert_shift
                #the_axis.set_position(box)
            # -----------------------------------------------------------------
                
            
            # Delete unused subplots
            for elem in ML_Classifier._get_unused_axes_coords\
                        (last_axis_coord,[subplots_x,subplots_y]):
                [x_coord,y_coord] = elem
                axes[x_coord,y_coord].axis("off")
                
                
            # Modify overall subplot layout to make room for global legend
            fig.tight_layout(pad=1.0)
            plt.subplots_adjust(top=0.9,bottom=0.15,hspace=0.4)
            #plt.tight_layout(pad=3.0)
            first_axis = ML_Classifier\
                ._get_nth_item_from_arb_collection(0,axes,coll_type=np.ndarray)
            handles, labels = first_axis.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',ncol=5)
            
            
            # Create folder to place in if it doesn't already exist
            folder_loc = self.output_folder + "/" + \
                ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                "/" + "2D_Clusters"
            if not os.path.exists(folder_loc):
                os.makedirs(folder_loc)
                
                
            # Make title, save fig, and close
            title_str = "Distribution of " + str(num_clusters) + \
                " clusters Across Variables"
            fig.suptitle(title_str,fontsize=24)
            fig_name = str(num_clusters) + "_clusters_plotted_on_vars.png"
            plt.savefig(folder_loc + "/" + fig_name) 
            plt.close()
        # ========== END PLOTTING 2D ==========
        
        
        # ========== PLOTTING 3D ==========
        
        if (len(var_lists_3D) > 0):
            
            # elev --> angle above x-y plane  ;  azim --> angle along azimuth
            [orig_elev, orig_azim] = self.DEFAULT_3D_VIEWING_ANGLES
            
            angles = [(orig_elev,orig_azim)]
            if extra_angles is not None:
                angles = ML_Classifier._determine_angles(angles[0],extra_angles)
            
            for angle_pair in angles:
            
                # Fig params
                num_plots = len(var_lists_3D)
                num_clusters = len(self.orig_data_by_cluster)
                plots_per_row = ML_Classifier._get_plots_per_row(num_plots)
                subplots_x = math.ceil( num_plots/plots_per_row )
                subplots_y = plots_per_row  
                last_axis_coord = [ int(num_plots/plots_per_row), num_plots % plots_per_row ]
                fig = plt.figure(figsize=self.MULTIPLOT_SIZE, dpi=120)
                #ax = fig.add_subplot(subplots_y,subplots_x,1, projection='3d')
                #fig, axes = plt.subplots(subplots_x, subplots_y,
                #                             figsize=self.MULTIPLOT_SIZE, dpi=120)
                
                
                # --- Create subplot for each pair of variables in var_lists_2D ---
                
                print("    ",end="")
                for i in range(num_plots):
                    triplet = var_lists_3D[i]
                    xlabel, ylabel, zlabel = triplet[0], triplet[1], triplet[2]
                    print("[",xlabel,"x",ylabel,"x",zlabel,"] ",end="")
                    
                    
                    # get axis to be used for current plot
                    #the_axis = axes
                    #if (num_plots > 1):
                    #    the_axis = ML_Classifier._get_nth_item_from_arb_collection\
                    #                (i,axes,coll_type=np.ndarray)
                    the_axis = fig.add_subplot(subplots_y,subplots_x,i+1, projection='3d')
                    
                    
                    # Retrieve data from each cluster and add it to current subplot
                    for n in range(num_clusters):
                        x_data = (self.orig_data_by_cluster[n])[triplet[0]].values
                        y_data = (self.orig_data_by_cluster[n])[triplet[1]].values
                        z_data = (self.orig_data_by_cluster[n])[triplet[2]].values
                        """the_axis.scatter(
                            x_data,y_data,z_data,label="Cluster " + str(n+1), s=60,
                            facecolors="None",edgecolors=self.DEFAULT_COLORS[n],
                            linewidth=2,marker=self.DEFAULT_MARKERS[n]
                            )"""
                        the_axis.scatter(
                            x_data,y_data,z_data,label="Cluster " + str(n+1), s=60,
                            edgecolors=self.DEFAULT_COLORS[n],linewidth=2,
                            facecolor=(0,0,0,0),marker=self.DEFAULT_MARKERS[n]
                            )
                    #the_axis.legend(loc="upper right")#,bbox_to_anchor=(0.5,-0.1),ncol=5)
                    the_axis.set_xlabel( xlabel )
                    the_axis.set_ylabel( ylabel )
                    the_axis.set_zlabel( zlabel )
                    the_axis.view_init(angle_pair[0],    # elev first
                                       angle_pair[1])    # then azim
                    the_axis.set_title(xlabel + " vs " + ylabel + " vs " + zlabel,
                                       fontweight="bold")
                    
                    # Plot bow shock if x,y,z
                    using_xyz_posit = (xlabel == "X (R_E)") and (ylabel == "Y (R_E)") \
                                        and (zlabel == "Z (R_E)")
                    if using_xyz_posit:
                        the_x_mins = [min(cluster_data[xlabel].values) for cluster_data \
                                    in self.orig_data_by_cluster]
                        global_x_min = min(the_x_mins)
                        ML_Classifier._plot_magpause(the_axis,dims="3d",
                                                      xmin=1.1*global_x_min)
                    
                    # Shift each subplot slightly upwards (to make room for overall legend)
                    # NOTE: (0,0) is at top-left of plots for graphics! So a POSITIVE
                    # vert_shit means all the plots are pushed DOWN
                    #box = the_axis.get_position()
                    #vert_shift = -0.05
                    #box.y0, box.y1 = box.y0 + vert_shift, box.y1 + vert_shift
                    #the_axis.set_position(box)
                # -----------------------------------------------------------------
                    
                
                # Delete unused subplots
                for elem in ML_Classifier._get_unused_axes_coords\
                            (last_axis_coord,[subplots_x,subplots_y]):
                    [x_coord,y_coord] = elem
                    axes[x_coord,y_coord].axis("off")
                    
                    
                # Modify overall subplot layout to make room for global legend
                fig.tight_layout(pad=1.0)
                plt.subplots_adjust(top=0.9,bottom=0.15,hspace=0.4)
                #plt.tight_layout(pad=3.0)
                first_axis = ML_Classifier\
                    ._get_nth_item_from_arb_collection(0,axes,coll_type=np.ndarray)
                handles, labels = first_axis.get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center',ncol=5)
                
                
                # Create folder to place in if it doesn't already exist
                folder_loc = self.output_folder + "/" + \
                    ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                    "/" + "3D_Clusters"
                if not os.path.exists(folder_loc):
                    os.makedirs(folder_loc)
                    
                    
                # Make title, save fig, and close
                title_str = "Distribution of " + str(num_clusters) + \
                    " clusters Across Variables"
                fig.suptitle(title_str,fontsize=24)
                angle_info = "elev" + str(angle_pair[0]) + "_azim" + \
                        str(angle_pair[1])
                fig_name = str(num_clusters) + "_clusters_plotted_on_vars_3D_" + \
                        angle_info + ".png"
                plt.savefig(folder_loc + "/" + fig_name) 
                plt.close()
        # ========== END PLOTTING 3D ==========
            
        print("")
        
        
        
        
        
        
        
        
        
        
        
    # Creates histograms of variables in original dataframe
    # do_not_plot_cols is list of variable names whose histograms SHOULDN'T
    # be plotted
    # log_y_axis - default arg that turns histogram y-axis into log format.
    def plot_cluster_hists(self,do_not_plot_cols=[],log_y_axis=None):
        print("Creating histograms of variables for",
              len(self.orig_data_by_cluster),"clusters: ",end="")
        
        
        # default args
        if log_y_axis is None: log_y_axis = False
        
        
        # Create fig params
        num_plots = len(list(self.norm_data)) - len(do_not_plot_cols)
        num_clusters = len(self.orig_data_by_cluster)
        plots_per_row = ML_Classifier._get_plots_per_row(num_plots)
        subplots_x = math.ceil( num_plots/plots_per_row )
        subplots_y = plots_per_row  
        last_axis_coord = [ int(num_plots/plots_per_row), num_plots % plots_per_row ]
                    
                
        # Filter specified columns out of both total data and
        # previously clustered data
        clustered_data = []
        for i in range(len(self.orig_data_by_cluster)):
            clustered_data.append( ML_Classifier._filter_cols_from_dataframe\
                                    (self.orig_data_by_cluster[i], do_not_plot_cols) )
        total_data = ML_Classifier\
            ._filter_cols_from_dataframe(self.orig_data, do_not_plot_cols)
            
            
        # --- For each cluster, plot comparisons of the cluster ---
        # --- data to the original data ---------------------------
            
        clusters = np.arange(num_clusters+1)[1:]
        for n in clusters:
            print(n," ",end="")
            fig, axes = plt.subplots(subplots_x, subplots_y,
                                     figsize=self.MULTIPLOT_SIZE, dpi=120)
            dataframe_in_cluster = clustered_data[n-1]
            #print(dataframe_in_cluster)
            
            
            # Create each subplot
            for i in range(num_plots):
                
                
                # Check to see if variable is discrete or appx continuous
                [unique_vals_in_cluster,val_count_in_cluster] = \
                    np.unique(dataframe_in_cluster.iloc[:,i].values,return_counts=True)
                [unique_vals_total,val_count_total] = \
                    np.unique(total_data.iloc[:,i].values,return_counts=True)
                num_unique_points = len(unique_vals_total)
                cluster_occupancy = sum(val_count_in_cluster) / sum(val_count_total)
                
                
                # check to see if column uses non-numeric data
                col_sample = total_data.iloc[0,i]   #.values
                current_label = list(total_data)[i]
                is_continuous = False
                is_discrete = False
                if ML_Classifier._is_number(col_sample):
                    is_continuous = num_unique_points > self.DISCRETE_LIMIT
                    is_discrete = True if not is_continuous else False
                else:
                    is_continuous = False
                    is_discrete = True
                    
                    
                # get axis to be used for current subplot
                the_axis = ML_Classifier\
                        ._get_nth_item_from_arb_collection(i,axes,coll_type=np.ndarray)
                    
                    
                # ***** "Continuous" Data Comparison *****
            
                # If continuous, compare histograms
                if is_continuous:
                    
                    
                    # Prepare bins for histograms - so as to use the same res
                    # for both data sets!
                    hist_bins = self._compute_hist_bins(total_data.iloc[:,i].values)
    
    
                    # Create background histogram for total data
                    total_avg = np.average( total_data.iloc[:,i].values )
                    total_stddev = np.std( total_data.iloc[:,i].values )
                    total_stats_color = "green"
                    #total_label = "Total (" + total_stats_color + ")"
                    total_label = "μ = " + "{:.2f}".format(total_avg) + " | " + \
                                    "σ = " + "{:.2f}".format(total_stddev)
                    the_axis.hist(total_data.iloc[:,i].values,
                                    alpha=0.5,color="red",label=total_label,
                                    bins=hist_bins,log=log_y_axis)
                    ML_Classifier._plot_avg_and_stddev(the_axis,
                                    total_data.iloc[:,i].values,
                                    alpha=1.0,color=total_stats_color,
                                    branch_point=0.7)
                    
                    
                    # Create foreground histogram for cluster data
                    cluster_avg = np.average( dataframe_in_cluster.iloc[:,i].values )
                    cluster_stddev = np.std( dataframe_in_cluster.iloc[:,i].values )
                    cluster_stats_color = "black"
                    #cluster_label = "Cluster " + str(n) + " (" + cluster_stats_color + ")"
                    cluster_label = "μ = " + "{:.2f}".format(cluster_avg) + " | " + \
                                    "σ = " + "{:.2f}".format(cluster_stddev)
                    the_axis.hist(dataframe_in_cluster.iloc[:,i].values,
                                             alpha=0.5,color="blue",label=cluster_label,
                                             bins=hist_bins,log=log_y_axis)
                    ML_Classifier._plot_avg_and_stddev(the_axis,
                                    dataframe_in_cluster.iloc[:,i].values,\
                                    alpha=1.0,color=cluster_stats_color,
                                    branch_point=0.5)
                        
                    
                    # Create legend for both histograms
                    the_axis.legend(loc="upper right",prop={"size":7})
                    
                    
                    # If data column was NOT used in KMeans clustering,
                    # then make title red
                    col_not_used_for_clustering = current_label in self.kmeans_ignored_cols
                    if col_not_used_for_clustering:
                        the_axis.set_title( list(total_data)[i], color="red")
                    else:
                        the_axis.set_title( list(total_data)[i] )
                        
                    
                    # Show grid for better visuals
                    the_axis.grid(True)
                # ****************************************
                    
                
                # ***** Discrete Data Comparisons *****
                
                # compare overlap of variables individually
                elif is_discrete:
                    rows = unique_vals_total
                    cols = ["# pts in cluster " + str(n),"# pts total"," % of total"]
                    
                    
                    # Create table with rows being different unique values and
                    # columns showing total amount e.g.
                    #     N/A  |  # in cluster  |  # total  |  % of total in cluster
                    #  "Var A" ------ 5 ------------- 10 ---------- 50 % -----
                    data_for_table = np.zeros((len(unique_vals_total),len(cols)))
                    for q in range(len(unique_vals_total)):
                        metaarray_of_sim_cols = np.where(unique_vals_in_cluster == unique_vals_total[q])
                        num_val_in_cluster = -1
                        if ( len(metaarray_of_sim_cols[0]) == 0 ): num_val_in_cluster = 0
                        else: num_val_in_cluster = val_count_in_cluster[ metaarray_of_sim_cols[0][0] ]
                        data_for_table[q,0] = int( num_val_in_cluster )
                        data_for_table[q,1] = int( val_count_total[q] )
                        data_for_table[q,2] = 100 * num_val_in_cluster / val_count_total[q]
                        
                        
                    # Check to see if more than row limit... if so, keep those with
                    # highest cluster population in overall data (and show how much is
                    # presented!)
                    too_many_rows = len(unique_vals_total) > self.ROW_TABLE_LIMIT
                    if too_many_rows:
                        
                        
                        # Sort according to cluster representation in a variable
                        # (highest percentage first)
                        sorted_col_indices = data_for_table[:,0].argsort()[::-1]
                        data_for_table = data_for_table[sorted_col_indices]
                        
                        
                        # See how much of the cluster is represent in the top specified
                        # number of rows
                        cluster_fraction = sum(data_for_table[:self.ROW_TABLE_LIMIT,0])\
                                        / sum(data_for_table[:,0])
                                        
                                        
                        # Then take row slices up to the row_table_limit for BOTH
                        # data_for_table and rows
                        data_for_table = data_for_table[:self.ROW_TABLE_LIMIT,:]
                        rows = (rows[sorted_col_indices])[:self.ROW_TABLE_LIMIT]
                        
                        
                    # Convert numpy array into formatted strings
                    data_for_table = [ ["{:.2f}".format(num) for num in row] for row in data_for_table ]
                        
                    
                    # Now turn table into a graphic
                    #print(data_for_table)
                    the_axis.table(cellText=data_for_table,
                          rowLabels=rows,
                          #rowColours=colors,
                          colLabels=cols,
                          loc='center')
                    the_axis.axis("off")
                    table_title = list(total_data)[i]
                    
                    
                    # If too many rows, then specify how much of cluster is visible
                    if too_many_rows:
                        table_title = table_title + " (top " + str(self.ROW_TABLE_LIMIT) + \
                            " shown with " + "{:.2f}".format(100*cluster_fraction) + \
                                " % of cluster)"
                                
                                
                    #also offset this upwards a little
                    # and if data used for cluster, make title red
                    col_not_used_for_clustering = current_label in self.kmeans_ignored_cols
                    if col_not_used_for_clustering:
                        the_axis.set_title( table_title, y=1.0, pad=10, color="red")
                    else:
                        the_axis.set_title( table_title, y=1.0, pad=10)
                # *******************************************
                        
                        
            # Delete unused subplots
            for elem in ML_Classifier._get_unused_axes_coords(last_axis_coord,[subplots_x,subplots_y]):
                [x_coord,y_coord] = elem
                axes[x_coord,y_coord].axis("off")
                
            
            # Create folder to hold figure if it doesn't already exist
            # Is of the form #_clusters (where # is num clusters)
            folder_loc = self.output_folder + "/" + \
                    ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                    "/" + "Histograms"
            if not os.path.exists(folder_loc):
                os.makedirs(folder_loc)
            
            # Save fig and close
            title_str = "Histograms of Cluster " + str(n) + \
                        " (out of " + str(num_clusters) + ") relative to Total Data\n" + \
                        "Cluster Occupancy: " + "{:.2f}".format(cluster_occupancy*100) + \
                        " % of All Data"
            title_str = title_str + "\n" + \
                        "Total Data in Red / Green | Cluster Data in Purple / Black"
            if log_y_axis: title_str = "LOG-"+title_str
            fig.suptitle(title_str,fontsize=20)#fontsize=24)
            fig.tight_layout()
            fig_name = "Cluster_Comparison_" + str(n) + "_out_of_" + \
                    str(len(clusters)) + ".png"
            plt.savefig(folder_loc + "/" + fig_name) 
            plt.close()
            # --------------------------------------------------
        print("")
            
            
            
            
            
            
            
            
            
            
    # Creates histograms of cluster distances of all points relative to each
    # centroid
    def plot_cluster_distances(self):
        num_clusters = len(self.orig_data_by_cluster)
        print("Creating histogram of relative cluster distances between",
                  num_clusters,"clusters")
            
            
        # Fig params
        num_plots = num_clusters
        plots_per_row = ML_Classifier._get_plots_per_row(num_plots)
        num_cols_plots = plots_per_row
        num_rows_plots = math.ceil( num_plots / plots_per_row )
        last_axis_coord = [ int(num_plots/plots_per_row), num_plots % plots_per_row ]
        fig, axes = plt.subplots(num_rows_plots, num_cols_plots,
                                 figsize=self.MULTIPLOT_SIZE, dpi=120)
        
        
        # Need to organize the data by cluster again, but this time the
        # normalized data (so as to get the correct weights from KMeans)
        norm_data_by_cluster_filtered = []
        for i in range(len(self.norm_data_by_cluster)):
            norm_data_by_cluster_filtered.append(
                    ML_Classifier._filter_cols_from_dataframe\
                        (self.norm_data_by_cluster[i],self.kmeans_ignored_cols)
                )
        
        
        # For n specified clusters, need to plot the histogram of the weights
        # for each one.
        for i in range(num_plots):
            
            
            # For each cluster, save distances of data relative to cluster center
            # (even data not found in cluster, which is why we use the flatten())
            all_weights_rel_to_cluster = self.kmeans\
                .transform( norm_data_by_cluster_filtered[i] )
            weights_in_cluster = all_weights_rel_to_cluster[:,i]
            all_weights_rel_to_cluster = all_weights_rel_to_cluster.flatten()
            
            
            # all_weights would hold the largest differences, so base the hist
            # off of that
            hist_bins = self._compute_hist_bins(all_weights_rel_to_cluster)
            
            
            # get axis to be used for current plot
            the_axis = ML_Classifier\
                ._get_nth_item_from_arb_collection(i,axes,coll_type=np.ndarray)
            
            
            # Plot all weights rel to cluster first...
            total_color = "green"
            total_label = "All Data (" + total_color + ")"
            the_axis.hist(all_weights_rel_to_cluster,alpha=0.7,\
                                     color="red",bins=hist_bins,label=total_label)
            ML_Classifier._plot_avg_and_stddev(the_axis,all_weights_rel_to_cluster,\
                            alpha=1.0,color=total_color,branch_point=0.7)
            
                
            # Then the distances of the data in the cluster
            cluster_color = "black"
            cluster_label = "Cluster " + str(i+1) + " Data (" + \
                            cluster_color + ")"
            the_axis.hist(weights_in_cluster,alpha=0.5,\
                                     color="blue",bins=hist_bins,\
                                   label=cluster_label)
            ML_Classifier._plot_avg_and_stddev(the_axis,weights_in_cluster,\
                            alpha=1.0,color=cluster_color,branch_point=0.5)
            
            
            # Create title for figure
            the_axis.set_title("Distances from Cluster " + str(i+1) + " Centroid")
            the_axis.legend(loc="upper right",prop={"size":7})
            the_axis.grid(True)
                
            
        # Delete unused subplots
        if ( (num_rows_plots > 1) and (num_plots > 1) ):
            for elem in ML_Classifier._get_unused_axes_coords(last_axis_coord,[num_rows_plots,num_cols_plots]):
                [x_coord,y_coord] = elem
                axes[x_coord,y_coord].axis("off")
                
                
        # Create folder to contain file if it doesn't already exist
        folder_loc = self.output_folder + "/" + \
                ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters)) + \
                "/" + "Histograms"
        if not os.path.exists(folder_loc):
                os.makedirs(folder_loc)
        
        
        # Save fig
        title_str = "Histogram of Distances Across " + str(num_plots) + " Clusters"
        plt.suptitle(title_str,fontsize=24)
        plt.tight_layout()
        fig_name = "KMeans_Weights_Hists_for_" + str(num_plots) + "_Clusters.png"
        plt.savefig(folder_loc + "/" + fig_name)
        plt.close()
        
        
        
        
        
        
        
        
        
        
    # Runs the KMeans clustering algorithm on the data in self.norm_data.
    # Then (1) a 1D array (of length == norm_data.shape[0]) where the number
    # at an index denotes which cluster the row of that same index in norm_data
    # belongs to, (2) the original data is classified into a list of array
    # where each array is the data belonging to that cluster, and (3)
    # the same as (2) but for norm_data.
    def run_kmeans(self,num_clusters,max_iter=None,cols_to_ignore=[]):
        print("Running KMeans for",num_clusters,"clusters ")
        self.num_clusters = num_clusters
        
        
        # Use default max number of iters if unspecified
        if max_iter is None:
            max_iter = self.DEFAULT_KMEANS_MAX_NUM_ITERS
        
        
        # Collect indices of columns to remove into class variable and
        # filter out columns not used for clustering
        cols_to_ignore_labels = ML_Classifier\
            ._get_list_elems_from_indices(list(self.norm_data),cols_to_ignore)
        self.kmeans_ignored_cols = cols_to_ignore_labels
        data_for_clustering = ML_Classifier\
            ._filter_cols_from_dataframe(self.norm_data,cols_to_ignore_labels)
        
        
        # Run KMeans
        self.kmeans = KMeans(n_clusters=num_clusters,
                        max_iter=self.DEFAULT_KMEANS_MAX_NUM_ITERS)\
                        .fit(data_for_clustering.values)
        
        
        # See which cluster data points belong to (both orig data and norm data)
        self.indices_by_cluster = self.kmeans.predict(data_for_clustering.values)
        self.orig_data_by_cluster = ML_Classifier\
            ._organize_data_by_cluster(self.indices_by_cluster, self.orig_data)
        self.norm_data_by_cluster = ML_Classifier\
            ._organize_data_by_cluster(self.indices_by_cluster, self.norm_data)
            
    
    
    
    
    
    
    
    
    """
    def compute_accuracy(self,return_accuracy=False):
        # In the future, accept labeled data in the future for comparison
        # with the classified data! But for now, just work with mms data
        num_clusters = len(self.orig_data_by_cluster)
        if (num_clusters != 2):
            print("This only works with two clusters for now!!! Quitting...")
            return
        print("Estimating Accuracy for",num_clusters,"clusters.")
        
        
        # The rest of this function is kind of tailored for the dataframe to
        # be in a certain way, so placing this here for ease of access in case
        # things change
        x_data_label = "X (R_E)"
        y_data_label = "Y (R_E)"
        
        
        # Build Bow Shock Ray Tracer and determine which points are inside
        num_lines_res = 100
        x_data = self.orig_data[x_data_label].values
        y_data = self.orig_data[y_data_label].values
        points_to_plot = []
        for i in range(len(x_data)):
            points_to_plot.append([x_data[i],y_data[i]])
        from Bow_Shock_Ray_Tracer import Bow_Shock_Ray_Tracer
        # these physics params taken from avg of bz, proton num density, and
        # speed from OMNIWeb data for 20181201 - 20181231
        bow_shock_raytrace = Bow_Shock_Ray_Tracer\
                    (points_to_plot,num_lines=num_lines_res,
                     num_dens=ML_Classifier.DEFAULT_NUM_DENS,
                     speed=ML_Classifier.DEFAULT_SPEED,
                     bz=ML_Classifier.DEFAULT_BZ)
        is_raytraced_point_inside_MS = bow_shock_raytrace.is_inside
        
        
        # Now build r values for data BY CLUSTER; use these to determine which
        # cluster from KMeans best corresponds to the data points inside the
        # magnetosphere
        avg_r_vals_by_cluster = []
        for single_cluster_data in self.orig_data_by_cluster:
            x_data_cluster = single_cluster_data[x_data_label].values
            y_data_cluster = single_cluster_data[y_data_label].values
            r_vals = []
            for i in range(len(single_cluster_data)):
                r_vals.append( 
                        math.sqrt( x_data_cluster[i]**2 + y_data_cluster[i]**2 )
                            )
            avg_r_vals_by_cluster.append( sum(r_vals) / len(r_vals) )
        MS_cluster = avg_r_vals_by_cluster.index( min(avg_r_vals_by_cluster) )
        IMF_cluster = avg_r_vals_by_cluster.index( max(avg_r_vals_by_cluster) )
        
        # Now iterate through all data, tracking which data points are counted
        # as inside by both the ray tracing algorithm and the KMeans clustering.
        points_inside_correct = []
        points_inside_incorrect = []
        points_outside_correct = []
        points_outside_incorrect= []
        for i in range(len(self.indices_by_cluster)):
            which_cluster = self.indices_by_cluster[i]
            the_point = [ x_data[i] , y_data[i] ]
            # both KMeans and raytracing agree point is inside
            if ((which_cluster == MS_cluster) and is_raytraced_point_inside_MS[i]):
                points_inside_correct.append( the_point )
            # KMeans thinks point is inside but raytracing disagrees
            if ((which_cluster == MS_cluster) and not is_raytraced_point_inside_MS[i]):
                points_inside_incorrect.append( the_point )
            # KMeans thinks point is outside and raytracing agrees
            if ((which_cluster == IMF_cluster) and not is_raytraced_point_inside_MS[i]):
                points_outside_correct.append( the_point )
            # KMeans thinks point is outside but raytracing disagrees
            if ((which_cluster == IMF_cluster) and is_raytraced_point_inside_MS[i]):
                points_outside_incorrect.append( the_point )
        accuracy_frac = (len(points_inside_correct) + len(points_outside_correct))\
                        / len(x_data)
                
        
        # --- START MAKING PLOTS ---
        
        # Knowing how well (or poorly) KMeans and raytracing agree, create 4 plots:
        # The original X vs Y plot of clusters, the raytraced points,
        # X vs Y for inner cluster (with bad points noted), and X vs Y for
        # outer cluster (also with bad points noted).
        fig, axes = plt.subplots(2,2,figsize=(12,8))


        # X vs Y plot with all clusters on it
        the_axis = axes[0,0]
        for n in range(num_clusters):
                x_data = (self.orig_data_by_cluster[n])[x_data_label].values
                y_data = (self.orig_data_by_cluster[n])[y_data_label].values
                the_axis.scatter(
                    x_data,y_data,label="Cluster " + str(n+1), s=60,
                    facecolors="None",edgecolors=self.DEFAULT_COLORS[n],
                    linewidth=2,marker=self.DEFAULT_MARKERS[n]
                    )
            #the_axis.legend(loc="upper right")#,bbox_to_anchor=(0.5,-0.1),ncol=5)
        the_axis.set_xlabel( x_data_label )
        the_axis.set_ylabel( y_data_label )
        the_axis.set_title("Both Clusters on " + x_data_label + " vs " + \
                           y_data_label,fontweight="bold")           
        ML_Classifier._plot_bow_shock(the_axis)
        the_axis.legend(loc="upper right")
        
        
        # The raytraced plot
        bow_shock_raytrace.plot_data(axis_given=axes[0,1])
        ML_Classifier._plot_bow_shock(axes[0,1])
        axes[0,1].set_title("Ray-traced Data",fontweight="bold")
        #bow_shock_raytrace.end_me()
        
        
        # X and Y for inner cluster (with bad points noted)
        bad_marker = self.DEFAULT_MARKERS[-1]
        bad_color = self.DEFAULT_COLORS[-1]
        good_MS_data = len(points_inside_correct) > 0
        bad_MS_data = len(points_inside_incorrect) > 0
        if good_MS_data:
            x_data_MS_good = np.asarray(points_inside_correct)[:,0]
            y_data_MS_good = np.asarray(points_inside_correct)[:,1]
            axes[1,0].scatter(
                    x_data_MS_good,y_data_MS_good,label="Inside",
                    s=60,facecolors="None",edgecolors=self.DEFAULT_COLORS[MS_cluster],
                    linewidth=2,marker=self.DEFAULT_MARKERS[MS_cluster]
                        )
        if bad_MS_data:
            x_data_MS_bad = np.asarray(points_inside_incorrect)[:,0]
            y_data_MS_bad = np.asarray(points_inside_incorrect)[:,1]
            axes[1,0].scatter(
                    x_data_MS_bad,y_data_MS_bad,label="Outside",
                    s=60,facecolors="None",edgecolors=bad_color,
                    linewidth=2,marker=bad_marker
                        )
        if good_MS_data or bad_MS_data:
            axes[1,0].legend(loc="upper right")
            axes[1,0].set_title("Cluster "+str(MS_cluster+1)+" Accuracy",
                        fontweight="bold")
            ML_Classifier._plot_bow_shock(axes[1,0])
        else:
            axes[1,0].set_title("No MS Data Found!")
        
        
        # X and Y for outer cluster (with bad points noted)
        good_IMF_data = len(points_outside_correct) > 0
        bad_IMF_data = len(points_outside_incorrect) > 0
        if good_IMF_data:
            x_data_IMF_good = np.asarray(points_outside_correct)[:,0]
            y_data_IMF_good = np.asarray(points_outside_correct)[:,1]
            axes[1,1].scatter(
                x_data_IMF_good,y_data_IMF_good,label="Outside",
                s=60,facecolors="None",edgecolors=self.DEFAULT_COLORS[IMF_cluster],
                linewidth=2,marker=self.DEFAULT_MARKERS[IMF_cluster]
                    )
        if bad_IMF_data:
            x_data_IMF_bad = np.asarray(points_outside_incorrect)[:,0]
            y_data_IMF_bad = np.asarray(points_outside_incorrect)[:,1]
            axes[1,1].scatter(
                    x_data_IMF_bad,y_data_IMF_bad,label="Inside",
                    s=60,facecolors="None",edgecolors=bad_color,
                    linewidth=2,marker=bad_marker
                        )
        if good_IMF_data or bad_IMF_data:
            axes[1,1].legend(loc="upper right")
            axes[1,1].set_title("Cluster "+str(IMF_cluster+1)+" Accuracy",
                        fontweight="bold")
            ML_Classifier._plot_bow_shock(axes[1,1])
        else:
            axes[1,1].set_title("No IMF Data Found!")
        
        
        # Overall fig stuff
        percent_acc = "{:.2f}".format(accuracy_frac * 100)
        the_title = "KMeans Clustering Accuracy (BS raytraced using " + \
                str(num_lines_res) + " lines): " + percent_acc
        plt.suptitle(the_title,fontsize=24)
        fig.tight_layout(pad=1.0)
        plt.subplots_adjust(top=0.9)#,bottom=0.15,hspace=0.4)
        cluster_folder = self.output_folder + "/" + \
                ML_Classifier.CLUSTER_FOLDER.replace("#",str(num_clusters))
        plt.savefig(cluster_folder + "/" + "KMeans_Accuracy.png")
        plt.close()
        
        
        # Finally, return accuracy if specified
        if return_accuracy:
            return accuracy_frac"""
        
    
    
    
    

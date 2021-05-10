#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:40:00 2021

@author: jedmond
"""

from SSI_Data_Manager import SSI_Data_Manager
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, importlib



# the_list is EITHER a list OR a list of lists; regardles, the basic elements of
# said list will be strings.
# this return a copied list where the elements are the values found in the_dict
def replace_list_elements_with_values_from_dict(the_list,the_dict):
    
    # confirm that all values in the_list exist as keys in the_dict
    new_list = []
    for elem in the_list:
        # If str, just replace it based on dict
        if isinstance(elem,str): new_list.append( the_dict[elem] )
        # If another list, parse then ane replace sub elements based on dict
        if isinstance(elem,list):
            new_sub_list = []
            for sub_elem in elem:
                new_sub_list.append( the_dict[sub_elem] )
            new_list.append( new_sub_list )
    return new_list
        





# Imports codes specified in args
def get_libraries(source_folder,source_files):
    
    # Find paths containing the source_files
    paths_to_keep = []
    for root, dirs, filenames in os.walk(source_folder):
        for name in filenames:
            if name in source_files:
                paths_to_keep.append( root )
                
    # Do the really, really, bad-bad cheat way ---
    # just append the path of the source file to sys.path!
    for the_path in paths_to_keep:
        sys.path.append(the_path)
        
    # import codes
    rawnames_source_files = [item[:item.rfind(".")] for item in source_files]
    for rawname_file in rawnames_source_files:
        mymodule = importlib.import_module(rawname_file)
        the_class = getattr(mymodule,rawname_file)
        globals()[rawname_file] = the_class











# ========== PARAMS ==========

# Custom python code that needs to be imported (but is not located within
# the same folder as this driver)
source_folder = "/home/jedmond/Documents/ML_Research/Source"
source_files = ["ML_Classifier.py"] # "SSI_Data_Prepper.py"
get_libraries(source_folder,source_files)


# folder where all figures and movies will be placed
main_output_folder = "/home/jedmond/Documents/ML_Research/KMeans_Runs/CurrentRun"


# For gathering themis data
fig_drop_folder = main_output_folder + "/inst_plots"
csv_drop_folder = main_output_folder[:main_output_folder.rfind("/")+1] + "saved_CSVs"
# remove month 200810; V_Y and V_Z folders are empty for some reason?
subfolders = [ "200801", "200802", "200803", "200804",
               "200805", "200806", "200807", "200808",
               "200809"          , "200811", "200812",
               "200901", "200902", "200903", "200904",
               "200905", "200906", "200907", "200908",
               "200909", "200910", "200911", "200912"  ]
themis_folders = ["/home/jedmond/Documents/ML_Research/THEMIS_Data/" + elem + \
                  "/input_files" for elem in subfolders]
instruments_to_use = [ "tha", "thb", "thc", "thd", "the" ]
vars_to_make_log = [ "B", "V", "temp", "density" ]


# For KMeans classification
SOM_num_iters = 1000
node_length = 25
min_clusters, max_clusters = 3, 9
downsample_resolution = 0.3   # Points within earth-radii of this amount are "same" in x,y,z
ignore_cols_for_training = ["time","X","Y","Z","R",
                            "B_X", "B_Y", "B_Z",
                            "V_X", "V_Y", "V_Z"]
vars_to_plot = [ ["X","Y"], ["X","Z"], ["time","R"],
                 ["time","B"], ["time","V"], ["time","density"],
                 ["time", "temp"] ]
vars_for_downsample = ["X", "Y", "Z"]
vars_for_movie = [ ["X","Y","Z"] ]
vars_for_heatmap = ["B", "temp", "density", "V_X", "V_Y", "V_Z", "B_X", "B_Y", "B_Z", "V"]
#extra_angles_for_3D = [  ["same","+45"], ["same","+90"], ["+15","same"], ["-15","same"]  ]

# ===========================









# ========== GATHERING THEMIS DATA ==========

# Check to see if csv_drop_folder contains any files; if it has none, then
# data will have to be re-processed
if not os.path.exists(csv_drop_folder): os.makedirs(csv_drop_folder)

# check if any csvs found in csv_drop_folder ...
csv_names = [elem + ".csv" for elem in instruments_to_use]
csvs_found = False
for possible_csv in csv_names:
    csvs_found = possible_csv in os.listdir(csv_drop_folder)
    if csvs_found: break

# ... and if so, just read those in 
if not csvs_found:
    data_manager = SSI_Data_Manager(original_data_folders=themis_folders,
                                    instruments=instruments_to_use)
    data_manager.save_data_to_csv(csv_drop_folder)
    

# Once CSV's exist, then just read those in
data_manager_from_csv = \
    SSI_Data_Manager(prepared_data_folder=csv_drop_folder,
                     instruments=instruments_to_use)

# Save plots of originally-processed data ...
data_manager_from_csv.convert_time_to("days")
fig_drop_folder_untrimmed = fig_drop_folder + "/untrimmed"
data_manager_from_csv.plot_data(fig_drop_folder_untrimmed,plot_duration="month")
 
   
# Convert time to days and convert some vars to log
#data_manager_from_csv.trim_data("B",max_val=150)
data_manager_from_csv.make_log(vars_to_make_log)



# ... Then trim data and plot what's left
data_manager_from_csv.trim_data("planar_angle",min_val=90,max_val=270,remove_interval=True)
fig_drop_folder_trimmed = fig_drop_folder + "/trimmed"
data_manager_from_csv.plot_data(fig_drop_folder_trimmed,plot_duration="month",
                                add_to_figname="_TRIMMED")

# Finally, get data out as single Pandas dataframe
themis_df = data_manager_from_csv.get_data(single_frame=True,add_instrument_col=True)

formal_var_names_dict = data_manager_from_csv.get_names_dict()

# ===========================================







# ========== RUNNING KMEANS ==========

# convert any lists containing variable names into lists containing their proper names
ignore_cols_for_training = replace_list_elements_with_values_from_dict(
                                            ignore_cols_for_training,
                                            formal_var_names_dict)
vars_to_plot = replace_list_elements_with_values_from_dict(vars_to_plot,
                                                           formal_var_names_dict)
vars_for_movie = replace_list_elements_with_values_from_dict(vars_for_movie,
                                                           formal_var_names_dict)
vars_for_downsample = replace_list_elements_with_values_from_dict(vars_for_downsample,
                                                           formal_var_names_dict)
vars_for_heatmap = replace_list_elements_with_values_from_dict(vars_for_heatmap,
                                                           formal_var_names_dict)




# additionally, if instrument column was added to themis_df, let's ignore that
# for training
if "instrument" in list(themis_df): ignore_cols_for_training.append( "instrument" )






# run kmeans
ML_obj = ML_Classifier(themis_df, main_output_folder)
ML_obj.run_SOM(node_length,plot_inertia=True,num_iters=SOM_num_iters)
ML_obj.plot_kmeans_inertia_vs_num_clusters(max_clusters,
                                ignore_cols=ignore_cols_for_training)


clusters_to_try = np.arange(min_clusters,max_clusters+1)
for i in clusters_to_try:
    ML_obj.run_kmeans(i,cols_to_ignore=ignore_cols_for_training)
    ML_obj.reduce_data_by_cluster(vars_for_downsample,downsample_resolution)
    ML_obj.plot_cluster_hists(log_y_axis=True)
    ML_obj.plot_cluster_distances()
    ML_obj.plot_all_clusters_on_vars(vars_to_plot)
    #                    extra_angles=extra_angles_for_3D)
    ML_obj.plot_separate_clusters_on_vars(vars_to_plot)
    #                    extra_angles=extra_angles_for_3D)
    
    # Heatmap var movies
    for var in vars_for_heatmap:
        ML_obj.make_movie_from_rotated_plots(vars_for_movie,rotate_horizontal=True,
                                      rotate_vertical=True,plot_per_degrees=5,
                                      framerate=10,
                                      separate_clusters=True,
                                      use_reduced_data=True,
                                      heatmap_var=var)
    
    # Movies of just cluster positions
    ML_obj.make_movie_from_rotated_plots(vars_for_movie,rotate_horizontal=True,
                                      rotate_vertical=True,plot_per_degrees=5,
                                      framerate=10,all_clusters=True,
                                      separate_clusters=True,
                                      use_reduced_data=True)
                                      #style="scatter")
    if i < max(clusters_to_try): print("")



#print( data_manager.start_date )
#print( data_manager.get_data(single_frame=True,add_instrument_col=True) )











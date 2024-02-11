'''
Module Contents

This module contains user-defined constants and functions to be used as part of the NCP Case Comparison Tool.
This module can be separated into 6 distinct sections:

	1. Define Constants - Metric Dictionaries
		- USER DEFINED constants in dictionary form, indicating which metrics to search and retrieve

	2. Define Functions - Data Extraction from Defined NCP Files
		- Functions that only require an NCP case directory argument and are hard-coded to pull specific values from specific files

	3. Define Functions - Data Extraction from User Provided NCP Files
		- Functions that extract specific values from a file path argument (file needs to contain the specific values in question. ex/ hourly values)

	4. Define Functions - Data Extraction from Metric Dictionary NCP Files
		- Functions that extract all metrics defined in the user defined metric dictionaries from an NCP case directory argument

	5. Define Functions - Metric Transformation and Aggregation
		- USER DEFINED metric transformations and aggregations to perform on the metrics DataFrame supplied as an argument

	6. Define Functions - Full Metric Development Process
		- Functions that combine the functions above to fully extract and develop the user defined metrics from an NCP case directory provided as an argument

The 6 distinct sections are explained in more detail below:

1. Define Constants - Metric Dictionaries

					  CSV_METRICS_DICT: A dictionary containing the names of the metrics to be extracted from each NCP case as keys 
										and a dictionary containing the "CSV File Name" and "Conversion Factor" as the corresponding items.
										Ex. {
												'Hydro Plant   | Turbined Outflow (m³/s)': {
													'CSV File Name': 'turbincp.csv',    # m³/s
													'Conversion Factor': None           # m³/s
												},
												'Hydro Plant   | Spilled Outflow (m³/s)': {
													'CSV File Name': 'vertimcp.csv',    # hm³
													'Conversion Factor': 1000000/3600   # m³/s
												}
											}	
										
					  DAT_METRICS_DICT: A dictionary containing the names of the metrics to be extracted from each NCP case as keys 
										and a dictionary containing the "DAT File Name" and "Conversion Factor" as the corresponding items.
										Ex. {
												'System        | Opportunity Export Price (USD $/MWh)': {
													'DAT File Name': 'cpdepr01.dat',    # USD $/MWh
													'Conversion Factor': None           # USD $/MWh
												},
												'System        | Import Price (USD $/MWh)': {
													'DAT File Name': 'cpfcst01.dat',    # USD $/MWh
													'Conversion Factor': None           # USD $/MWh
												}
											}
							
2. Define Functions - Data Extraction from Defined NCP Files

			  extract_run_parameters(): A function that extracts the start date and run duration from dadger.grf of the NCP case directory provided.
										Requirements - directory (str)
											 Outputs - start_date (datetime.datetime),
													   duration_in_hours (int)
										Dependencies - None
											Warnings - None
										
					extract_run_time(): A function that extracts the model run time from cpmodel.log of the NCP case directory provided.
										Requirements - directory (str)
											Outputs  - final_time (datetime.timedelta)
										Dependencies - None
											Warnings - None
										
				extract_convergences(): A function the extracts the average and worst convergence interval gaps from ncpconv.csv of the NCP case directory provided.
										Requirements - directory (str)
											 Outputs - average_convergence (float),
													   highest_convergence (float)
										Dependencies - None
											Warnings - None
										
				  extract_financials(): A function that extracts the revenue, costs, violations and net revenues from ncpcope.csv of the NCP case directory provided.
										Requirements - directory (str)
											 Outputs - revenue (float),
										 			   costs (float),
													   violations (float),
													   net_revenue (float)
										Dependencies - None
											Warnings - None
										
	  extract_hydro_plant_parameters(): A function that extracts the hydro plant parameters and penstock coefficients from chidro01.dat and penstock.dat of the NCP case directory provided.
										Requirements - directory (str)
											 Outputs - hydro_plant_parameters_df (pd.DataFrame with Plant Number, Plant Name, Penstock Max Flow (m³/s) and Penstock Alpha as the columns for each of the hydro plants)
										Dependencies - None
											Warnings - None

3. Define Functions - Data Extraction from User Provided NCP Files

			   extract_hourly_inputs(): A function that extracts hourly inputs from the user specified DAT file of the NCP case directory provided.
										Requirements - directory (str),
													   file_name (str),
													   start_date (datetime.datetime),
													   duration_in_hours (int)
											 Outputs - hourly_inputs_df (pd.DataFrame with Hour as the index and the Agent Names as the columns)
										Dependencies - start_date (from extract_run_parameters()),
													   duration_in_hours (from extract_run_parameters())
											Warnings - None

4. Define Functions - Data Extraction from Metric Dictionary NCP Files

				 extract_csv_metrics(): A function that extracts the metrics from the file names specified in CSV_METRICS_DICT of the NCP case directory provided.
										Requirements - directory (str),
													   case_name (str),
													   csv_metrics_dict (dict)
											 Outputs - metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
										Dependencies - CSV_METRICS_DICT
											Warnings - At least one metric file specified in CSV_METRICS_DICT must contain data
										
				 extract_dat_metrics(): A function that extracts the metrics from the file names specified in DAT_METRICS_DICT of the NCP case directory provided.
										Requirements - directory (str),
													   case_name (str),
													   dat_metrics_dict (dict)
											 Outputs - metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
										Dependencies - DAT_METRICS_DICT,
													   extract_hourly_inputs()
											Warnings - At least one metric file specified in DAT_METRICS_DICT must contain data

5. Define Functions - Metric Transformation and Aggregation

				 update_flow_metrics(): A function that updates the DC Link Flow metrics to incorporate DC Link Loss from the NCP case metrics provided.
										Requirements - case_name (str),
										 			   metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
											 Outputs - metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
										Dependencies - metrics_df (from extract_csv_metrics() and/or extract_dat_metrics())
											Warnings - None
										
				create_hydro_metrics(): A function that creates new hydro metrics including total inflows, total outflows and net head from the NCP case metrics provided.
										Requirements - directory (str),
										 			   case_name (str),
													   metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
											 Outputs - metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
										Dependencies - metrics_df (from extract_csv_metrics() and/or extract_dat_metrics()),
													   extract_hydro_plant_parameters()
											Warnings - None

		   create_aggregated_metrics(): A function that creates new aggregated metrics from the NCP case metrics provided.
										Requirements - case_name (str),
													   metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
											 Outputs - metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
										Dependencies - metrics_df (from extract_csv_metrics() and/or extract_dat_metrics())
											Warnings - None
	
			  create_summary_metrics(): A function that creates summary metrics from the NCP case metrics provided.
										Requirements - directory (str),
													   case_name (str),
													   metrics_df (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
											 Outputs - summary_df (pd.DataFrame with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels)
										Dependencies - metrics_df (from extract_csv_metrics() and/or extract_dat_metrics()),
													   extract_run_time(),
													   extract_convergences(),
													   extract_financials()
											Warnings - None
		
6. Define Functions - Full Metric Development Process

						select_cases(): A function that prompts the user to select the NCP cases they wish to compare
										Requirements - None
											 Outputs - case_directory_dict (dict)
										Dependencies - None
											Warnings - None

				develop_case_metrics(): A function that performs the full process of collecting and developing the metrics specified in CSV_METRICS_DICT and DAT_METRICS_DICT of the NCP case directory provided
										and appending the metric and summary results to their respective master DataFrames.
										Requirements - directory (str),
													   case_name (str),
													   csv_metrics_dict (dict),
													   dat_metrics_dict (dict),
													   master_metrics_dict (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels),
													   master_summary_dict (pd.DataFrame with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels)
											 Outputs - master_metrics_dict (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels),
													   master_summary_dict (pd.DataFrame with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels)
										Dependencies - extract_csv_metrics(),
													   extract_dat_metrics(),
													   update_flow_metrics(),
													   create_hydro_metrics(),
													   create_aggregated_metrics(),
													   create_summary_metrics()
											Warnings - None

					generate_filters(): A function that generates lists of filters for all cases, metrics, and agents from the unique multiindex column level names of the master metric DataFrame.
										Requirements - master_metrics_dict (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels)
											 Outputs - case_list (list),
													   metric_list (list),
													   metric_agent_df (pd.DataFrame with Metric Name and Agent as the columns for all unique metric-agent combinations in master_metrics_df)
										Dependencies - master_metrics_df (from develop_case_metrics())
											Warnings - None

		   retrieve_case_metric_data(): Run the entire case metric collection process by running the select_cases(), develop_case_metrics(), and generate_filters() functions
		   								Requirements - None
										     Outputs - master_metrics_dict (pd.DataFrame with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels),
													   master_summary_dict (pd.DataFrame with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels)
													   case_list (list),
													   metric_list (list),
													   metric_agent_df (pd.DataFrame with Metric Name and Agent as the columns for all unique metric-agent combinations in master_metrics_df)
										Dependencies - extract_csv_metrics(),
													   extract_dat_metrics(),
													   update_flow_metrics(),
													   create_hydro_metrics(),
													   create_aggregated_metrics(),
													   create_summary_metrics()
											Warnings - None

'''

import tkinter as tk
import tkfilebrowser
import pandas as pd
import numpy as np
import linecache
import datetime
import pathlib
import time
import os
import re

#--------------------------------------------------------------------------------------------------#
# 1. Define Constants - Metric Dictionaries
#--------------------------------------------------------------------------------------------------#

# NOTE: At least one metric specified in the dictionary below must exist in each case file
CSV_METRICS_DICT = {
	'Hydro Plant   | Local Inflow (m³/s)': {
		'CSV File Name': 'inflow.csv',    # m³/s
		'Conversion Factor': None         # m³/s
	},
	'Hydro Plant   | Received Outflow (m³/s)': {
		'CSV File Name': 'outtotcp.csv',  # m³/s
		'Conversion Factor': None         # m³/s
	},
	'Hydro Plant   | Turbined Outflow (m³/s)': {
		'CSV File Name': 'turbincp.csv',  # m³/s
		'Conversion Factor': None         # m³/s
	},
	'Hydro Plant   | Spilled Outflow (m³/s)': {
		'CSV File Name': 'vertimcp.csv',  # hm³
		'Conversion Factor': 1000000/3600 # m³/s
	},
	'Hydro Plant   | Forebay Elevation (masl)': {
		'CSV File Name': 'cotrescp.csv',  # masl
		'Conversion Factor': None         # masl
	},
	'Hydro Plant   | Tailwater Elevation (masl)': {
		'CSV File Name': 'cotjuscp.csv',  # masl
		'Conversion Factor': None         # masl
	},
	'Hydro Plant   | Generation (MW)': {
		'CSV File Name': 'gerhidcp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Hydro Plant   | Generation Constraint Violation (MW)': {
		'CSV File Name': 'violrgcp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Hydro Unit    | Turbined Outflow (m³/s)': {
		'CSV File Name': 'turbiucp.csv',  # m³/s
		'Conversion Factor': None         # m³/s
	},
	'Hydro Unit    | Generation (MW)': {
		'CSV File Name': 'gerunicp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Thermal Plant | Generation (MW)': {
		'CSV File Name': 'gertercp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Renewable     | Generation (MW)': {
		'CSV File Name': 'gergndcp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Battery       | Generation (MW)': {
		'CSV File Name': 'gerbatcp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Battery       | Energy Storage (MWh)': {
		'CSV File Name': 'batstgcp.csv',  # MWh
		'Conversion Factor': None         # MWh
	},
	'Transmission  | Bus Deficit (MW)': {
		'CSV File Name': 'defbuscp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Transmission  | Bus Demand (MW)': {
		'CSV File Name': 'demxba.csv',    # MW
		'Conversion Factor': None         # MW
	},
	'Transmission  | DC Link Loss (MW)': {
		'CSV File Name': 'lossescp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Transmission  | DC Link Flow (MW)': {
		'CSV File Name': 'cirflwcp.csv',  # MW
		'Conversion Factor': None         # MW
	},
	'Transmission  | DC Link Loading (%)': {
		'CSV File Name': 'usecircp.csv',  # %
		'Conversion Factor': None         # %
	},
	'System        | Opportunity Export (MW)': {
		'CSV File Name': 'demelecp.csv',  # MW
		'Conversion Factor': None         # MW
	},
}

# NOTE: At least one metric specified in the dictionary below must exist in each case file
DAT_METRICS_DICT = {
	'System        | Opportunity Export Price (USD $/MWh)': {
		'DAT File Name': 'cpdepr01.dat',  # USD $/MWh
		'Conversion Factor': None         # USD $/MWh
	},
	'System        | Import Price (USD $/MWh)': {
		'DAT File Name': 'cpfcst01.dat',  # USD $/MWh
		'Conversion Factor': None         # USD $/MWh
	}
}

#--------------------------------------------------------------------------------------------------#
# 2. Define Functions - Data Extraction from Defined NCP Files
#--------------------------------------------------------------------------------------------------#

def extract_run_parameters(directory: str):
	'''Extract run parameters from dadger.grf of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		
	Returns:
		start_date: The starting date of the NCP case as a datetime.datetime object
		duration_in_hours: The duration (in hours) of the NCP case as an int
	
	'''
	# Read the dadger.grf file located in the selected folder to obtain the run start year/month, duration in hours and start date
	file_path = f'{directory}\\dadger.grf'
	# Check if file exists prior to opening
	if os.path.isfile(file_path):
		duration_in_hours = int(linecache.getline(filename=file_path, lineno=6).strip()[-5:])
		start_date = datetime.datetime.strptime(linecache.getline(filename=file_path, lineno=10).strip()[-16:], '%d/%m/%Y %H:%M')
	return start_date, duration_in_hours

def extract_run_time(directory: str):
	'''Extract the run time from cpmodel.log of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		
	Returns:
		final_time: The model run time of the NCP case as a datetime.timedelta object
	
	'''
	# Read the last 20 lines of cpmodel.log line by line and search for "Total CPU Time:"
	file_path = f'{directory}\\cpmodel.log'
	# Check if file exists prior to opening
	if os.path.isfile(file_path):
		with open(file_path, 'r') as f:
			lines = f.readlines()
			final_time = None
			for line_number_from_end in range(-1, -21, -1):
				selected_line = lines[line_number_from_end]
				run_time = re.search(r'Total CPU Time: (\S*)', selected_line)
				# If a Total CPU Time was found, convert to a datetime.timedelta value and assign to final_time
				if run_time:
					run_time = run_time.group(1)
					run_time = float(run_time)
					final_time = datetime.timedelta(seconds=run_time)
					final_time = final_time - datetime.timedelta(microseconds=final_time.microseconds) # Remove microseconds by subtracting the microseconds from final_time
				# If no Total CPU Time was found, assign a datetime.timedelta value of 0 to final_time  
				if line_number_from_end == -20 and not final_time:    
					final_time = datetime.timedelta(0)
	return final_time

def extract_convergences(directory: str):
	'''Extract the convergence interval gaps from ncpconv.csv of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		
	Returns:
		average_convergence: The average convergence interval gap (%) of the NCP case as a float rounded to 3 decimals of precision
		highest_convergence: The highest convergence interval gap (%) of the NCP case as a float rounded to 3 decimals of precision
	
	'''
	# Read the ncpconv.csv file located in the selected folder to obtain the average and highest convergence interval gaps
	file_path = f'{directory}\\ncpconv.csv'
	# Check if file exists prior to opening
	if os.path.isfile(file_path):
		temp_df = pd.read_csv(file_path)
		convergence_list = temp_df.iloc[:,4].tolist() # Select only the 5th column, which contains the convergence interval gaps, and convert it into a list
		average_convergence = round(np.mean(convergence_list), 3)
		highest_convergence = round(np.max(convergence_list), 3)
	return average_convergence, highest_convergence

def extract_financials(directory: str):
	'''Extract the financial results from ncpcope.csv of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		
	Returns:
		revenue: The sum of elastic demand and revenues (USD k$) from the NCP case as a float rounded to 2 decimals of precision
		costs: The sum of costs (USD k$) from the NCP case as a float rounded to 2 decimals of precision
		violations: The sum of violations (USD k$) from the NCP case as a float rounded to 2 decimals of precision
		net_revenue: The sum of elastic demand, revenues and costs (USD k$) from the NCP case as a float rounded to 2 decimals of precision
	
	'''
	# Read the ncpcope.csv file located in the selected folder to obtain the revenue, costs, violations and net revenue
	file_path = f'{directory}\\ncpcope.csv'
	# Check if file exists prior to opening
	if os.path.isfile(file_path):
		temp_df = pd.read_csv(file_path)
		revenue = round(temp_df.loc[temp_df['Values of the Objective Function Terms'].str.contains('Elastic Demand|Revenue'), '(k$)'].sum(), 2)
		costs = round(temp_df.loc[temp_df['Values of the Objective Function Terms'].str.contains('Cost'), '(k$)'].sum(), 2)
		violations = round(temp_df.loc[temp_df['Values of the Objective Function Terms'].str.contains('Violation'), '(k$)'].sum(), 2)
		net_revenue = revenue + costs
	return revenue, costs, violations, net_revenue

def extract_hydro_plant_parameters(directory: str):
	'''Extract hydro plant parameters from chidro01.dat and penstock.dat of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		
	Returns:
		hydro_plant_parameters_df: A DataFrame with Plant Number, Plant Name, Penstock Max Flow (m³/s) and Penstock Alpha as the columns for each of the hydro plants in the NCP case provided
	
	'''
	# Create an empty placeholder DataFrame to store the plant number and plant name for each hydro plant
	hydro_plant_parameters_df = pd.DataFrame(columns=['Plant Number', 'Plant Name'])
	# Read the chidro01.dat file located in the selected folder to obtain the penstock max flow and alpha values
	file_path = f'{directory}\\chidro01.dat'
	# Check if file exists prior to opening
	if os.path.isfile(file_path):
		with open(file_path, 'r') as f:
			for line in f.readlines()[1:]:
				plant_number = int(line[0:5])
				plant_name = line[5:18].strip()
				hydro_plant_parameters_df.loc[len(hydro_plant_parameters_df.index)] = [plant_number, plant_name]
		# Read data from penstock.dat and merge with the plant_parameter_inputs_df DataFrame
		temp_df = pd.read_csv(f'{directory}\\penstock.dat', delim_whitespace=True).iloc[:, [1, 3, 4]]
		temp_df.columns = ['Plant Number', 'Penstock Max Flow (m³/s)', 'Penstock Alpha']
		hydro_plant_parameters_df = pd.merge(hydro_plant_parameters_df, temp_df, how='left', on='Plant Number')
	return hydro_plant_parameters_df

#--------------------------------------------------------------------------------------------------#
# 3. Define Functions - Data Extraction from User Provided NCP Files
#--------------------------------------------------------------------------------------------------#

def extract_hourly_inputs(directory: str, file_name: str, start_date: datetime.datetime, duration_in_hours: int):
	'''Extract inputs provided in hourly formats from the provided file name of the NCP case provided
	
	Accepted hourly format:
	**** Agent:    1   Name: MILO           Block: 01   Bus:    22   Type:    0   Equip:    1
	dd/mm/aaaa ....00h ....01h ....02h ....03h ....04h ....05h ....06h ....07h ....08h ....09h ....10h ....11h ....12h ....13h ....14h ....15h ....16h ....17h ....18h ....19h ....20h ....21h ....22h ....23h 
	01/01/2037    37.6    37.6    37.6   38.74   38.74   38.74   40.31   40.31   35.83   35.83   40.31   40.31   40.31   40.31   40.31   40.31   40.31   35.83   42.14   42.14   42.14   42.14   38.74    37.6
	02/01/2037    37.6    37.6   38.74   38.74   38.74   38.74   40.31   35.83   37.22   37.22   35.83   35.83   40.31   40.31   40.31   35.83   35.83   37.22   52.27   52.27   57.79   57.79    41.1    41.1
	...
	
	Args:
		directory: The directory of the NCP case as a string
		file_name: The name of the DAT file (including file extension) containing hourly inputs to extract as a string
		start_date: The starting date of the NCP case as a datetime.datetime object
		duration_in_hours: The duration (in hours) of the NCP case as an int
		
	Returns:
		hourly_inputs_df: A DataFrame containing the hourly values for each Agent in the DAT file, with Hour as the index and the Agent Names as the columns

	'''
	# Create an empty placeholder DataFrame to store the hourly inputs
	hourly_inputs_df = pd.DataFrame(columns=['Date', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 'Agent'])
	# Calculate the duration of the NCP case in days using the duration_in_hours
	duration_in_days = duration_in_hours // 24
	# Read the file located in the selected folder
	file_path = f'{directory}\\{file_name}'
	# Check if file exists prior to opening
	if os.path.isfile(file_path):
		# Retrieve the agent name and starting line values for each unique agent
		agent_names = {}
		with open(file_path, 'r') as f:
			for line_number, line_text in enumerate(f):
				if '**** Agent' in line_text:
					agent_name = line_text.split('Name: ')[1][:15].strip()
					agent_names[agent_name] = line_number + 2
				if '**** Fuel' in line_text:
					agent_name = line_text[18:].strip()
					agent_names[agent_name] = line_number + 2
		# For each unique agent in the agent_names dicitonary, read the hourly data into a DataFrame and append to the hourly_inputs_df
		for agent, line_start in agent_names.items():
			temp_df = pd.read_csv(file_path, skiprows=line_start, delim_whitespace=True, header=None, nrows=duration_in_days) # header=None returns a DataFrame with headers as integers [0, 1, 2, ..., 24]
			temp_df = temp_df.rename(columns={temp_df.columns[0]: 'Date'})
			temp_df['Agent'] = agent
			hourly_inputs_df = temp_df if hourly_inputs_df.empty else pd.concat([hourly_inputs_df, temp_df], ignore_index=True, sort=False)
		# Unpivot the hourly_inputs_df hour columns
		hourly_inputs_df['Date'] = pd.to_datetime(hourly_inputs_df['Date'], format='%d/%m/%Y')
		hourly_inputs_df = hourly_inputs_df.melt(id_vars=['Date', 'Agent'], var_name='Hour', value_name='Values')
		hourly_inputs_df['Date'] = (hourly_inputs_df['Date'] - start_date).dt.days # Calculate the number of days between each Date in the hourly_inputs_df and the start_date
		hourly_inputs_df['Hour'] = hourly_inputs_df['Hour'] + (hourly_inputs_df['Date'] * 24) # Convert the hours in the Hour column from hours of each day (1 to 24) to the hours of the run (1 to n)
		hourly_inputs_df = hourly_inputs_df.drop(columns='Date')
		hourly_inputs_df = hourly_inputs_df.pivot(index='Hour', columns='Agent', values='Values').rename_axis(None, axis=1).sort_index()
	return hourly_inputs_df

#--------------------------------------------------------------------------------------------------#
# 4. Define Functions - Data Extraction from Metric Dictionary NCP Files
#--------------------------------------------------------------------------------------------------#

def extract_csv_metrics(directory: str, case_name: str, csv_metrics_dict: dict):
	'''Extract metrics from the file names specified in csv_metrics_list of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		case_name: The name of the NCP case as a string
		csv_metrics_dict: A dictionary containing Metric Name as keys, and a dictionary consisting of CSV File Name and Conversion Factor as items
		
	Returns:
		csv_metrics_df: A DataFrame containing the metrics specified in csv_metrics_dict, with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
	
	'''
	print(csv_metrics_dict)
	# Create an empty placeholder DataFrame to store the csv metrics specified in csv_metrics_dict
	csv_metrics_df = pd.DataFrame()
	# Read each file specified in the csv_metrics_dict and store the results in the csv_metrics_df DataFrame
	for metric, parameters in csv_metrics_dict.items():
		file_path = f'{directory}\\{parameters["CSV File Name"]}'
		# Check if file exists prior to opening
		if os.path.isfile(file_path):
			temp_df = pd.read_csv(file_path, skiprows=3, index_col=0).iloc[:, 2:]
			agent_names = temp_df.columns.str.strip()
			if parameters['Conversion Factor'] is not None:
				temp_df = temp_df * parameters['Conversion Factor']
			temp_df.columns = pd.MultiIndex.from_product([[case_name], [metric], agent_names]) # Generate multiindex columns (Case -> Metric Name -> Agent) and assign to temp_df columns
			csv_metrics_df = pd.concat([csv_metrics_df, temp_df], axis=1)
	# Name the index
	csv_metrics_df.index.name = 'Hour'
	# Sort columns alphabetically in ascending order by Metric Name (level 1) and Agent (level 2)
	csv_metrics_df = csv_metrics_df.sort_index(axis=1, level=[1,2], ascending=[True, True])
	return csv_metrics_df

def extract_dat_metrics(directory: str, case_name: str, dat_metrics_dict: dict):
	'''Extract metrics from the file names specified in dat_metrics_list of the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		case_name: The name of the NCP case as a string
		dat_metrics_list: A dictionary containing Metric Name as keys, and a dictionary consisting of DAT File Name and Conversion Factor as items

	Returns:
		dat_metrics_df: A DataFrame containing the metrics specified in dat_metrics_dict, with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
	
	'''
	# extract_run_parameters() returns the start_date and duration_in_hours which will be used in extract_hourly_inputs() to filter out only the hourly inputs relevant to the specified NCP case
	start_date, duration_in_hours = extract_run_parameters(directory=directory)
	# Create an empty placeholder DataFrame to store the dat metrics specified in dat_metrics_dict
	dat_metrics_df = pd.DataFrame()
	# Read each file specified in the dat_metrics_dict and store the results in the dat_metrics_df DataFrame
	for metric, parameters in dat_metrics_dict.items():
		# extract_hourly_inputs() returns a DataFrame containing the hourly values for each Agent in the DAT file, with Hour as the index and the Agent Names as the columns
		temp_df = extract_hourly_inputs(directory=directory, file_name=parameters['DAT File Name'], start_date=start_date, duration_in_hours=duration_in_hours)
		agent_names = temp_df.columns.str.strip()
		if parameters['Conversion Factor'] is not None:
			temp_df = temp_df * parameters['Conversion Factor']
		temp_df.columns = pd.MultiIndex.from_product([[case_name], [metric], agent_names]) # Generate multiindex columns (Case -> Metric Name -> Agent) and assign to temp_df columns
		dat_metrics_df = pd.concat([dat_metrics_df, temp_df], axis=1)
	# Sort columns alphabetically in ascending order by Metric Name (level 1) and Agent (level 2)
	dat_metrics_df = dat_metrics_df.sort_index(axis=1, level=[1,2], ascending=[True, True])
	return dat_metrics_df

#--------------------------------------------------------------------------------------------------#
# 5. Define Functions - Metric Transformation and Aggregation
#--------------------------------------------------------------------------------------------------#

def update_flow_metrics(case_name: str, metrics_df: pd.DataFrame):
	'''Update the DC Link Flow metrics contained in metrics_df to incorporate DC Link Loss
	
	Args:
		case_name: The name of the NCP case as a string
		metrics_df: A DataFrame containing the metrics from a specific NCP case with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		
	Returns:
		metrics_df: The metrics_df DataFrame supplied as an argument with the updated DC Link Flow metrics added as newly created columns
	
	'''
	# Factor DC Link Loss into DC Link Flow metrics
	metrics_df.loc[:, (case_name, 'Transmission  | DC Link Flow (MW)', slice(None))] = np.where(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)')] > 0, metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)')].values - (metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)')].values * 0.5), metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)')].values + (metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)')].values * 0.5))
	return metrics_df

def create_hydro_metrics(directory: str, case_name: str, metrics_df: pd.DataFrame):
	'''Create additional hydro metrics from the hydro metrics contained in metrics_df and the hydro plant parameters in hydro_plant_parameters_df for the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		case_name: The name of the NCP case as a string
		metrics_df: A DataFrame containing the csv metrics from a specific NCP case with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		
	Returns:
		metrics_df: The metrics_df DataFrame supplied as an argument with the newly created hydro metrics added as newly created columns
	
	'''
	# extract_hydro_plant_parameters() returns a DataFrame containing the Plant Number, Plant Name, Penstock Max Flow (m³/s) and Penstock Alpha for each of the hydro plants in the NCP case provided
	# Penstock Alpha will be used to calculate the Net Head for plants with Penstock Alphas
	hydro_plant_parameters_df = extract_hydro_plant_parameters(directory=directory)
	
	# Create a Total Inflows metric for each plant with both local inflow and received outflow values (Local Inflow + Received Outflow)
	local_inflow_plant_list = metrics_df.loc[:,(metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Local Inflow (m³/s)')].columns.unique(level=2)
	received_inflow_plant_list = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Received Outflow (m³/s)')].columns.unique(level=2)
	inflow_plant_list = sorted(list(set(local_inflow_plant_list) | set(received_inflow_plant_list)))
	for plant in inflow_plant_list:
		metrics_df.loc[:, (case_name, 'Hydro Plant   | Total Inflows (m³/s)', plant)] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Local Inflow (m³/s)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1) + metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Received Outflow (m³/s)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1)
	
	# Create a Total Outflows metric for each plant with both turbined and spilled outflow values (Turbined Outflow + Spilled Outflow)
	turbined_outflow_plant_list = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Turbined Outflow (m³/s)')].columns.unique(level=2)
	spilled_outflow_plant_list = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Spilled Outflow (m³/s)')].columns.unique(level=2)
	outflow_plant_list = sorted(list(set(turbined_outflow_plant_list) | set(spilled_outflow_plant_list)))
	for plant in outflow_plant_list:
		metrics_df.loc[:, (case_name, 'Hydro Plant   | Total Outflows (m³/s)', plant)] =  metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Turbined Outflow (m³/s)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1) + metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Spilled Outflow (m³/s)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1)
	
	# Create a Net Head metric for each plant with both forebay and tailwater elevation values (Forebay Elevation - Tailwater Elevation)
	# For plants that have a Penstock Alpha value in the hydro_plant_parameters_df DataFrame (Forebay Elevation - Tailwater Elevation - ((Turbined Outflow)^2 * Penstock Alpha)))
	forebay_elevation_plant_list =  metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Forebay Elevation (masl)')].columns.unique(level=2)
	tailwater_elevation_plant_list =  metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Tailwater Elevation (masl)')].columns.unique(level=2)
	elevation_plant_list = sorted(list(set(forebay_elevation_plant_list) & set(tailwater_elevation_plant_list)))
	for plant in elevation_plant_list:
		if plant in hydro_plant_parameters_df.loc[hydro_plant_parameters_df['Penstock Alpha'] > 0, 'Plant Name'].unique():
			metrics_df.loc[:, (case_name, 'Hydro Plant   | Net Head (m)', plant)] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Forebay Elevation (masl)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1) - metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Tailwater Elevation (masl)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1) - (( metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Turbined Outflow (m³/s)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1) ** 2) * hydro_plant_parameters_df.loc[hydro_plant_parameters_df['Plant Name'] == plant, 'Penstock Alpha'].values[0])
		else:
			metrics_df.loc[:, (case_name, 'Hydro Plant   | Net Head (m)', plant)] =  metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Forebay Elevation (masl)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1) - metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Tailwater Elevation (masl)') & (metrics_df.columns.get_level_values(2) == plant)].sum(axis=1)
	return metrics_df

def create_aggregated_metrics(case_name: str, metrics_df: pd.DataFrame):
	'''Create additional aggregated metrics from the metrics contained in metrics_df
	
	Args:
		case_name: The name of the NCP case as a string
		metrics_df: A DataFrame containing the metrics from a specific NCP case with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		
	Returns:
		metrics_df: The metrics_df DataFrame supplied as an argument with the newly created aggregated metrics added as newly created columns
	
	'''
	# Create a Sum of Imports @ Market Source metric (Sum of all Thermal Plant Generation metrics with Agent Names ending with "_dummy")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Imports @ Market Source (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Thermal Plant | Generation (MW)') & (metrics_df.columns.get_level_values(2).str.endswith(('_dummy')))].sum(axis=1)
	# Create a Sum of Imports @ Common Bus metric (Absolute sum of ONLY negative DC Link Flow metrics with Agent Names starting with 'MILO', 'Ent', 'Bisk')
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Imports @ Common Bus (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))].where(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))] < 0).sum(axis=1).abs()
	# Create a Sum of Opportunity Exports @ Border metric (Sum of all Opportunity Export metrics)
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Opportunity Exports @ Border (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'System        | Opportunity Export (MW)')].sum(axis=1)
	# Create a Sum of Opportunity Exports @ Common Bus metric (Sum of ONLY positive DC Link Flow metrics with Agent Names starting with "MILO")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Opportunity Exports @ Common Bus (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('MILO')))].where(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('MILO')))] > 0).sum(axis=1)
	# Create a Sum of Exports @ Common Bus metric (Sum of ONLY positive DC Link Flow metrics with Agent Names starting with 'MILO', 'Ent', 'Bisk')
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Exports @ Common Bus (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))].where(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))] > 0).sum(axis=1)
	
	# Create a Sum of Provincial Generation @ Generation metric (Sum of all Generation metrics, except for those with Agent Names ending with "_dummy")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Provincial Generation @ Generation (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1).str.endswith('Generation (MW)')) & (metrics_df.columns.get_level_values(1).str.startswith(('Hydro Plant', 'Thermal Plant', 'Renewable', 'Battery'))) & ~(metrics_df.columns.get_level_values(2).str.endswith(('_dummy')))].sum(axis=1)
	# Create a Sum of Provincial Generation @ Common Bus metric (Sum of all Generation metrics, except for those with Agent Names ending with "_dummy" subtract sum of all DC Link Loss metrics, except for those with Agent Names starting with 'MILO', 'Ent', 'Bisk')
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Provincial Generation @ Common Bus (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1).str.endswith('Generation (MW)')) & (metrics_df.columns.get_level_values(1).str.startswith(('Hydro Plant', 'Thermal Plant', 'Renewable', 'Battery'))) & ~(metrics_df.columns.get_level_values(2).str.endswith(('_dummy')))].sum(axis=1) - metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)') & ~(metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))].sum(axis=1)
	# Create a Sum of Hydro Generation @ Generation metric (Sum of all Hydro Plant Generation metrics)
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Hydro Generation @ Generation (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Generation (MW)')].sum(axis=1)
	# Create a Sum of Hydro Generation @ Common Bus metric (Sum of all Hydro Plant Generation metrics subtract sum of all DC Link Loss metrics, except for those with Agent Names starting with 'MILO', 'Ent', 'Bisk')
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Hydro Generation @ Common Bus (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Generation (MW)')].sum(axis=1) - metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)') & ~(metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))].sum(axis=1)
	# Create a Sum of Thermal Generation @ Generation metric (Sum of all Thermal Plant Generation metrics, except for those with Agent Names ending with "_dummy")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Thermal Generation @ Generation (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Thermal Plant | Generation (MW)') & ~(metrics_df.columns.get_level_values(2).str.endswith(('_dummy')))].sum(axis=1)
	# Create a Sum of Renewable Generation @ Generation metric (Sum of all Renewable Generation metrics)
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Renewable Generation @ Generation (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Renewable     | Generation (MW)')].sum(axis=1)

	# Create a Sum of TL1 DC Link Flow metric (Sum of all DC Link Flow metrics with Agent Names starting with "TL1")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of TL1 DC Link Flow (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('TL1')))].sum(axis=1)
	# Create a Sum of TL1 DC Link Loss metric (Sum of all DC Link Loss metrics with Agent Names starting with "TL1")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of TL1 DC Link Loss (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('TL1')))].sum(axis=1)
	# Create a Sum of TL2 DC Link Flow metric (Sum of all DC Link Flow metrics with Agent Names starting with "TL2")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of TL2 DC Link Flow (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('TL2')))].sum(axis=1)
	# Create a Sum of TL2 DC Link Loss metric (Sum of all DC Link Loss metrics with Agent Names starting with "TL2")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of TL2 DC Link Loss (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('TL2')))].sum(axis=1)
	# Create a Sum of TL3 DC Link Flow metric (Sum of all DC Link Flow metrics with Agent Names starting with "TL3")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of TL3 DC Link Flow (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Flow (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('TL3')))].sum(axis=1)
	# Create a Sum of TL3 DC Link Loss metric (Sum of all DC Link Loss metrics with Agent Names starting with "TL3")
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of TL3 DC Link Loss (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)') & (metrics_df.columns.get_level_values(2).str.startswith(('TL3')))].sum(axis=1)
	
	# Create a Sum of Load metric (Sum of all Bus Demand metrics)
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Load (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | Bus Demand (MW)')].sum(axis=1)
	# Create a Sum of Load and Loss metric (Sum of all Bus Demand metrics and DC Link Loss metrics except for those with Agent Names starting with 'MILO', 'Ent', 'Bisk')
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Load and Loss (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | Bus Demand (MW)')].sum(axis=1) + metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | DC Link Loss (MW)') & ~(metrics_df.columns.get_level_values(2).str.startswith(('MILO', 'Ent', 'Bisk')))].sum(axis=1)
	# Create a Sum of Hydro Generation Constraint Violations metric (Sum of all Hydro Plant Generation Constraint metrics)
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Hydro Generation Constraint Violations (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Hydro Plant   | Generation Constraint Violation (MW)')].sum(axis=1)
	# Create a Sum of Deficits metric (Sum of all Bus Deficit metrics)
	metrics_df.loc[:, (case_name, 'Aggregated    | Sum of Deficits (MW)', '')] = metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Transmission  | Bus Deficit (MW)')].sum(axis=1)
	return metrics_df

def create_summary_metrics(directory: str, case_name: str, metrics_df: pd.DataFrame):
	'''Create summary metrics from the metrics contained in metrics_df and additional files found in the NCP case provided
	
	Args:
		directory: The directory of the NCP case as a string
		case_name: The name of the NCP case as a string
		metrics_df: A DataFrame containing the metrics from a specific NCP case with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		
	Returns:
		summary_df: A DataFrame containing summary metrics from the NCP case provided, with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels
	
	'''
	# extract_run_time() returns the model run time of the NCP case provided as a datetime.timedelta object
	run_time = extract_run_time(directory=directory)
	# extract_convergences() returns the average and worst convergence interval gaps (%) of the NCP case provided as floats rounded to 3 decimals of precision
	average_convergence, worst_convergence = extract_convergences(directory=directory)
	# extract_financials() returns the revenue, costs, violations and net revenues (USD k$) from the NCP case provided as floats rounded to 2 decimals of precision
	revenue, costs, violations, net_revenue = extract_financials(directory=directory)
	# Assign the summary variables created above to summary_df and create additional summary metrics by summing columns from metrics_df
	summary_df = pd.DataFrame({('', 'Case Name'): 							case_name, 
							   ('Overview', 'Run Time'): 					str(run_time), 
							   ('Convergence (%)', 'Average'): 				round(average_convergence, 2),
							   ('Convergence (%)', 'Worst'): 				round(worst_convergence, 2),
							   ('Sum of Generation (MW)', 'Provincial'): 	round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Provincial Generation @ Common Bus (MW)')].sum().values[0], 0),
							   ('Sum of Generation (MW)', 'Hydro'): 		round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Hydro Generation @ Common Bus (MW)')].sum().values[0], 0),
							   ('Sum of Generation (MW)', 'Thermal'): 		round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Thermal Generation @ Generation (MW)')].sum().values[0], 0),
							   ('Sum of Generation (MW)', 'Renewables'): 	round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Renewable Generation @ Generation (MW)')].sum().values[0], 0),
							   ('Sum of Load (MW)', 'Demand'): 				round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Load (MW)')].sum().values[0], 0),
							   ('Sum of Load (MW)', 'Deficits'): 			round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Deficits (MW)')].sum().values[0] * -1, 0),
							   ('Sum of Exchange (MW)', 'Imports'): 		round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Imports @ Common Bus (MW)')].sum().values[0], 0),
							   ('Sum of Exchange (MW)', 'Opp. Exports'): 	round(metrics_df.loc[:, (metrics_df.columns.get_level_values(1) == 'Aggregated    | Sum of Opportunity Exports @ Common Bus (MW)')].sum().values[0], 0),
							   ('Financials (USD k$)', 'Violations'): 		round(violations, 0),
							   ('Financials (USD k$)', 'Revenues'): 		round(revenue, 0),
							   ('Financials (USD k$)', 'Costs'): 			round(costs, 0),
							   ('Financials (USD k$)', 'Net Revenue'): 		round(net_revenue, 0)}, 
							   index=[0])
	summary_df = summary_df.set_index(('', 'Case Name'))
	summary_df.index.name = None
	return summary_df

#--------------------------------------------------------------------------------------------------#
# 6. Define Functions - Full Metric Development Process
#--------------------------------------------------------------------------------------------------#

def select_cases():
	'''Prompt the user to select the NCP cases they wish to compare
	
	Args:
		None
		
	Returns:
		case_directory_dict: A dictionary of all selected NCP case names as keys and their respective directories as items
	
	'''
	# Tkinter not working properly when deployed via Streamlit, therefore, automatically select the following cases
	case_directory_dict = {fr'Case 1\Jan\1941': os.path.join(os.path.dirname(os.path.dirname(__file__)), fr'NCP Cases\Case 1\Jan\1941'),
				fr'Case 2\Jan\1941': os.path.join(os.path.dirname(os.path.dirname(__file__)), fr'NCP Cases\Case 2\Jan\1941'),
				fr'Case 3\Jan\1941': os.path.join(os.path.dirname(os.path.dirname(__file__)), fr'NCP Cases\Case 3\Jan\1941'),
				fr'Case 4\Jan\1941': os.path.join(os.path.dirname(os.path.dirname(__file__)), fr'NCP Cases\Case 4\Jan\1941'),
				fr'Case 5\Jan\1941': os.path.join(os.path.dirname(os.path.dirname(__file__)), fr'NCP Cases\Case 5\Jan\1941')}
	return case_directory_dict
	
	# #-----------------------User Select Network Location----------------------#
	# # List of available network locations
	# network_list = [fr'C:\Users\Dallas\Documents\Streamlit Project']

	# # Initialize the tk.TK() interpreter and create the root window
	# root = tk.Tk()
	# root.title('Network Selection')
	# root.attributes('-topmost', True)

	# def save_network_location_button():
	# 	global selected_network
	# 	selected_network = var.get()
	# 	root.quit()
	# 	root.destroy()
	
	# # Setup frame and style
	# frame = tk.Frame(root, bg='#FFFFFF', highlightbackground='#000000', highlightthickness=1)
	# frame.grid(row=0, column=0, padx=5, pady=5, sticky='n')
	# tk.Label(frame, text='Network Location', bg='#0079C1', fg='white', width=50).pack(expand=True, fill='x', anchor='n')

	# # Add a list of available network locations for the user to choose from
	# var = tk.StringVar(root, network_list[0])
	# for network_location in network_list:
	# 	tk.Radiobutton(frame, text=network_location, value=network_location, variable=var, bg='white').pack(fill='x', anchor='nw')
	
	# # Add a Save button at the bottom of the tkinter window
	# frame_button = tk.Frame(root, bg='#FFFFFF', highlightbackground='#000000', highlightthickness=1)
	# frame_button.grid(row=1, column=0, columnspan=1, sticky=tk.W+tk.E, padx=5, pady=5)
	# btn = tk.Button(frame_button, text='Save', command=save_network_location_button, bg='#0079C1', fg='white')
	# btn.pack(fill='x')

	# # Remain in this loop until root.quit() is called
	# root.mainloop()

	# # Check that selected_network contains a value before continuing
	# if 'selected_network' not in globals():
	# 	return None

	# #----------------------------User Select Cases----------------------------#
	# # Initialize the tk.TK() interpreter and create the root window
	# root = tk.Tk()
	# root.attributes('-topmost', True)
	# selected_folders_tuple = tkfilebrowser.askopendirnames(parent=root, title='Select NCP cases to compare', initialdir=selected_network, foldercreation=False)
	# selected_folders_list = list(selected_folders_tuple)
	# # Close the instance of tk.TK() called root
	# root.quit()                                             
	# root.destroy()
	# time.sleep(1)
	
	# # Create an empty placeholder dictionary to store the case name and directory for all valid selected NCP cases from the selected_folders_list
	# case_directory_dict = {}
	# # Create an empty placeholder list to store lists of subfolder directory names at their respective subfolder level for all valid NCP cases in the selected_folders_list
	# subfolder_level_name_lists = []

	# # Loop through each directory in the selected_folders_list and populate the placeholder lists above
	# for directory in selected_folders_list:
	# 	# In each directory, loop through all subfolders
	# 	for dirpath, dirnames, filenames in os.walk(directory):
	# 		# In each bottom-most subfolder, if an instance of ncpcope.csv or ncpconv.csv are found, then that subfolder is considered a valid NCP case
	# 		if not dirnames and ('ncpcope.csv' in filenames or 'ncpconv.csv' in filenames):
	# 			# If any instances of .tmp files exist in the valid NCP case folder, then that case is considered to be running and will be ignored
	# 			if len(list(pathlib.Path(dirpath).glob('*.tmp'))) < 1:
	# 				# Extract the case name from dirpath
	# 				case_parent_directory = directory.rsplit('\\', 1)[0]
	# 				case_name = dirpath.rsplit(case_parent_directory)[-1]
	# 				# Add a new item to the case_directory_dict with case_name as the key and dirpath as the item directory
	# 				case_directory_dict[case_name] = dirpath
	# 				# Extract a list of subfolder names from the case_name
	# 				case_subfolder_list = case_name.split('\\')
	# 				case_subfolder_list = [x for x in case_subfolder_list if x != '']
	# 				# Loop through each subfolder name in the case_subfolder_list
	# 				for subfolder_level, subfolder_name in enumerate(case_subfolder_list):
	# 					# If the subfolder_level doesn't have a corresponding subfolder level name list in subfolder_level_name_lists, append an empty list in subfolder_level_name_lists
	# 					if len(subfolder_level_name_lists) <= subfolder_level:
	# 						subfolder_level_name_lists.append([])
	# 					# Append the subfolder_name into the corresponding subfolder level name list in subfolder_level_name_lists
	# 					subfolder_level_name_lists[subfolder_level].append(subfolder_name)

	# # Count the number of subfolder levels in subfolder_level_name_lists
	# subfolder_level_count = len(subfolder_level_name_lists)	
	# # Remove all duplicate names from each subfolder level name list in subfolder_level_name_lists
	# for level in range(subfolder_level_count):
	# 	subfolder_level_name_lists[level] = sorted(list(set(subfolder_level_name_lists[level])))
	# # Create an empty placeholder list to store lists of tk.StringVar() objects corresponding to the subfolder level names in subfolder_level_name_lists
	# subfolder_level_var_lists = [[] for i in range(subfolder_level_count)]
	
	# # Check that case_directory_dict contains values before continuing
	# if not case_directory_dict:
	# 	return None
	
	# #--------------------------User Select Subfolders-------------------------#
	# # Initialize the tk.TK() interpreter and create the root window
	# root = tk.Tk()
	# root.title('Subfolder Selection')
	# root.attributes('-topmost', True)
	# save_flag = False

	# def save_subfolders_button():
	# 	nonlocal save_flag
	# 	save_flag = True
	# 	for subfolder_level, subfolder_var_list in enumerate(subfolder_level_var_lists):
	# 		# Overwrite each subfolder level name list in subfolder_level_name_lists with only the checked names from subfolder_level_var_lists
	# 		subfolder_level_name_lists[subfolder_level] = [var.get() for var in subfolder_var_list if var.get()]
	# 	root.quit()
	# 	root.destroy()
	
	# # Add columns to the tkinter window representing each subfolder level in subfolder_level_name_lists
	# for subfolder_level, subfolder_level_list in enumerate(subfolder_level_name_lists):
	# 	frame = tk.Frame(root, bg='#FFFFFF', highlightbackground='#000000', highlightthickness=1)
	# 	frame.grid(row=0, column=subfolder_level, padx=5, pady=5, sticky='n')
	# 	tk.Label(frame, text=f'Subfolder Level {subfolder_level}', bg='#0079C1', fg='white', width=50).pack(expand=True, fill='x', anchor='n')
	# 	# Populate each column with it's corresponding subfolder level names
	# 	for subfolder_name in subfolder_level_list:
	# 		var = tk.StringVar(value=subfolder_name)
	# 		level = tk.Checkbutton(frame, text=subfolder_name, variable=var, onvalue=subfolder_name, offvalue=None, bg='white')
	# 		level.select()
	# 		level.pack(anchor='nw')
	# 		# Append all subfolder level vars to the corresponding subfolder level list in subfolder_level_var_lists
	# 		subfolder_level_var_lists[subfolder_level].append(var)
	
	# # Add a Save button at the bottom of the tkinter window
	# frame_button = tk.Frame(root, bg='#FFFFFF', highlightbackground='#000000', highlightthickness=1)
	# frame_button.grid(row=1, column=0, columnspan=subfolder_level_count, sticky=tk.W+tk.E, padx=5, pady=5)
	# btn = tk.Button(frame_button, text='Save', command=save_subfolders_button, bg='#0079C1', fg='white')
	# btn.pack(fill='x')

	# # Remain in this loop until root.quit() is called
	# root.mainloop()

	# # Check that the Save button was clicked before continuing
	# if not save_flag:
	# 	return None

	# #--------------------Generate Case Directory Dictionary-------------------#
	# # Create an empty placeholder list to store the case_names to be removed from case_directory_dict
	# cases_to_be_removed = []
	# # Loop through each NCP case in case_directory_dict again
	# for case_name, directory in case_directory_dict.items():
	# 	# Extract a list of subfolder names from the case_name
	# 	case_subfolder_list = case_name.split('\\')
	# 	case_subfolder_list = [x for x in case_subfolder_list if x != '']
	# 	# Loop through each subfolder name in the case_subfolder_list
	# 	for subfolder_level, subfolder_name in enumerate(case_subfolder_list):
	# 		# If a subfolder_name can't be found in it's corresponding subfolder level name list, add the case_name to the cases_to_be_removed list
	# 		if subfolder_name not in subfolder_level_name_lists[subfolder_level]:
	# 			cases_to_be_removed.append(case_name)
	# 			break
	# # Remove cases from case_directory_dict
	# for case_name in cases_to_be_removed:
	# 	del case_directory_dict[case_name]
	# return case_directory_dict

def develop_case_metrics(directory: str, case_name: str, csv_metrics_dict: dict, dat_metrics_dict: dict, master_metrics_df: pd.DataFrame, master_summary_df: pd.DataFrame):
	'''From the single NCP case provided, generate metrics_df and summary_df DataFrames from the 2 provided metric lists and append these DataFrames to the master_metrics_df and master_summary_df DataFrames
	
	Args:
		directory: The directory of the NCP case as a string
		case_name: The name of the NCP case as a string
		csv_metrics_dict: A dictionary containing Metric Name as keys, and a dictionary consisting of CSV File Name and Conversion Factor as items
		dat_metrics_list: A dictionary containing Metric Name as keys, and a dictionary consisting of DAT File Name and Conversion Factor as items
		master_metrics_df: A DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		master_summary_df: A DataFrame containing summary metrics from all selected NCP cases with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels
		
	Returns:
		master_metrics_df: A DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		master_summary_df: A DataFrame containing summary metrics from all selected NCP cases with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels
	
	'''
	# Call the user-created functions above to generate metrics_df and summary_df for the specified NCP case
	csv_metrics_df = extract_csv_metrics(directory=directory, case_name=case_name, csv_metrics_dict=csv_metrics_dict)
	dat_metrics_df = extract_dat_metrics(directory=directory, case_name=case_name, dat_metrics_dict=dat_metrics_dict)
	metrics_df = pd.concat([csv_metrics_df, dat_metrics_df], axis=1)
	metrics_df = update_flow_metrics(case_name=case_name, metrics_df=metrics_df)
	metrics_df = create_hydro_metrics(directory=directory, case_name=case_name, metrics_df=metrics_df)
	metrics_df = create_aggregated_metrics(case_name=case_name, metrics_df=metrics_df)
	summary_df = create_summary_metrics(directory=directory, case_name=case_name, metrics_df=metrics_df)
	# Merge metrics_df and summary_metrics_df (case specific DataFrames) with master_metrics_df and master_summary_df (all cases included DataFrames) respectively
	if master_metrics_df.empty:
		master_metrics_df = metrics_df.copy()
		master_summary_df = summary_df.copy()
	else:
		master_metrics_df = master_metrics_df.merge(metrics_df, left_index=True, right_index=True, how='outer')
		master_summary_df = pd.concat([master_summary_df, summary_df])
	return master_metrics_df, master_summary_df

def generate_filters(master_metrics_df: pd.DataFrame):
	'''Generate filter lists from the unique multiindex column level names supplied in the master_metrics_df DataFrame
	
	Args:
		master_metrics_df: A DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		
	Returns:
		case_list: A sorted list of all unique cases found in the Case multiindex column level (0) in master_metrics_df
		metric_list: A sorted list of all unique metrics found in the Metric Name multiindex column levels (1) in master_metrics_df
		metric_agent_df: A DataFrame with Metric Name and Agent as the columns for all unique combinations in master_metrics_df (used to update the agent_list when the corresponding metric multiselect widget is modified)

	'''
	case_array = sorted(master_metrics_df.loc[:, (slice(None), slice(None), slice(None))].columns.unique(level=0))
	metric_array = sorted(master_metrics_df.loc[:, (slice(None), slice(None), slice(None))].columns.unique(level=1))
	metric_agent_df	= master_metrics_df.columns.to_frame(index=False)
	metric_agent_df.columns	= ['Case Name', 'Metric Name', 'Agent']
	metric_agent_df	= metric_agent_df[['Metric Name', 'Agent']].drop_duplicates()
	return case_array, metric_array, metric_agent_df

def retrieve_case_metric_data():
	'''Run the entire case metric collection process by running the select_cases(), develop_case_metrics(), and generate_filters() functions

	Args:
		None

	Returns:
		master_metrics_df: A DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		master_summary_df: A DataFrame containing summary metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		case_list: A sorted list of all unique cases found in the Case multiindex column level (0) in master_metrics_df
		metric_list: A sorted list of all unique metrics found in the Metric Name multiindex column level (1) in master_metrics_df
		metric_agent_df: A DataFrame containing all unique Metric Name to Agent combinations (used to update the agent_list when the corresponding metric multiselect widget is modified) in master_metrics_df
	
	'''
	# Call the select_cases function
	case_directory_dict = select_cases()
	# Create empty placeholder DataFrames to store metric and summary values from all valid selected NCP cases
	master_metrics_df = pd.DataFrame()
	master_summary_df = pd.DataFrame()
	# Loop through each NCP case in case_directory_dict
	for case_name, directory in case_directory_dict.items():
		master_metrics_df, master_summary_df = develop_case_metrics(directory=directory, case_name=case_name, csv_metrics_dict=CSV_METRICS_DICT, dat_metrics_dict=DAT_METRICS_DICT, master_metrics_df=master_metrics_df, master_summary_df=master_summary_df)
		case_list, metric_list, metric_agent_df = generate_filters(master_metrics_df=master_metrics_df)
	return master_metrics_df, master_summary_df, case_list, metric_list, metric_agent_df

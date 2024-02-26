import streamlit as st
import pandas as pd
import numpy as np
import math
import os

from plotly.subplots import make_subplots
import plotly.express as px

from data.data_module import select_cases, develop_case_metrics, generate_filters, CSV_METRICS_DICT, DAT_METRICS_DICT

#--------------------------------------------------------------------------------------------------#
# 1. Streamlit Web App - Base Configuration
#--------------------------------------------------------------------------------------------------#

# NOTE: Anytime a widget on the Streamlit interface is interacted with, Streamlit will rerun the code below
if __name__ == '__main__':

	# Change the directory to the directory containing the Home script
	cwd = os.path.dirname(os.path.dirname(__file__))
	os.chdir(cwd)

	# Configure the default settings of the Streamlit web page
	st.set_page_config(page_title='NCP Case Comparison Tool', page_icon='💼', layout='wide',)

	# Read and apply CSS styling from styles.css
	with open(fr'{cwd}/css/styles.css') as f:
		css_styles = f.read()
	st.markdown(f'<style>{css_styles}/style>', unsafe_allow_html=True)

	# Initialize Session State variables to store variables across mulitple pages and app reruns
	if 'metrics_df' not in st.session_state:
		st.session_state.metrics_df = pd.DataFrame()
		st.session_state.summary_df = pd.DataFrame()
		st.session_state.case_list = list()
		st.session_state.metric_list = list()
		st.session_state.metric_agent_df = list()
		st.session_state.data_progress = 0
		st.session_state.data_progress_text = 'No Data Loaded'

	# Instantiate a sidebar navigation menu
	with st.sidebar:
		data_progress_bar = st.progress(st.session_state.data_progress, text=st.session_state.data_progress_text)
		fetch_data_button = st.button(label='Fetch Case Data', key='fetch-data-btn', use_container_width=True)
		if fetch_data_button:
			try:
				# Reset case_directory_dict
				case_directory_dict = None
				# Call the select_cases function
				case_directory_dict = select_cases()
				# Check if case_directory_dict is populated before continuing
				if not case_directory_dict:
					raise ValueError('No case directories selected')
				# Clear Session State variables back to their original empty state
				st.session_state.metrics_df = pd.DataFrame()
				st.session_state.summary_df = pd.DataFrame()
				st.session_state.case_list = list()
				st.session_state.metric_list = list()
				st.session_state.metric_agent_df = pd.DataFrame()
				# Clear page specific Case Selection Session State variables
				if 'line_chart_case_selection' in st.session_state:
					st.session_state.line_chart_case_selection = None
				if 'multi_line_chart_case_selection' in st.session_state:
					st.session_state.multi_line_chart_case_selection = None
				if 'bar_chart_case_selection' in st.session_state:
					st.session_state.bar_chart_case_selection = None
				if 'duration_curve_case_selection' in st.session_state:
					st.session_state.duration_curve_case_selection = None
				# Count number of cases in case_directory_dict to use when updating the data_progress_bar
				case_count = len(case_directory_dict)
				progress_interval = 0
				progress_interval_size = 1 / case_count
				# Loop through each NCP case in case_directory_dict
				for case_name, directory in case_directory_dict.items():
					# Update the data_progress_bar
					st.session_state.data_progress = progress_interval
					st.session_state.data_progress_text = f'Fetching Data: {round(progress_interval * 100, 0)}% Complete'
					data_progress_bar.progress(st.session_state.data_progress, text=st.session_state.data_progress_text)
					progress_interval += progress_interval_size
					# Retrieve the case data
					st.session_state.metrics_df, st.session_state.summary_df = develop_case_metrics(directory, case_name, CSV_METRICS_DICT, DAT_METRICS_DICT, st.session_state.metrics_df, st.session_state.summary_df)
					st.session_state.case_list, st.session_state.metric_list, st.session_state.metric_agent_df = generate_filters(st.session_state.metrics_df)
				# Update the data_progress_bar upon completion
				st.session_state.data_progress = 100
				st.session_state.data_progress_text = 'Data Loaded'
				data_progress_bar.progress(st.session_state.data_progress, text=st.session_state.data_progress_text)
			except ValueError:
				pass

#--------------------------------------------------------------------------------------------------#
# Define Functions - Page Specific
#--------------------------------------------------------------------------------------------------#

	def update_agent_multiselect(self_multiselect_key: str, agent_multiselect_key: str, agent_session_state_key: str, agent_options_session_state_key: str):
		'''A function to be called whenever a Metric Multiselect widget is modified that will udpate the specified Agent Multiselect widget selections and available Agent options

		Args:
			self_multiselect_key: The key of the Multiselect widget calling this function as a string
			agent_multiselect_key: The key of the Agent Multiselect widget to be updated as a string
			agent_session_state_key: The key of the Agent select Session State to be updated as a string
			agent_options_session_state_key: The name of the Session State variable containing the list of available Agent options as a string

		Returns:
			None

		'''
		# Assign the selections in the self_multiselect_key to itself unless itself is None, in which case assign an empty list
		st.session_state[self_multiselect_key] = st.session_state[self_multiselect_key] if st.session_state[self_multiselect_key] else []
		# Generate a list of available Agent names based on the selected Metrics and assign to the Agent options Session State variable
		st.session_state[agent_options_session_state_key] = sorted(st.session_state.metric_agent_df.loc[st.session_state.metric_agent_df['Metric Name'].isin(st.session_state[self_multiselect_key]), 'Agent'].unique())
		# Update the list of available Agent names by removing Agents with '' as a name (Corresponding to Aggregated metrics)
		if '' in st.session_state[agent_options_session_state_key]:                           
			st.session_state[agent_options_session_state_key].remove('')
		# Update the selected Agents in the Agent Multiselect widget if an Agent is no longer associated with any of the selected Metrics in the Metric Multiselect widget
		selected_agents_list = []
		for agent in st.session_state[agent_multiselect_key]:
			if agent in st.session_state[agent_options_session_state_key]:
				selected_agents_list.append(agent)
		st.session_state[agent_multiselect_key] = selected_agents_list
		st.session_state[agent_session_state_key] = selected_agents_list
	

	def filter_metrics_df(case_multiselect_key: str, case_session_state_key: str, 
					   	  metric_multiselect_key: str, metric_session_state_key: str, 
						  agent_multiselect_key: str, agent_session_state_key: str,
						  filtered_metrics_df_session_state_key: str):
		'''A function to be called when the Apply Filters button is pressed that will update the Session State selection variables and filter and update the filtered_metrics_df Session State variables

		Args:
			case_multiselect_key: The key of the Case Multiselect widget as a string
			case_session_state_key: The key of the Case select Session State to be updated as a string
			metric_multiselect_key: The key of the Metric Multiselect widget as a string
			metric_session_state_key: The key of the Metric select Session State to be updated as a string
			agent_multiselect_key: The key of the Metric Multiselect widget as a string
			agent_session_state_key: The key of the Agent select Session State to be updated as a string
			filtered_metrics_df_session_state_key: The key of the metrics_df Session State variable to be updated as a string

		Returns:
			None

		'''
		# Update the Session State selection variables
		st.session_state[case_session_state_key] = st.session_state[case_multiselect_key]
		st.session_state[metric_session_state_key] = st.session_state[metric_multiselect_key]
		st.session_state[agent_session_state_key] = st.session_state[agent_multiselect_key]
		# Using the Multiselect filters above, filter st.session_state.metrics_df and assign to the filtered_metrics_df Session State variables
		# NOTE: st.session_state.metrics_df is a DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		st.session_state[filtered_metrics_df_session_state_key] = st.session_state.metrics_df.loc[:, (st.session_state.metrics_df.columns.get_level_values(0).isin(st.session_state[case_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(1).isin(st.session_state[metric_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(2).isin(st.session_state[agent_multiselect_key] + ['']))]
		# Sort column order in the order of Case Alias - Metric Names - Agent Names selected
		st.session_state[filtered_metrics_df_session_state_key] = st.session_state[filtered_metrics_df_session_state_key].reindex(level=2, columns=st.session_state[agent_multiselect_key] + [''])
		st.session_state[filtered_metrics_df_session_state_key] = st.session_state[filtered_metrics_df_session_state_key].reindex(level=1, columns=st.session_state[metric_multiselect_key])
		st.session_state[filtered_metrics_df_session_state_key] = st.session_state[filtered_metrics_df_session_state_key].reindex(level=0, columns=st.session_state[case_multiselect_key])
		

	def define_initial_y_axis_bounds(min_value: float, max_value: float):
		'''Create an initial set of Y-Axis bounds based on the minimum and maximum values to be plotted
		This function ensures that the primary and secondary Y-Axis gridlines line up with each other
		
		Args:
			min_value: The minimum value to be plotted as a float
			max_value: The maximum value to be plotted as a float
			
		Returns:
			lower_bound: The Y-Axis lower bound as a float
			upper_bound: The Y-Axis upper bound as a float
		
		'''
		# Check if minimum and maximum values exist
		if pd.isna(min_value) or pd.isna(max_value):
			# Return default values if either value doesn't exist
			lower_bound = 0
			upper_bound = 1
		# Check if minimum and maximum values are equal
		elif min_value == max_value:
			lower_bound = min_value - 1
			upper_bound = max_value + 1
		else:
			# Calculate the difference between the minimum and maximum values provided
			difference = max_value - min_value
			# Convert the difference into scientific notation (sn), get the exponent (ex) and convert to base-10 (b10)
			difference_sn = f'{difference: .2E}'
			difference_ex = int(difference_sn.split('E')[-1])
			difference_b10 = 10 ** difference_ex
			# Calculate the upper and lower bounds by rounding the maximum (up) and minimum (down) values based on the difference base-10 value
			lower_bound = math.floor(min_value / difference_b10) * difference_b10
			upper_bound = math.ceil(max_value / difference_b10) * difference_b10
		return lower_bound, upper_bound


	def create_single_metric_duration_curve(df: pd.DataFrame):
		'''Create a plotly line chart containing all NCP cases from the data supplied in the primary and secondary axis DataFrames
		
		Args:
			df: A DataFrame containing the metrics to plot with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
			
		Returns:
			None
		
		'''
		# Return a copy of df with the Case multiindex column level converted into a separate column called "Case Name" and the Metric Name and Agent multiindex column levels flattened with the format "Metric Name [Agent]" (or only "Metric Name" for columns with no Agent)
		case_name_index = df.columns.get_level_values(0).unique()
		df = df.stack(level=0) # Move the Case (0) multiindex column level into the index (shifting Metric Name and Agent multiindex column levels from 1 -> 0 and 2 -> 1 respectivley)
		df = df.reindex(level=1, index=case_name_index) # Reindex the Case Name index (level 1) to match the order before performing stack()
		df.index.names = ['Hour', 'Case Name']
		df = df.reset_index(level='Case Name')
		df.columns = df.columns.map(lambda x: (f'{x[0]} [{x[1]}]') if x[1] != '' else (x[0])) # Metric Name (1), Agent (0)
		metric_list = list(df.loc[:, df.columns != 'Case Name'].columns)
		case_name_list = list(case_name_index)

		# For each Metric-Agent combination provided in df, create a new subplot with Metric-Agent as the title
		for metric_name in metric_list:
			master_fig = make_subplots(subplot_titles=[f'<b>{metric_name}</b>'])
			metric_df = df[['Case Name', metric_name]].copy()
			metric_df = metric_df.reset_index().pivot(index='Hour', columns='Case Name', values=metric_name) # Pivot metric_df to get Hour as index and Case Names as columns
			metric_df = metric_df.reindex(columns=case_name_index) # Reindex the Case Name index to match the order before performing pivot()
			# Create an empty DataFrame with the various quantiles we would like to calculate and then populate using data from metric_df
			quantile_df = pd.DataFrame(index=list(np.arange(0, 1.01, 0.01))).rename_axis(index='Quantiles')
			for case_name in case_name_list:
				quantile_df[case_name] = metric_df[case_name].quantile(list(np.arange(0, 1.01, 0.01)))
			quantile_df.index = quantile_df.index * 100
			quantile_df.index = quantile_df.index.astype(int)
			quantile_df_fig = px.line(data_frame=quantile_df,
									x=quantile_df.index,
									y=quantile_df.columns)
			master_fig.add_traces(quantile_df_fig.data)
			master_fig.update_layout(yaxis=dict(title_text=metric_name))
			master_fig.update_layout(autosize=True, xaxis_title='Quantiles', colorway=px.colors.qualitative.Light24)
			
			# Add a divider to separate the filters from the chart
			st.divider()
			# Update the margins
			master_fig.update_layout(margin={'l':20, 'r':20, 't':20, 'b':0})
			# Create an st.empty() placeholder object to display our line chart (Redraw the chart in place anytime the X or Y-Axis bounds are updated)
			redraw_chart = st.empty()

			# Calculate the Y-Axis upper and lower bounds
			y_max = quantile_df.max().max()
			y_min = quantile_df.min().min()
			y_lower_bound, y_upper_bound = define_initial_y_axis_bounds(min_value=y_min, max_value=y_max)
			y_tick_size = (y_upper_bound - y_lower_bound) / 10
			# Calculate the X-Axis upper and lower bounds
			x_lower_bound = quantile_df.index.min()
			x_upper_bound = quantile_df.index.max()

			# Split the lower portion of the page into 3 columns, with column 1 containing download chart data button, column 2 containing lower bound X-Axis adjustment input, and column 3 containing upper bound X-Axis adjustment input
			axis_expander_column, _, download_chart_data_column = st.columns((3, 4, 3))
			with axis_expander_column:
				with st.expander('Update Axis Bounds'):
					axis_label_column, axis_lower_bounds_column, axis_upper_bounds_column = st.columns((2.5, 1.25, 1.25))
					with axis_label_column:
						st.text_input('Y-Axis Label', value='Y-Axis', key=f'y_label_text_input_{metric_name}', disabled=True, label_visibility='collapsed')
						st.text_input('X-Axis Label', value='X-Axis', key=f'x_label_text_input_{metric_name}', disabled=True, label_visibility='collapsed')
					with axis_lower_bounds_column:
						y_lower_bound_input = st.number_input(label='Lower Bound', value=y_lower_bound, key=f'y_lower_bound_number_input_{metric_name}', label_visibility='collapsed')
						x_lower_bound_input = st.number_input(label='Lower Bound', value=x_lower_bound, key=f'x_lower_bound_number_input_{metric_name}', label_visibility='collapsed')
					with axis_upper_bounds_column:
						y_upper_bound_input = st.number_input(label='Upper Bound', value=y_upper_bound, key=f'y_upper_bound_number_input_{metric_name}', label_visibility='collapsed')
						x_upper_bound_input = st.number_input(label='Upper Bound', value=x_upper_bound, key=f'x_upper_bound_number_input_{metric_name}', label_visibility='collapsed')

			# Recalculate the Y-Axes tick sizes and update the master_fig layout
			y_tick_size = (y_upper_bound_input - y_lower_bound_input) / 10
			master_fig.update_layout(yaxis={'range': [y_lower_bound_input, y_upper_bound_input], 'tick0': y_lower_bound_input, 'dtick': y_tick_size},
									xaxis={'range': [x_lower_bound_input, x_upper_bound_input]})
			# Redraw the chart
			redraw_chart.write(master_fig)
			
			with download_chart_data_column:
				chart_output_df = quantile_df.copy()
				# chart_output_df = chart_output_df.set_index('Case Name', append=True).unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)
				chart_output = chart_output_df.to_csv()
				st.download_button(label='Download Chart Data', data=chart_output, use_container_width=True, file_name=f'{metric_name}_duration_curve_data.csv', key=f'duration_curve_download_button_{metric_name}')

#--------------------------------------------------------------------------------------------------#
# 3. Streamlit Web App - Page Specific Configuration
#--------------------------------------------------------------------------------------------------#
	
	# Initialize Session State variables to store variables across mulitple pages and app reruns
	if 'duration_curve_case_selection' not in st.session_state:
		st.session_state.duration_curve_case_selection = None
		st.session_state.duration_curve_metric_selection = None
		st.session_state.duration_curve_agent_options = []
		st.session_state.duration_curve_agent_selection = None
		st.session_state.duration_curve_filtered_metrics_df = pd.DataFrame()
		
	# Instantiate a container object
	with st.container():
		# If Case Data is loaded
		if st.session_state.data_progress == 100:
			# Case Select - Create a Multiselect widget to select cases to plot
			selected_case = st.multiselect('Case Select', 
								  		options=st.session_state.case_list, 
										default=st.session_state.duration_curve_case_selection,
										placeholder='Select a Case',
										key='duration_curve_case_multiselect')

			# Metric Select - Create a Multiselect widget to select metrics to plot
			metric_column, _ = st.columns(2)
			with metric_column:
				selected_metric = st.multiselect('Metric Select', 
											   	options=st.session_state.metric_list, 
												default=st.session_state.duration_curve_metric_selection,
												placeholder='Select a Metric',
												key='duration_curve_metric_multiselect', 
												on_change=update_agent_multiselect, 
												kwargs=dict(self_multiselect_key='duration_curve_metric_multiselect',
															agent_multiselect_key='duration_curve_agent_multiselect',
															agent_session_state_key='duration_curve_agent_selection',
															agent_options_session_state_key='duration_curve_agent_options'))

			# Agent Select - Create a Multiselect widget to select agents to plot
			agent_column, _ = st.columns(2)
			with agent_column:
				selected_agent = st.multiselect('Agent Select', 
											  	options=st.session_state.duration_curve_agent_options, 
												default=st.session_state.duration_curve_agent_selection,
												placeholder='Select an Agent',
												key='duration_curve_agent_multiselect')
				
			# Apply Filters - Create a button to apply the above filters
			apply_filters_button = st.button('Apply Filters', 
											 use_container_width=True,
											 key='duration_curve_apply_filters',
											 on_click=filter_metrics_df,
											 kwargs=dict(case_multiselect_key='duration_curve_case_multiselect',
														case_session_state_key='duration_curve_case_selection',
														metric_multiselect_key='duration_curve_metric_multiselect',
														metric_session_state_key='duration_curve_metric_selection',
														agent_multiselect_key='duration_curve_agent_multiselect',
														agent_session_state_key='duration_curve_agent_selection',
														filtered_metrics_df_session_state_key='duration_curve_filtered_metrics_df'))
			
			# Line Chart - Create a line chart when the filtered_metrics_df has been udpated (via Apply Filters button)
			if not st.session_state.duration_curve_filtered_metrics_df.empty:
				create_single_metric_duration_curve(df=st.session_state.duration_curve_filtered_metrics_df)

		# If not Case Data is loaded
		else:
			# Create a markdown text explaining that no Data has been loaded
			st.markdown('''<b><font size="5">NO DATA LOADED</font></b>
			  			</br>
						To get NCP case data, first open the navigation menu on the left and press the <b>Fetch Case Data</b> button. Follow the instructions provided and select the NCP cases you wish to review and analyze:<br>
						<ol>
			  				<li>Select one of the 4 available network drive locations to retrieve NCP model results from</li>
			  				<li>Select the folder containing the NCP model results you would like to review and analyze</li>
			  				<li>Filter out any NCP model result folders that are not required save</li>
			  			</ol>
						''',
						unsafe_allow_html=True)
			

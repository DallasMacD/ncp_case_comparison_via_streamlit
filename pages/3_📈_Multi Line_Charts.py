import streamlit as st
import pandas as pd
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
# 2. Define Functions - Page Specific
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
					   	  primary_metric_multiselect_key: str, primary_metric_session_state_key: str, 
						  primary_agent_multiselect_key: str, primary_agent_session_state_key: str, 
						  secondary_metric_multiselect_key: str, secondary_metric_session_state_key: str, 
						  secondary_agent_multiselect_key: str, secondary_agent_session_state_key: str, 
						  primary_filtered_metrics_df_session_state_key: str, secondary_filtered_metrics_df_session_state_key: str):
		'''A function to be called when the Apply Filters button is pressed that will update the Session State selection variables and filter and update the primary and secondary filtered_metrics_df Session State variables

		Args:
			case_multiselect_key: The key of the Case Multiselect widget as a string
			case_session_state_key: The key of the Case select Session State to be updated as a string
			primary_metric_multiselect_key: The key of the Primary Metric Multiselect widget as a string
			primary_metric_session_state_key: The key of the Primary Metric select Session State to be updated as a string
			primary_agent_multiselect_key: The key of the Primary Metric Multiselect widget as a string
			primary_agent_session_state_key: The key of the Primary Agent select Session State to be updated as a string
			secondary_metric_multiselect_key: The key of the Secondary Metric Multiselect widget as a string
			secondary_metric_session_state_key: The key of the Secondary Metric select Session State to be updated as a string
			secondary_agent_multiselect_key: The key of the Secondary Metric Multiselect widget as a string
			secondary_agent_session_state_key: The key of the Secondary Agent select Session State to be updated as a string
			primary_filtered_metrics_df_session_state_key: The key of the Primary metrics_df Session State variable to be updated as a string
			secondary_filtered_metrics_df_session_state_key: The key of the Secondary metrics_df Session State variable to be updated as a string

		Returns:
			None

		'''
		# Update the Session State selection variables
		st.session_state[case_session_state_key] = st.session_state[case_multiselect_key]
		st.session_state[primary_metric_session_state_key] = st.session_state[primary_metric_multiselect_key]
		st.session_state[primary_agent_session_state_key] = st.session_state[primary_agent_multiselect_key]
		st.session_state[secondary_metric_session_state_key] = st.session_state[secondary_metric_multiselect_key]
		st.session_state[secondary_agent_session_state_key] = st.session_state[secondary_agent_multiselect_key]
		# Using the Multiselect filters above, filter st.session_state.metrics_df and assign to the Primary and Secondary metrics_df Session State variables
		# NOTE: st.session_state.metrics_df is a DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		st.session_state[primary_filtered_metrics_df_session_state_key] = st.session_state.metrics_df.loc[:, (st.session_state.metrics_df.columns.get_level_values(0).isin(st.session_state[case_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(1).isin(st.session_state[primary_metric_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(2).isin(st.session_state[primary_agent_multiselect_key] + ['']))]
		st.session_state[secondary_filtered_metrics_df_session_state_key] = st.session_state.metrics_df.loc[:, (st.session_state.metrics_df.columns.get_level_values(0).isin(st.session_state[case_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(1).isin(st.session_state[secondary_metric_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(2).isin(st.session_state[secondary_agent_multiselect_key] + ['']))]
		# Sort column order in the order of Case Alias - Metric Names - Agent Names selected
		st.session_state[primary_filtered_metrics_df_session_state_key] = st.session_state[primary_filtered_metrics_df_session_state_key].reindex(level=2, columns=st.session_state[primary_agent_multiselect_key] + [''])
		st.session_state[primary_filtered_metrics_df_session_state_key] = st.session_state[primary_filtered_metrics_df_session_state_key].reindex(level=1, columns=st.session_state[primary_metric_multiselect_key])
		st.session_state[primary_filtered_metrics_df_session_state_key] = st.session_state[primary_filtered_metrics_df_session_state_key].reindex(level=0, columns=st.session_state[case_multiselect_key])
		st.session_state[secondary_filtered_metrics_df_session_state_key] = st.session_state[secondary_filtered_metrics_df_session_state_key].reindex(level=2, columns=st.session_state[secondary_agent_multiselect_key] + [''])
		st.session_state[secondary_filtered_metrics_df_session_state_key] = st.session_state[secondary_filtered_metrics_df_session_state_key].reindex(level=1, columns=st.session_state[secondary_metric_multiselect_key])
		st.session_state[secondary_filtered_metrics_df_session_state_key] = st.session_state[secondary_filtered_metrics_df_session_state_key].reindex(level=0, columns=st.session_state[case_multiselect_key])


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


	def create_multi_case_line_chart(primary_axis_df: pd.DataFrame, secondary_axis_df: pd.DataFrame, unique_id: int):
		'''Create a plotly line chart containing all NCP cases from the data supplied in the primary and secondary axis DataFrames
		
		Args:
			primary_axis_df: A DataFrame containing the metrics to plot on the primary Y-Axis with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
			secondary_axis_df: A DataFrame containing the metrics to plot on the secondary Y-Axis with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
			unique_id: A unique ID to append onto widget keys to prevent repetition as an int
			
		Returns:
			None
		
		'''
		# Create boolean variables to indicate the empty status of each axis DataFrame
		primary_axis_empty   = False
		secondary_axis_empty = False
		# If primary_axis_df or secondary_axis_df are empty, set their respective empty flag to True
		# Else, return a copy of each respective DataFrame with the Case multiindex column level converted into a separate column called "Case Name"
		# 		and the Metric Name and Agent multiindex column levels flattened with the format "Metric Name [Agent]" (or only "Metric Name" for columns with no Agent)
		if primary_axis_df.empty:
			primary_axis_empty = True
		else:
			primary_axis_case_name_index = primary_axis_df.columns.get_level_values(0).unique()
			primary_axis_df = primary_axis_df.stack(level=0) # Move the Case (0) multiindex column level into the index (shifting Metric Name and Agent multiindex column levels from 1 -> 0 and 2 -> 1 respectivley)
			primary_axis_df = primary_axis_df.reindex(level=1, index=primary_axis_case_name_index) # Reindex the Case Name index (level 1) to match the order before performing stack()
			primary_axis_df.index.names = ['Hour', 'Case Name']
			primary_axis_df = primary_axis_df.reset_index(level='Case Name')
			primary_axis_df.columns = primary_axis_df.columns.map(lambda x: (f'{x[0]} [{x[1]}]') if x[1] != '' else (x[0])) # Metric Name (1), Agent (0)
			case_list = list(primary_axis_df['Case Name'].unique())
		if secondary_axis_df.empty:
			secondary_axis_empty = True
		else:
			secondary_axis_case_name_index = secondary_axis_df.columns.get_level_values(0).unique()
			secondary_axis_df = secondary_axis_df.stack(level=0) # Move the Case (0) multiindex column level into the index (shifting Metric Name and Agent multiindex column levels from 1 -> 0 and 2 -> 1 respectivley)
			secondary_axis_df = secondary_axis_df.reindex(level=1, index=secondary_axis_case_name_index) # Reindex the Case Name index (level 1) to match the order before performing stack()
			secondary_axis_df.index.names = ['Hour', 'Case Name']
			secondary_axis_df = secondary_axis_df.reset_index(level='Case Name')
			secondary_axis_df.columns = secondary_axis_df.columns.map(lambda x: (f'{x[0]} [{x[1]}]') if x[1] != '' else (x[0])) # Metric Name (1), Agent (0)
			case_list = list(secondary_axis_df['Case Name'].unique())

		# For each Case Name provided in primary_axis_df and secondary_axis_df, add to the subplot under it's own unique legend grouping
		master_fig = make_subplots(specs=[[{'secondary_y': True}]])
		for case_name in case_list:
			if not primary_axis_empty:
				primary_axis_case_df = primary_axis_df[primary_axis_df['Case Name'] == case_name].copy()
				primary_axis_index = primary_axis_case_df.loc[:, primary_axis_case_df.columns != 'Case Name'].index
				primary_axis_columns = primary_axis_case_df.loc[:, primary_axis_case_df.columns != 'Case Name'].columns
				primary_axis_fig = px.line(data_frame=primary_axis_case_df,
										x=primary_axis_index,
										y=primary_axis_columns).update_traces(legendgroup=case_name, 
																			  legendgrouptitle_text=case_name)
				master_fig.add_traces(primary_axis_fig.data)
				master_fig.update_layout(yaxis=dict(title_text='<br>'.join(primary_axis_case_df.loc[:, primary_axis_case_df.columns != 'Case Name'])))
			if not secondary_axis_empty:
				secondary_axis_case_df = secondary_axis_df[secondary_axis_df['Case Name'] == case_name].copy()
				secondary_axis_index = secondary_axis_case_df.loc[:, secondary_axis_case_df.columns != 'Case Name'].index
				secondary_axis_columns = secondary_axis_case_df.loc[:, secondary_axis_case_df.columns != 'Case Name'].columns
				secondary_axis_fig = px.line(data_frame=secondary_axis_case_df,
											x=secondary_axis_index,
											y=secondary_axis_columns).update_traces(legendgroup=case_name, 
																					legendgrouptitle_text=case_name)
				secondary_axis_fig.update_traces(yaxis='y2')
				master_fig.add_traces(secondary_axis_fig.data)
				master_fig.update_layout(yaxis2=dict(title_text='<br>'.join(secondary_axis_case_df.loc[:, secondary_axis_case_df.columns != 'Case Name'])))
		master_fig.update_layout(autosize=True, xaxis_title='Hour', colorway=px.colors.qualitative.Light24)
		
		# Add a divider to separate the filters from the chart
		st.divider()
		# Recolor the traces to ensure that lines from the primary and secondary axis figures don't share colors
		master_fig.for_each_trace(lambda trace: trace.update(line=dict(color=trace.marker.color)))
		# Update the behavior of the grouped legend to be able to select single items
		master_fig.update_layout(legend={'groupclick': 'toggleitem'})
		# Update the margins
		master_fig.update_layout(margin={'l':20, 'r':20, 't':20, 'b':0})
		# Create an st.empty() placeholder object to display our line chart (Redraw the chart in place anytime the X or Y-Axis bounds are updated)
		redraw_chart = st.empty()

		# Combine the primary and secondary axis dataframes and remove any duplicate columns ('Hour' and 'Case Name')
		# combined_case_chart_df used to calculate the x-axis upper and lower bounds and download chart data
		if primary_axis_empty and not secondary_axis_empty:
			combined_case_chart_df = secondary_axis_df.copy()
		if not primary_axis_empty and secondary_axis_empty:
			combined_case_chart_df = primary_axis_df.copy()
		if not primary_axis_empty and not secondary_axis_empty:
			combined_case_chart_df = primary_axis_df.merge(right=secondary_axis_df, on=['Hour', 'Case Name'], how='outer')

		# Calculate the Primary Y-Axis upper and lower bounds
		y1_max = primary_axis_df.loc[:, primary_axis_df.columns != 'Case Name'].max().max()
		y1_min = primary_axis_df.loc[:, primary_axis_df.columns != 'Case Name'].min().min()
		y1_lower_bound, y1_upper_bound = define_initial_y_axis_bounds(min_value=y1_min, max_value=y1_max)
		y1_tick_size = (y1_upper_bound - y1_lower_bound) / 10
		# Calculate the Secondary Y-Axis upper and lower bounds
		y2_max = secondary_axis_df.loc[:, secondary_axis_df.columns != 'Case Name'].max().max()
		y2_min = secondary_axis_df.loc[:, secondary_axis_df.columns != 'Case Name'].min().min()
		y2_lower_bound, y2_upper_bound = define_initial_y_axis_bounds(min_value=y2_min, max_value=y2_max)
		y2_tick_size = (y2_upper_bound - y2_lower_bound) / 10
		# Calculate the X-Axis upper and lower bounds
		x1_lower_bound = combined_case_chart_df.index.min()
		x1_upper_bound = combined_case_chart_df.index.max()

		# Split the lower portion of the page into 3 columns, with column 1 containing download chart data button, column 2 containing lower bound X-Axis adjustment input, and column 3 containing upper bound X-Axis adjustment input
		axis_expander_column, _, download_chart_data_column = st.columns((3, 4, 3))
		with axis_expander_column:
			with st.expander('Update Axis Bounds'):
				axis_label_column, axis_lower_bounds_column, axis_upper_bounds_column = st.columns((2.5, 1.25, 1.25))
				with axis_label_column:
					st.text_input('Primary Y-Axis Label', value='Primary Y-Axis', key=f'y1_label_text_input_{unique_id}', disabled=True, label_visibility='collapsed')
					st.text_input('Secondary Y-Axis Label', value='Secondary Y-Axis', key=f'y2_label_text_input_{unique_id}', disabled=True, label_visibility='collapsed')
					st.text_input('X-Axis Label', value='X-Axis', key=f'x1_label_text_input_{unique_id}', disabled=True, label_visibility='collapsed')
				with axis_lower_bounds_column:
					y1_lower_bound_input = st.number_input(label='Lower Bound', disabled=primary_axis_empty, value=y1_lower_bound, key=f'y1_lower_bound_number_input_{unique_id}', label_visibility='collapsed')
					y2_lower_bound_input = st.number_input(label='Lower Bound', disabled=secondary_axis_empty, value=y2_lower_bound, key=f'y2_lower_bound_number_input_{unique_id}', label_visibility='collapsed')
					x1_lower_bound_input = st.number_input(label='Lower Bound', value=x1_lower_bound, key=f'x1_lower_bound_number_input_{unique_id}', label_visibility='collapsed')
				with axis_upper_bounds_column:
					y1_upper_bound_input = st.number_input(label='Upper Bound', disabled=primary_axis_empty, value=y1_upper_bound, key=f'y1_upper_bound_number_input_{unique_id}', label_visibility='collapsed')
					y2_upper_bound_input = st.number_input(label='Upper Bound', disabled=secondary_axis_empty, value=y2_upper_bound, key=f'y2_upper_bound_number_input_{unique_id}', label_visibility='collapsed')
					x1_upper_bound_input = st.number_input(label='Upper Bound', value=x1_upper_bound, key=f'x1_upper_bound_number_input_{unique_id}', label_visibility='collapsed')

		# Recalculate the Y-Axes tick sizes and update the master_fig layout
		y1_tick_size = (y1_upper_bound_input - y1_lower_bound_input) / 10
		y2_tick_size = (y2_upper_bound_input - y2_lower_bound_input) / 10
		master_fig.update_layout(yaxis={'range': [y1_lower_bound_input, y1_upper_bound_input], 'tick0': y1_lower_bound_input, 'dtick': y1_tick_size},
						   		yaxis2={'range': [y2_lower_bound_input, y2_upper_bound_input], 'tick0': y2_lower_bound_input, 'dtick': y2_tick_size},
								xaxis={'range': [x1_lower_bound_input, x1_upper_bound_input]})
		# Redraw the chart
		redraw_chart.write(master_fig)
		
		with download_chart_data_column:
			chart_output_df = combined_case_chart_df.copy()
			chart_output_df = chart_output_df.set_index('Case Name', append=True).unstack().swaplevel(0, 1, axis=1)
			chart_output = chart_output_df.to_csv()
			st.download_button(label='Download Chart Data', data=chart_output, use_container_width=True, file_name=f'multi_line_chart_data.csv', key=f'multi_line_chart_download_button_{unique_id}')

#--------------------------------------------------------------------------------------------------#
# 3. Streamlit Web App - Page Specific Configuration
#--------------------------------------------------------------------------------------------------#
	
	# Initialize Session State variables to store variables across mulitple pages and app reruns
	if 'multi_line_chart_case_selection' not in st.session_state:
		st.session_state.multi_line_chart_case_selection = None
		st.session_state.multi_line_chart_primary_metric_selection_1 = None
		st.session_state.multi_line_chart_primary_agent_options_1 = []
		st.session_state.multi_line_chart_primary_agent_selection_1 = None
		st.session_state.multi_line_chart_primary_filtered_metrics_df_1 = pd.DataFrame()
		st.session_state.multi_line_chart_secondary_metric_selection_1 = None
		st.session_state.multi_line_chart_secondary_agent_options_1 = []
		st.session_state.multi_line_chart_secondary_agent_selection_1 = None
		st.session_state.multi_line_chart_secondary_filtered_metrics_df_1 = pd.DataFrame()

		st.session_state.multi_line_chart_case_selection = None
		st.session_state.multi_line_chart_primary_metric_selection_2 = None
		st.session_state.multi_line_chart_primary_agent_options_2 = []
		st.session_state.multi_line_chart_primary_agent_selection_2 = None
		st.session_state.multi_line_chart_primary_filtered_metrics_df_2 = pd.DataFrame()
		st.session_state.multi_line_chart_secondary_metric_selection_2 = None
		st.session_state.multi_line_chart_secondary_agent_options_2 = []
		st.session_state.multi_line_chart_secondary_agent_selection_2 = None
		st.session_state.multi_line_chart_secondary_filtered_metrics_df_2 = pd.DataFrame()
		
	# Instantiate a container object
	with st.container():
		# If Case Data is loaded
		if st.session_state.data_progress == 100:
			# Case Select - Create a Multiselect widget to select cases to plot
			selected_case = st.multiselect('Case Select', 
								  		options=st.session_state.case_list, 
										default=st.session_state.multi_line_chart_case_selection,
										placeholder='Select a Case',
										key='multi_line_chart_case_multiselect')

			# Metric Select - Create 2 Multiselect widgets to select metrics to plot on either the primary or secondary axes
			primary_metric_column_1, secondary_metric_column_1 = st.columns(2)
			with primary_metric_column_1:
				selected_primary_metric_1 = st.multiselect('Metric Select', 
											   			options=st.session_state.metric_list, 
														default=st.session_state.multi_line_chart_primary_metric_selection_1,
														placeholder='Select a Primary Axis Metric',
														key='multi_line_chart_primary_metric_multiselect_1', 
														on_change=update_agent_multiselect, 
														kwargs=dict(self_multiselect_key='multi_line_chart_primary_metric_multiselect_1',
																	agent_multiselect_key='multi_line_chart_primary_agent_multiselect_1',
																	agent_session_state_key='multi_line_chart_primary_agent_selection_1',
																	agent_options_session_state_key='multi_line_chart_primary_agent_options_1'))
			with secondary_metric_column_1:
				selected_secondary_metric_1 = st.multiselect('Metric Select', 
											   			options=st.session_state.metric_list,
														default=st.session_state.multi_line_chart_secondary_metric_selection_1,
														placeholder='Select a Secondary Axis Metric',
														key='multi_line_chart_secondary_metric_multiselect_1',
														label_visibility='hidden',
														on_change=update_agent_multiselect, 
														kwargs=dict(self_multiselect_key='multi_line_chart_secondary_metric_multiselect_1',
																	agent_multiselect_key='multi_line_chart_secondary_agent_multiselect_1',
																	agent_session_state_key='multi_line_chart_secondary_agent_selection_1',
																	agent_options_session_state_key='multi_line_chart_secondary_agent_options_1'))

			# Agent Select - Create 2 Multiselect widgets to select agents to plot on either the primary or secondary axes
			primary_agent_column_1, secondary_agent_column_1 = st.columns(2)
			with primary_agent_column_1:
				selected_primary_agent_1 = st.multiselect('Agent Select', 
											  			options=st.session_state.multi_line_chart_primary_agent_options_1, 
														default=st.session_state.multi_line_chart_primary_agent_selection_1,
														placeholder='Select a Primary Axis Agent',
														key='multi_line_chart_primary_agent_multiselect_1')
			with secondary_agent_column_1:
				selected_secondary_agent_1 = st.multiselect('Agent Select', 
														options=st.session_state.multi_line_chart_secondary_agent_options_1, 
														default=st.session_state.multi_line_chart_secondary_agent_selection_1,
														placeholder='Select a Secondary Axis Agent',
														key='multi_line_chart_secondary_agent_multiselect_1',
														label_visibility='hidden')

			# Apply Filters - Create a button to apply the above filters
			apply_filters_button_1 = st.button('Apply Filters', 
											 use_container_width=True,
											 key='multi_line_chart_apply_filters_1',
											 on_click=filter_metrics_df,
											 kwargs=dict(case_multiselect_key='multi_line_chart_case_multiselect',
														case_session_state_key='multi_line_chart_case_selection',
														primary_metric_multiselect_key='multi_line_chart_primary_metric_multiselect_1',
														primary_metric_session_state_key='multi_line_chart_primary_metric_selection_1',
														primary_agent_multiselect_key='multi_line_chart_primary_agent_multiselect_1',
														primary_agent_session_state_key='multi_line_chart_primary_agent_selection_1',
														secondary_metric_multiselect_key='multi_line_chart_secondary_metric_multiselect_1',
														secondary_metric_session_state_key='multi_line_chart_secondary_metric_selection_1',
														secondary_agent_multiselect_key='multi_line_chart_secondary_agent_multiselect_1',
														secondary_agent_session_state_key='multi_line_chart_secondary_agent_selection_1',
														primary_filtered_metrics_df_session_state_key='multi_line_chart_primary_filtered_metrics_df_1',
														secondary_filtered_metrics_df_session_state_key='multi_line_chart_secondary_filtered_metrics_df_1'))
			
			# Line Chart - Create a line chart when either the primary or secondary filtered_metrics_df has been udpated (via Apply Filters button)
			if (not st.session_state.multi_line_chart_primary_filtered_metrics_df_1.empty or not st.session_state.multi_line_chart_secondary_filtered_metrics_df_1.empty):
				create_multi_case_line_chart(primary_axis_df=st.session_state.multi_line_chart_primary_filtered_metrics_df_1,
											secondary_axis_df=st.session_state.multi_line_chart_secondary_filtered_metrics_df_1,
											unique_id=1)
				
				# Create a checkbox to add a second multi case chart
				st.divider()
				add_second_chart_column, _ = st.columns((2, 8))
				with add_second_chart_column:
					second_chart_selection = st.checkbox('Add a 2nd Chart', key='multi_line_chart_second_chart_checkbox')

				# If "Add a 2nd Chart" is checked
				if second_chart_selection:
					# Metric Select - Create 2 Multiselect widgets to select metrics to plot on either the primary or secondary axes
					primary_metric_column_2, secondary_metric_column_2 = st.columns(2)
					with primary_metric_column_2:
						selected_primary_metric_2 = st.multiselect('Metric Select', 
																options=st.session_state.metric_list, 
																default=st.session_state.multi_line_chart_primary_metric_selection_2,
																placeholder='Select a Primary Axis Metric',
																key='multi_line_chart_primary_metric_multiselect_2', 
																on_change=update_agent_multiselect, 
																kwargs=dict(self_multiselect_key='multi_line_chart_primary_metric_multiselect_2',
																			agent_multiselect_key='multi_line_chart_primary_agent_multiselect_2',
																			agent_session_state_key='multi_line_chart_primary_agent_selection_2',
																			agent_options_session_state_key='multi_line_chart_primary_agent_options_2'))
					with secondary_metric_column_2:
						selected_secondary_metric_2 = st.multiselect('Metric Select', 
																options=st.session_state.metric_list,
																default=st.session_state.multi_line_chart_secondary_metric_selection_2,
																placeholder='Select a Secondary Axis Metric',
																key='multi_line_chart_secondary_metric_multiselect_2',
																label_visibility='hidden',
																on_change=update_agent_multiselect, 
																kwargs=dict(self_multiselect_key='multi_line_chart_secondary_metric_multiselect_2',
																			agent_multiselect_key='multi_line_chart_secondary_agent_multiselect_2',
																			agent_session_state_key='multi_line_chart_secondary_agent_selection_2',
																			agent_options_session_state_key='multi_line_chart_secondary_agent_options_2'))

					# Agent Select - Create 2 Multiselect widgets to select agents to plot on either the primary or secondary axes
					primary_agent_column_2, secondary_agent_column_2 = st.columns(2)
					with primary_agent_column_2:
						selected_primary_agent_2 = st.multiselect('Agent Select', 
																options=st.session_state.multi_line_chart_primary_agent_options_2, 
																default=st.session_state.multi_line_chart_primary_agent_selection_2,
																placeholder='Select a Primary Axis Agent',
																key='multi_line_chart_primary_agent_multiselect_2')
					with secondary_agent_column_2:
						selected_secondary_agent_2 = st.multiselect('Agent Select', 
																options=st.session_state.multi_line_chart_secondary_agent_options_2, 
																default=st.session_state.multi_line_chart_secondary_agent_selection_2,
																placeholder='Select a Secondary Axis Agent',
																key='multi_line_chart_secondary_agent_multiselect_2',
																label_visibility='hidden')

					# Apply Filters - Create a button to apply the above filters
					apply_filters_button_2 = st.button('Apply Filters', 
													use_container_width=True,
													key='multi_line_chart_apply_filters_2',
													on_click=filter_metrics_df,
													kwargs=dict(case_multiselect_key='multi_line_chart_case_multiselect',
																case_session_state_key='multi_line_chart_case_selection',
																primary_metric_multiselect_key='multi_line_chart_primary_metric_multiselect_2',
																primary_metric_session_state_key='multi_line_chart_primary_metric_selection_2',
																primary_agent_multiselect_key='multi_line_chart_primary_agent_multiselect_2',
																primary_agent_session_state_key='multi_line_chart_primary_agent_selection_2',
																secondary_metric_multiselect_key='multi_line_chart_secondary_metric_multiselect_2',
																secondary_metric_session_state_key='multi_line_chart_secondary_metric_selection_2',
																secondary_agent_multiselect_key='multi_line_chart_secondary_agent_multiselect_2',
																secondary_agent_session_state_key='multi_line_chart_secondary_agent_selection_2',
																primary_filtered_metrics_df_session_state_key='multi_line_chart_primary_filtered_metrics_df_2',
																secondary_filtered_metrics_df_session_state_key='multi_line_chart_secondary_filtered_metrics_df_2'))
					
					# Line Chart - Create a line chart when either the primary or secondary filtered_metrics_df has been udpated (via Apply Filters button)
					if (not st.session_state.multi_line_chart_primary_filtered_metrics_df_2.empty or not st.session_state.multi_line_chart_secondary_filtered_metrics_df_2.empty):
						create_multi_case_line_chart(primary_axis_df=st.session_state.multi_line_chart_primary_filtered_metrics_df_2,
													secondary_axis_df=st.session_state.multi_line_chart_secondary_filtered_metrics_df_2,
													unique_id=2)



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
			

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
					   	  bar_metric_multiselect_key: str, bar_metric_session_state_key: str, 
						  bar_agent_multiselect_key: str, bar_agent_session_state_key: str, 
						  line_metric_multiselect_key: str, line_metric_session_state_key: str, 
						  line_agent_multiselect_key: str, line_agent_session_state_key: str, 
						  bar_filtered_metrics_df_session_state_key: str, line_filtered_metrics_df_session_state_key: str):
		'''A function to be called when the Apply Filters button is pressed that will update the Session State selection variables and filter and update the primary and secondary filtered_metrics_df Session State variables

		Args:
			case_multiselect_key: The key of the Case Multiselect widget as a string
			case_session_state_key: The key of the Case select Session State to be updated as a string
			bar_metric_multiselect_key: The key of the Stacked Bar Chart Metric Multiselect widget as a string
			bar_metric_session_state_key: The key of the Stacked Bar Chart Metric select Session State to be updated as a string
			bar_agent_multiselect_key: The key of the Stacked Bar Chart Metric Multiselect widget as a string
			bar_agent_session_state_key: The key of the Stacked Bar Chart Agent select Session State to be updated as a string
			line_metric_multiselect_key: The key of the Line Chart Metric Multiselect widget as a string
			line_metric_session_state_key: The key of the Line Chart Metric select Session State to be updated as a string
			line_agent_multiselect_key: The key of the Line Chart Metric Multiselect widget as a string
			line_agent_session_state_key: The key of the Line Chart Agent select Session State to be updated as a string
			bar_filtered_metrics_df_session_state_key: The key of the Stacked Bar Chart metrics_df Session State variable to be updated as a string
			line_filtered_metrics_df_session_state_key: The key of the Line Chart metrics_df Session State variable to be updated as a string

		Returns:
			None

		'''
		# Update the Session State selection variables
		st.session_state[case_session_state_key] = st.session_state[case_multiselect_key]
		st.session_state[bar_metric_session_state_key] = st.session_state[bar_metric_multiselect_key]
		st.session_state[bar_agent_session_state_key] = st.session_state[bar_agent_multiselect_key]
		st.session_state[line_metric_session_state_key] = st.session_state[line_metric_multiselect_key]
		st.session_state[line_agent_session_state_key] = st.session_state[line_agent_multiselect_key]
		# Using the Multiselect filters above, filter st.session_state.metrics_df and assign to the Stacked Bar Chart and Line Chart metrics_df Session State variables
		# NOTE: st.session_state.metrics_df is a DataFrame containing the metrics from all selected NCP cases with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
		st.session_state[bar_filtered_metrics_df_session_state_key] = st.session_state.metrics_df.loc[:, (st.session_state.metrics_df.columns.get_level_values(0).isin(st.session_state[case_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(1).isin(st.session_state[bar_metric_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(2).isin(st.session_state[bar_agent_multiselect_key] + ['']))]
		st.session_state[line_filtered_metrics_df_session_state_key] = st.session_state.metrics_df.loc[:, (st.session_state.metrics_df.columns.get_level_values(0).isin(st.session_state[case_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(1).isin(st.session_state[line_metric_multiselect_key])) & (st.session_state.metrics_df.columns.get_level_values(2).isin(st.session_state[line_agent_multiselect_key] + ['']))]
		# Sort column order in the order of Metric Names selected
		st.session_state[bar_filtered_metrics_df_session_state_key] = st.session_state[bar_filtered_metrics_df_session_state_key].reindex(level=1, columns=st.session_state[bar_metric_multiselect_key])
		st.session_state[line_filtered_metrics_df_session_state_key] = st.session_state[line_filtered_metrics_df_session_state_key].reindex(level=1, columns=st.session_state[line_metric_multiselect_key])

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

	def create_single_case_bar_chart(bars_df: pd.DataFrame, lines_df: pd.DataFrame):
		'''Create a plotly line chart containing all NCP cases from the data supplied in the primary and secondary axis DataFrames
		
		Args:
			bars_df: A DataFrame containing the metrics to plot as stacked bars with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
			lines_df: A DataFrame containing the metrics to plot as lines with Hour as the index and Case -> Metric Name -> Agent as the multiindex column levels
			
		Returns:
			None
		
		'''
		# Create boolean variables to indicate the empty status of each DataFrame
		bars_empty   = False
		lines_empty = False
		# Create an empty list to store selected Case Names in each axis DataFrame
		case_list = []
		# If bars_df or lines_df are empty, set their respective empty flag to True
		# Else, return a copy of each respective DataFrame with the Case multiindex column level converted into a separate column called "Case Name"
		# 		and the Metric Name and Agent multiindex column levels flattened with the format "Metric Name [Agent]" (or only "Metric Name" for columns with no Agent)
		if bars_df.empty:
			bars_empty = True
		else:
			bars_df = bars_df.stack(level=0) # Move the Case (0) multiindex column level into the index (shifting Metric Name and Agent multiindex column levels from 1 -> 0 and 2 -> 1 respectivley)
			bars_df.index.names = ['Hour', 'Case Name']
			bars_df = bars_df.reset_index(level='Case Name')
			bars_df.columns = bars_df.columns.map(lambda x: (f'{x[0]} [{x[1]}]') if x[1] != '' else (x[0])) # Metric Name (1), Agent (0)
			case_list += list(bars_df['Case Name'].unique())
		if lines_df.empty:
			lines_empty = True
		else:
			lines_df = lines_df.stack(level=0) # Move the Case (0) multiindex column level into the index (shifting Metric Name and Agent multiindex column levels from 1 -> 0 and 2 -> 1 respectivley)
			lines_df.index.names = ['Hour', 'Case Name']
			lines_df = lines_df.reset_index(level='Case Name')
			lines_df.columns = lines_df.columns.map(lambda x: (f'{x[0]} [{x[1]}]') if x[1] != '' else (x[0])) # Metric Name (1), Agent (0)
			case_list += list(lines_df['Case Name'].unique())

		# Remove duplicate cases in case_list
		case_list = list(set(case_list))

		# For each Case Name provided in bars_df and lines_df, create a new subplot with Case Name as the title
		for case_name in case_list:
			master_fig = make_subplots(subplot_titles=[f'<b>{case_name}</b>'])
			if not bars_empty:
				bars_case_df = bars_df[bars_df['Case Name'] == case_name].copy()
				bars_index = bars_case_df.loc[:, bars_case_df.columns != 'Case Name'].index
				bars_columns = bars_case_df.loc[:, bars_case_df.columns != 'Case Name'].columns
				bars_fig = px.bar(data_frame=bars_case_df,
									x=bars_index,
									y=bars_columns)
				master_fig.add_traces(bars_fig.data)
				master_fig.update_layout(barmode='stack', yaxis=dict(title_text='<br>'.join(bars_case_df.loc[:, bars_case_df.columns != 'Case Name'])))
			if not lines_empty:
				lines_case_df = lines_df[lines_df['Case Name'] == case_name].copy()
				lines_index = lines_case_df.loc[:, lines_case_df.columns != 'Case Name'].index
				lines_columns = lines_case_df.loc[:, lines_case_df.columns != 'Case Name'].columns
				lines_fig = px.line(data_frame=lines_case_df,
									x=lines_index,
									y=lines_columns)
				lines_fig.update_traces(line_color='#000000')
				master_fig.add_traces(lines_fig.data)
				master_fig.update_layout(yaxis2=dict(title_text='<br>'.join(lines_case_df.loc[:, lines_case_df.columns != 'Case Name'])))
			master_fig.update_layout(autosize=True, xaxis_title='Hour', colorway=px.colors.qualitative.Light24)
			
			# Add a divider to separate the filters from the chart
			st.divider()
			# Update the margins
			master_fig.update_layout(margin={'l':20, 'r':20, 't':20, 'b':0})
			# Create an st.empty() placeholder object to display our line chart (Redraw the chart in place anytime the X or Y-Axis bounds are updated)
			redraw_chart = st.empty()
			redraw_chart.write(master_fig)

			# Combine the primary and secondary axis dataframes and remove any duplicate columns ('Hour' and 'Case Name')
			# combined_case_chart_df used to calculate the x-axis upper and lower bounds and download chart data
			if bars_empty and not lines_empty:
				combined_case_chart_df = lines_case_df.copy()
			if not bars_empty and lines_empty:
				combined_case_chart_df = bars_case_df.copy()
			if not bars_empty and not lines_empty:
				combined_case_chart_df = bars_case_df.merge(right=lines_case_df, on=['Hour', 'Case Name'], how='outer')

			# Calculate the Y-Axis upper and lower bounds
			y_max = combined_case_chart_df.loc[:, combined_case_chart_df.columns != 'Case Name'].max().max()
			y_min = combined_case_chart_df.loc[:, combined_case_chart_df.columns != 'Case Name'].min().min()
			y_lower_bound, y_upper_bound = define_initial_y_axis_bounds(min_value=y_min, max_value=y_max)
			y_tick_size = (y_upper_bound - y_lower_bound) / 10
			# Calculate the X-Axis upper and lower bounds
			x_lower_bound = combined_case_chart_df.index.min()
			x_upper_bound = combined_case_chart_df.index.max()

			# Split the lower portion of the page into 3 columns, with column 1 containing download chart data button, column 2 containing lower bound X-Axis adjustment input, and column 3 containing upper bound X-Axis adjustment input
			axis_expander_column, _, download_chart_data_column = st.columns((3, 4, 3))
			with axis_expander_column:
				with st.expander('Update Axis Bounds'):
					axis_label_column, axis_lower_bounds_column, axis_upper_bounds_column = st.columns((2.5, 1.25, 1.25))
					with axis_label_column:
						st.text_input('Y-Axis Label', value='Y-Axis', key=f'y_label_text_input_{case_name}', disabled=True, label_visibility='collapsed')
						st.text_input('X-Axis Label', value='X-Axis', key=f'x_label_text_input_{case_name}', disabled=True, label_visibility='collapsed')
					with axis_lower_bounds_column:
						y_lower_bound_input = st.number_input(label='Lower Bound', disabled=bars_empty, value=y_lower_bound, key=f'y_lower_bound_number_input_{case_name}', label_visibility='collapsed')
						x_lower_bound_input = st.number_input(label='Lower Bound', value=x_lower_bound, key=f'x_lower_bound_number_input_{case_name}', label_visibility='collapsed')
					with axis_upper_bounds_column:
						y_upper_bound_input = st.number_input(label='Upper Bound', disabled=bars_empty, value=y_upper_bound, key=f'y_upper_bound_number_input_{case_name}', label_visibility='collapsed')
						x_upper_bound_input = st.number_input(label='Upper Bound', value=x_upper_bound, key=f'x_upper_bound_number_input_{case_name}', label_visibility='collapsed')

			# Recalculate the Y-Axes tick sizes and update the master_fig layout
			y_tick_size = (y_upper_bound_input - y_lower_bound_input) / 10
			master_fig.update_layout(yaxis={'range': [y_lower_bound_input, y_upper_bound_input], 'tick0': y_lower_bound_input, 'dtick': y_tick_size},
									xaxis={'range': [x_lower_bound_input, x_upper_bound_input]})
			# Redraw the chart
			redraw_chart.write(master_fig)
			
			with download_chart_data_column:
				chart_output_df = combined_case_chart_df.copy()
				chart_output_df = chart_output_df.set_index('Case Name', append=True).unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)
				chart_output = chart_output_df.to_csv()
				st.download_button(label='Download Chart Data', data=chart_output, use_container_width=True, file_name=f'{case_name}_bar_chart_data.csv', key=f'bar_chart_download_button_{case_name}')

#--------------------------------------------------------------------------------------------------#
# 3. Streamlit Web App - Page Specific Configuration
#--------------------------------------------------------------------------------------------------#
	
	# Initialize Session State variables to store variables across mulitple pages and app reruns
	if 'bar_chart_case_selection' not in st.session_state:
		st.session_state.bar_chart_case_selection = None
		st.session_state.bar_chart_bar_metric_selection = None
		st.session_state.bar_chart_bar_agent_options = []
		st.session_state.bar_chart_bar_agent_selection = None
		st.session_state.bar_chart_bar_filtered_metrics_df = pd.DataFrame()
		st.session_state.bar_chart_line_metric_selection = None
		st.session_state.bar_chart_line_agent_options = []
		st.session_state.bar_chart_line_agent_selection = None
		st.session_state.bar_chart_line_filtered_metrics_df = pd.DataFrame()
		
	# Instantiate a container object
	with st.container():
		# If Case Data is loaded
		if st.session_state.data_progress == 100:
			# Case Select - Create a Multiselect widget to select cases to plot
			selected_case = st.multiselect('Case Select', 
								  		options=st.session_state.case_list, 
										default=st.session_state.bar_chart_case_selection,
										placeholder='Select a Case',
										key='bar_chart_case_multiselect')

			# Metric Select - Create 2 Multiselect widgets to select metrics to plot on either the bar or line axes
			bar_metric_column, line_metric_column = st.columns(2)
			with bar_metric_column:
				selected_bar_metric = st.multiselect('Metric Select', 
											   			options=st.session_state.metric_list, 
														default=st.session_state.bar_chart_bar_metric_selection,
														placeholder='Select a Stacked Bar Chart Metric',
														key='bar_chart_bar_metric_multiselect', 
														on_change=update_agent_multiselect, 
														kwargs=dict(self_multiselect_key='bar_chart_bar_metric_multiselect',
																	agent_multiselect_key='bar_chart_bar_agent_multiselect',
																	agent_session_state_key='bar_chart_bar_agent_selection',
																	agent_options_session_state_key='bar_chart_bar_agent_options'))
			with line_metric_column:
				selected_line_metric = st.multiselect('Metric Select', 
											   			options=st.session_state.metric_list,
														default=st.session_state.bar_chart_line_metric_selection,
														placeholder='Select a Line Chart Metric',
														key='bar_chart_line_metric_multiselect',
														label_visibility='hidden',
														on_change=update_agent_multiselect, 
														kwargs=dict(self_multiselect_key='bar_chart_line_metric_multiselect',
																	agent_multiselect_key='bar_chart_line_agent_multiselect',
																	agent_session_state_key='bar_chart_line_agent_selection',
																	agent_options_session_state_key='bar_chart_line_agent_options'))

			# Agent Select - Create 2 Multiselect widgets to select agents to plot on either the bar or line axes
			bar_agent_column, line_agent_column = st.columns(2)
			with bar_agent_column:
				selected_bar_agent = st.multiselect('Agent Select', 
											  			options=st.session_state.bar_chart_bar_agent_options, 
														default=st.session_state.bar_chart_bar_agent_selection,
														placeholder='Select a Stacked Bar Chart Agent',
														key='bar_chart_bar_agent_multiselect')
			with line_agent_column:
				selected_line_agent = st.multiselect('Agent Select', 
														options=st.session_state.bar_chart_line_agent_options, 
														default=st.session_state.bar_chart_line_agent_selection,
														placeholder='Select a Line Chart Agent',
														key='bar_chart_line_agent_multiselect',
														label_visibility='hidden')

			# Apply Filters - Create a button to apply the above filters
			apply_filters_button = st.button('Apply Filters', 
											 use_container_width=True,
											 key='bar_chart_apply_filters',
											 on_click=filter_metrics_df,
											 kwargs=dict(case_multiselect_key='bar_chart_case_multiselect',
														case_session_state_key='bar_chart_case_selection',
														bar_metric_multiselect_key='bar_chart_bar_metric_multiselect',
														bar_metric_session_state_key='bar_chart_bar_metric_selection',
														bar_agent_multiselect_key='bar_chart_bar_agent_multiselect',
														bar_agent_session_state_key='bar_chart_bar_agent_selection',
														line_metric_multiselect_key='bar_chart_line_metric_multiselect',
														line_metric_session_state_key='bar_chart_line_metric_selection',
														line_agent_multiselect_key='bar_chart_line_agent_multiselect',
														line_agent_session_state_key='bar_chart_line_agent_selection',
														bar_filtered_metrics_df_session_state_key='bar_chart_bar_filtered_metrics_df',
														line_filtered_metrics_df_session_state_key='bar_chart_line_filtered_metrics_df'))
			
			# Line Chart - Create a line chart when either the bar or line filtered_metrics_df has been udpated (via Apply Filters button)
			if (not st.session_state.bar_chart_bar_filtered_metrics_df.empty or not st.session_state.bar_chart_line_filtered_metrics_df.empty):
				create_single_case_bar_chart(bars_df=st.session_state.bar_chart_bar_filtered_metrics_df,
											lines_df=st.session_state.bar_chart_line_filtered_metrics_df)

		# If not Case Data is loaded
		else:
			# Create a markdown text explaining that no Data has been loaded
			st.markdown('''<b><font size="5">NO DATA LOADED</font></b>
			  			</br>
						To get NCP case data, first open the navigation menu on the left and press the <b>Fetch Case Data</b> button. Follow the instructions provided and select the NCP cases you wish to review and analyze:<br>
						<ol>
			  				<li>Select one of the 4 available network drive locations to retrieve NCP model results from (only 1 available in this demo)</li>
			  				<li>Select the folder containing the NCP model results you would like to review and analyze</li>
			  				<li>Filter out any NCP model result folders that are not required save</li>
			  			</ol>
						''',
						unsafe_allow_html=True)
					
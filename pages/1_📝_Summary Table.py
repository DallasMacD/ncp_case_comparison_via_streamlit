import streamlit as st
import pandas as pd
import os

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

	def filter_summary_df(case_filter_text: str, summary_df: pd.DataFrame) -> pd.DataFrame:
		'''A function that filters summary_df based on the case_filter_text

		Args:
			summary_df: A DataFrame containing summary metrics from the NCP case provided, with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels

		Returns:
			filtered_summary_df: A filtered DataFrame containing summary metrics from the NCP case provided, with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels

		'''
		# Filter summary_df by case_filter_text
		filtered_summary_df = summary_df.copy()
		if case_filter_text:
			case_filter_text = case_filter_text.replace('\\', '\\\\')
			case_filter_text_list = case_filter_text.split('&')
			for text in case_filter_text_list:
				filtered_summary_df = filtered_summary_df[filtered_summary_df.index.str.contains(text, case=False, regex=True)]
		return filtered_summary_df
	
	def stylize_summary_df(summary_df: pd.DataFrame):
		'''A function that stylizes the summary dataframe provided

		Args:
			summary_df: A DataFrame containing summary metrics from the NCP case provided, with Case Name as the index and Summary Group -> Metric Names as the multiindex column levels

		Returns:
			stylized_summary: summary_df DataFrame as a pandas Styler object

		'''
		# Create a stylized DataFrame object from the summary_df and format it to meet our specifications
		stylized_summary = summary_df.style
		# Format the numerical columns as floats with 0 or 2 decimal places
		stylized_summary.format({('Convergence (%)', 'Average'): 			'{:.2f}', 
								('Convergence (%)', 'Worst'): 				'{:.2f}', 
								('Sum of Generation (MW)', 'Provincial'): 	'{:,.0f}', 
								('Sum of Generation (MW)', 'Hydro'): 		'{:,.0f}', 
								('Sum of Generation (MW)', 'LNR'): 			'{:,.0f}', 
								('Sum of Generation (MW)', 'Thermal'): 		'{:,.0f}', 
								('Sum of Generation (MW)', 'Renewables'): 	'{:,.0f}', 
								('Sum of Load (MW)', 'Demand'): 			'{:,.0f}', 
								('Sum of Load (MW)', 'Deficits'): 			'{:,.0f}', 
								('Sum of Exchange (MW)', 'Imports'): 		'{:,.0f}', 
								('Sum of Exchange (MW)', 'Opp. Exports'): 	'{:,.0f}', 
								('Financials (USD k$)', 'Violations'): 		'${:,.0f}', 
								('Financials (USD k$)', 'Revenues'): 		'${:,.0f}', 
								('Financials (USD k$)', 'Costs'): 			'${:,.0f}', 
								('Financials (USD k$)', 'Net Revenue'): 	'${:,.0f}'})
		# Initialize Positive and Negative bar chart colors
		positive_bar_color = '#ABD331'
		negative_bar_color = '#F59421'
		# Format specific columns to include bar charts
		for column in [('Sum of Generation (MW)', 'Provincial'), ('Sum of Generation (MW)', 'Hydro'), ('Sum of Generation (MW)', 'LNR'), ('Sum of Generation (MW)', 'Thermal'), ('Sum of Exchange (MW)', 'Imports'), ('Sum of Exchange (MW)', 'Opp. Exports'), ('Financials (USD k$)', 'Revenues')]:
			bar_min = summary_df[column].min()
			bar_max = summary_df[column].max() if summary_df[column].max() != summary_df[column].min() else summary_df[column].max() + 1
			stylized_summary.bar(subset=pd.IndexSlice[:, pd.IndexSlice[:, column[1]]], axis=0, vmin=bar_min, vmax=bar_max, color=positive_bar_color)
		for column in [('Financials (USD k$)', 'Costs'), ('Financials (USD k$)', 'Violations'), ('Sum of Load (MW)', 'Deficits')]:
			bar_min = summary_df[column].min()
			bar_max = summary_df[column].max() if summary_df[column].max() != summary_df[column].min() else summary_df[column].max() + 1
			stylized_summary.bar(subset=pd.IndexSlice[:, pd.IndexSlice[:, column[1]]], axis=0, vmin=bar_min, vmax=bar_max, color=negative_bar_color)
		for column in [('Financials (USD k$)', 'Net Revenue')]:
			bar_min = summary_df[column].min()
			bar_max = summary_df[column].max() if summary_df[column].max() != summary_df[column].min() else summary_df[column].max() + 1
			stylized_summary.bar(subset=pd.IndexSlice[:, pd.IndexSlice[:, column[1]]], axis=0, vmin=bar_min, vmax=bar_max, color=[negative_bar_color, positive_bar_color])
		return stylized_summary

#--------------------------------------------------------------------------------------------------#
# 3. Streamlit Web App - Page Specific Configuration
#--------------------------------------------------------------------------------------------------#
	
	# Initialize Session State variables to store variables across mulitple pages and app reruns
	if 'summary_table_filter' not in st.session_state:
		st.session_state.summary_table_filter = ''
	
	# Instantiate a container object
	with st.container():
		# If Case Data is loaded
		if st.session_state.data_progress == 100:
			case_filter_column, _, export_summary_column = st.columns((2.5, 5, 2.5))
			# Create text input to filter summary_df
			with case_filter_column:
				case_filter_text = st.text_input('Case Filter', value=st.session_state.summary_table_filter, key='summary_table_filter_text_input', placeholder='Filter by Case Name ("&" and "|")', label_visibility='collapsed')
				filtered_summary_df = filter_summary_df(case_filter_text, st.session_state.summary_df)
				st.session_state.summary_table_filter = case_filter_text
			# Create a download button to download filtered_summary_df
			with export_summary_column:
				filtered_summary_output = filtered_summary_df.to_csv()
				export_summary_button = st.download_button('Export Summary Table', data=filtered_summary_output, file_name='Summary Table.csv', key='summary_table_export_button')
			# Create a table by converting filtered_summary_df into a pandas Styler object then converting it into an HTML object
			stylized_summary_df = stylize_summary_df(filtered_summary_df)
			summary_html = stylized_summary_df.to_html()
			summary_table = st.write(summary_html, unsafe_allow_html=True)
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

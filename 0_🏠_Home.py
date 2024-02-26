import streamlit as st
import pandas as pd
import os

from data.data_module import select_cases, develop_case_metrics, generate_filters, CSV_METRICS_DICT, DAT_METRICS_DICT

#--------------------------------------------------------------------------------------------------#
# 1. Streamlit Web App - Base Configuration
#--------------------------------------------------------------------------------------------------#

# NOTE: Anytime a widget on the Streamlit interface is interacted with, Streamlit will rerun the code below
if __name__ == '__main__':

	# Change the directory to the directory containing this script
	cwd = os.path.dirname(__file__)
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
# 2. Streamlit Web App - Page Specific Configuration
#--------------------------------------------------------------------------------------------------#
			
	with st.container():
		st.markdown('''<b><font size="5">WELCOME</font></b>
			  		</br>
					This app is designed to improve the review and analysis process relating to the NCP system simulation model. You can use the navigation menu on the left to view NCP model outputs in various formats.
			  		<br>
			  		<br>
					<b><font size="5">GETTING STARTED</font></b>
			  		</br>
					To get started, first open the navigation menu on the left and press the <b>Fetch Case Data</b> button. Follow the instructions provided and select the NCP cases you wish to review and analyze:<br>
					<ol>
			  			<li>Select one of the 4 available network drive locations to retrieve NCP model results from</li>
			  			<li>Select the folder containing the NCP model results you would like to review and analyze</li>
			  			<li>Filter out any NCP model result folders that are not required save</li>
			  		</ol>
					''',
					unsafe_allow_html=True)

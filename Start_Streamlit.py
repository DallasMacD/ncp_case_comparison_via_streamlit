import os
import subprocess
import socket
import webbrowser

# Change the directory to the directory containing this script
cwd = os.path.dirname(__file__)
os.chdir(cwd)

# Run the ncp_case_comparison_using_streamlit.py script from a command prompt window
subprocess.Popen('streamlit run ' + '"' + cwd + '\\0_🏠_Home.py" --theme.base="light" --theme.primaryColor="#0079C1" --theme.backgroundColor="#FFFFFF" --theme.secondaryBackgroundColor="#E5F5FD" --theme.textColor="#262730" --theme.font="sans serif" --server.headless true', creationflags=subprocess.CREATE_NEW_CONSOLE)

# Get the IP address of your computer
host_name = socket.gethostname()
ip_address = socket.gethostbyname(host_name)

# Open a web browser using your computer's IP address and the port used by streamlit (8501)
webbrowser.open('http:\\\\' + str(ip_address) + ':8501')
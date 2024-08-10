import streamlit as st
import pandas as pd
import os
import json

from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying, flatten_dict_column
from typing import List

def generate_config(server_endpoint, pat_name, pat_key, site_name, env):
    config = {
        env: {
            "server": server_endpoint,
            "api_version": "3.19",
            "personal_access_token_name": pat_name,
            "personal_access_token_secret": pat_key,
            "site_name": site_name,
            "site_url": site_name
        },
        "env": env
    }
    return config

def connect_to_ts(config,env):
    conn = TableauServerConnection(config_json=config, env=env)
    response = conn.sign_in()
    return conn,response

def load_dashboard_file(dashboardfile):
    # Load the dashboard file CSV
    dashboarddf = pd.read_csv(dashboardfile)
    st.write(f'Dashboard file read')
        # Extract unique project names
    dashboarddf['project'] = dashboarddf['project'].apply(lambda x: json.loads(x.replace("'", "\"")))  # Convert string to JSON
    unique_projects = dashboarddf['project'].apply(lambda x: x['name']).unique()
    
    return dashboarddf, unique_projects

def get_dashboard_img(selected_dashboard,dashboardfile, conn):
    # Load dashboard file CSV to get the ID of the selected dashboard
    df = pd.read_csv(dashboardfile)
    # Find the ID corresponding to the selected dashboard
    row = df[df['name'] == selected_dashboard].iloc[0]
    dashid = row['id']
    st.write(f"Dashid : {dashid}")  # Display dashid for verification
    
    # Filename for saving the image
    filename = selected_dashboard + '.png'
    
    # Fetch dashboard image from Tableau Server
    view_png = conn.query_view_image(view_id=dashid)
    return view_png
    
def logout(conn):
    conn.sign_out()
    st.write(f'Connection logged out')

def main():
    portalname = "890Portal"
    project_folder = portalname
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(ROOT_PATH,project_folder)):
        os.makedirs(project_folder)

    projectpath = os.path.join(ROOT_PATH,project_folder)
    viewsfile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_Views.csv")
    sitesfile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_Sites.csv")
    workbookfile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_Workbooks.csv")
    dashboardfile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_Dashboard.csv")
    filtereddashboardfile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_Demo_Dashboards.csv")
    lineagefile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_metadata.csv")

    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "df" not in st.session_state:
        st.session_state.df = None

    if "unique_projects" not in st.session_state:
        st.session_state.unique_projects = []

    if "selected_project" not in st.session_state:
        st.session_state.selected_project = None

    if "selected_workbook" not in st.session_state:
        st.session_state.selected_workbook = None

    if "site_name" not in st.session_state:
        st.session_state.site_name = None

    # Set the title of the app
    st.title("Tableau Metadata Extraction")

    # Sidebar - always displayed
    if st.session_state.authenticated:
        st.sidebar.title(st.session_state.site_name)

        if st.session_state.df is not None:
            selected_project = st.sidebar.selectbox("Project", ["Choose an option"] + list(st.session_state.unique_projects), index=0)
            st.session_state.selected_project = selected_project

            # Filter workbooks based on selected project
            project_workbooks = st.session_state.df[st.session_state.df['project'].apply(lambda x: x['name'] == selected_project)]['workbook_name'].unique()
            selected_workbook = st.sidebar.selectbox("Workbook", ["Choose an option"] + list(project_workbooks), index=0)
            st.session_state.selected_workbook = selected_workbook

            # Filter dashboards based on selected workbook
            workbook_dashboards = st.session_state.df[(st.session_state.df['project'].apply(lambda x: x['name'] == selected_project)) & (st.session_state.df['workbook_name'] == selected_workbook)]['name'].unique()
            selected_dashboard = st.sidebar.selectbox("Dashboard", ["Choose an option"] + list(workbook_dashboards))
            view_png = get_dashboard_img(selected_dashboard,dashboardfile, connection)
            image_path = os.path.join(ROOT_PATH, project_folder, 'Dashboard_images', selected_dashboard)
            with open(image_path, 'wb') as v:
                v.write(view_png.content)
            st.write(f'The Dashboard image for {selected_dashboard} is extracted')
            # Display the extracted image in the main area
            st.image(image_path)
        else:
            st.sidebar.write("Dashboard file could not be loaded. Please check the file path and format.")
    else:
        st.sidebar.title("Sidebar")

    if not st.session_state.authenticated:
        # Main area
        st.subheader("Please enter the following details:")

        # Text inputs for Server endpoint, PAT name, PAT key, Site name, and Environment name
        server_endpoint = st.text_input("Server Endpoint")
        pat_name = st.text_input("PAT Name")
        pat_key = st.text_input("PAT Key", type="password")
        site_name = st.text_input("Site Name")
        env = st.text_input("Environment name")

        # Sign-in button
        if st.button("Sign In"):
            config = generate_config(server_endpoint, pat_name, pat_key, site_name, env)
            connection, response = connect_to_ts(config, env)

            # Check the response status
            if response.status_code == 200:
                st.success("Successfully signed in to Tableau Server.")
                st.session_state.authenticated = True
                st.session_state.site_name = site_name

                # Load dashboard file and get unique projects
                df, unique_projects = load_dashboard_file(dashboardfile)
                st.session_state.df = df
                st.session_state.unique_projects = unique_projects
                # Rerun the app to refresh the sidebar
                st.experimental_rerun()
                # Save dashboard image
                
                
            else:
                st.error(f"Failed to sign in to Tableau Server. Status code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    main()

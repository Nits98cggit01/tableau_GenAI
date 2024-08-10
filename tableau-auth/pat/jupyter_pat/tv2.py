import streamlit as st
import pandas as pd
import os
import json
import openai

from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying, flatten_dict_column
from typing import List
from config import *

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI

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
    st.subheader(f"Dashid : {dashid}")  # Display dashid for verification
    
    # Filename for saving the image
    filename = selected_dashboard + '.png'
    
    # Fetch dashboard image from Tableau Server
    view_png = conn.query_view_image(view_id=dashid)
    return view_png

def save_dashboard_image(view_png, selected_dashboard,image_dir):
    filename = selected_dashboard + ".png"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    image_path = os.path.join(image_dir, filename)
    with open(image_path, 'wb') as v:
        v.write(view_png.content)

def filter_dashboard_demo(dashboardfile,selected_workbook,selected_dashboard):
    dash_df = pd.read_csv(dashboardfile)
    desired_dashboard = selected_dashboard
    st.subheader(f'Desired dashboard is : {desired_dashboard}')
    newdashdf = dash_df[(dash_df['name'] == selected_dashboard) & (dash_df['workbook_name'] == selected_workbook)]
    return newdashdf   

def display_dash(dash_dir,filename):
    image_path = os.path.join(dash_dir,f'{filename}.png')
    # Check if the image file exists
    if os.path.exists(image_path):
        # st.image(image_path, caption=f"Displaying {filename}", use_column_width=True)
        st.image(image_path, width=300)
    else:
        st.write(f"Image file '{filename}' not found in '{dash_dir}'.")

def get_lineage(conn,dashboardfile,portalname):
    dashboard_df = pd.read_csv(dashboardfile)
    print(f'Dashboard df read')
    luids = dashboard_df['id'].tolist()
    rows = []

    print(f'Dashboard id is extracted and is passed to the query')
    for index, row in dashboard_df.iterrows():
        luid = row['id']
        print(f"Query for the dashboard id {luid}")
        query = f'''
        {{
          dashboards (filter: {{luid: "{luid}"}}) {{
            id
            name
            index
            luid
            documentViewId
            sheets {{
              id
              name
              upstreamTables {{
                id
                name
                columns {{
                  id
                }}
              }}
              upstreamFields {{
                id
                name
                isHidden
                directSheets {{
                  id
                }}
              }}
              upstreamColumns {{
                id
                name
                table {{
                  id
                  name
                }}
                isNullable
              }}
              upstreamDatabases {{
                id
                name
                connectionType
              }}
              upstreamDatasources {{
                id
                name
                datasourceFilters {{
                  id
                }}
              }}
            }}
          }}
        }}

        '''
        
        response = conn.metadata_graphql_query(query=query)
        response_json = response.json()
        data = response_json
        dashboards = data['data']['dashboards']

        for dashboard in dashboards:
            dashboard_name = dashboard['name']
            for sheet in dashboard['sheets']:
                sheet_name = sheet['name']
                
#                 datasources_count = len(sheet['upstreamDatasources'])
                datasource_names = ','.join([ds['name'] for ds in sheet['upstreamDatasources']])
                database_names = ','.join([db['name'] for db in sheet['upstreamDatabases']])
                database_connection_types = ','.join([db['connectionType'] for db in sheet['upstreamDatabases']])
                column_count = len(sheet['upstreamColumns'])
                column_names = ','.join([col['name'] for col in sheet['upstreamColumns']])
                table_names = ','.join([col['table']['name'] for col in sheet['upstreamColumns'] if col['table']])
                project_json = row['project'].replace("'", '"')
                project_name = json.loads(project_json)['name']
                workbook_name = row['workbook_name']

                rows.append({
                    'Site_name': portalname,
                    'Project_name': project_name,
                    'Workbook_name': workbook_name,
                    'Dashboard_name': dashboard_name,
                    'Sheet_name': sheet_name,
                    'Datasource_name': datasource_names,
                    'Database_name': database_names,
                    'Database_connection': database_connection_types,
                    'Number of columns' : column_count,
                    'Column_name': column_names,
                    'Table_name': table_names
                })


    lineage_df = pd.DataFrame(rows)
    return lineage_df

def display_analysis(filterdashfile):
    # Read the CSV file
    df = pd.read_csv(filterdashfile)
    
    # Display the Sheet names in a comma-separated manner
    sheet_names = ', '.join(df['Sheet_name'].unique())
    st.subheader(f"Sheet Names: {sheet_names}")
    
    # Display the unique Datasource names in a comma-separated manner
    datasource_names = ', '.join(df['Datasource_name'].unique())
    st.subheader(f"Unique Datasource Names: {datasource_names}")

    unique_sheet_names = df['Sheet_name'].unique()
    for i in range(0, len(unique_sheet_names), 2):
        cols = st.columns(2)
        
        for j in range(2):
            if i + j < len(unique_sheet_names):
                sheet_name = unique_sheet_names[i + j]
                sheet_df = df[df['Sheet_name'] == sheet_name]
                
                # Normalize the data for the current sheet
                normalized_data = []
                for index, row in sheet_df.iterrows():
                    column_names = row['Column_name'].split(',')
                    table_names = row['Table_name'].split(',')
                    for column_name, table_name in zip(column_names, table_names):
                        normalized_data.append({
                            'Column Name': column_name.strip(),
                            'Table Name': table_name.strip(),
                            'Database Name': row['Database_name'],
                            'Database Connection': row['Database_connection']
                        })
                
                # Convert normalized data to DataFrame
                normalized_df = pd.DataFrame(normalized_data)
                
                # Display the table in the appropriate column
                with cols[j]:
                    st.subheader(f"Columns in View for Sheet: {sheet_name}")
                    st.table(normalized_df)
    
    # # Iterate through each sheet name to create tables
    # for sheet_name in df['Sheet_name'].unique():
    #     sheet_df = df[df['Sheet_name'] == sheet_name]
        
    #     # Create a container to hold the tables
    #     with st.container():
    #         st.subheader(f"Columns in View for Sheet: {sheet_name}")
            
    #         # Normalize the data for the current sheet
    #         normalized_data = []
    #         for index, row in sheet_df.iterrows():
    #             column_names = row['Column_name'].split(',')
    #             table_names = row['Table_name'].split(',')
    #             for column_name, table_name in zip(column_names, table_names):
    #                 normalized_data.append({
    #                     'Column Name': column_name.strip(),
    #                     'Table Name': table_name.strip(),
    #                     'Database Name': row['Database_name'],
    #                     'Database Connection': row['Database_connection']
    #                 })
            
    #         # Convert normalized data to DataFrame
    #         normalized_df = pd.DataFrame(normalized_data)
            
    #         # Display the table
    #         st.table(normalized_df)

def call_openai(prompt):
    openai.api_type = API_TYPE
    openai.api_base = API_BASE
    openai.api_version = API_VERSION
    openai.api_key = API_KEY

    message_text = [{"role":"system","content":f"{prompt}"}]

    response = openai.ChatCompletion.create(
          engine="gpt-35-turbo",
          messages = message_text,
          temperature=0.2,
          max_tokens=500,
          top_p=0.95,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None
        )

    res_output = response['choices'][0]['message']['content'] 
    return res_output

def get_prompt(lineagefile):
    # Read the CSV file
    try:
        df = pd.read_csv(lineagefile)
        df['Description'] = ""
        # Create a list to store the prompts
        prompts = []
        
        # Iterate through each row and create the prompt
        for index, row in df.iterrows():
            data = f'''
    Dashboard - {row['Dashboard_name']}
    Workbook - {row['Workbook_name']}
    Project - {row['Project_name']}
    Site name - {row['Site_name']}
    Sheet name - {row['Sheet_name']}
    Datasource - {row['Datasource_name']}
    Database - {row['Database_name']}
    Db connection - {row['Database_connection']}
    Column count - {row['Number of columns']}
    Column name - {row['Column_name']}
    Table name - {row['Table_name']}
    '''
            prompt = f'''
            Act as a Tableau expert, you are given the metadata for a dashboard below \n
            {data}. Provide a brief summary of the sheet with the data
            '''

            res_output = call_openai(prompt)
            df.at[index, 'Description'] = res_output
        
        # Save the updated DataFrame back to the CSV file
        df.to_csv(lineagefile, index=False)

        prompts.append(prompt)
        st.success(f'The description is updated for the metadata')
    except:
        st.warning(f'Error occured...Cannot update the description')
    return prompts

def logout(conn):
    conn.sign_out()
    st.write(f'Connection logged out')

def main():
    st.set_page_config(layout="wide")
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

    if "selected_dashboard" not in st.session_state:
        st.session_state.selected_dashboard = None

    if "site_name" not in st.session_state:
        st.session_state.site_name = None

    # Set the title of the app
    st.title("Tableau Metadata Extraction")

    # Sidebar - always displayed
    if st.session_state.authenticated:
        st.sidebar.title(st.session_state.site_name)

        if st.session_state.df is not None:
            selected_project = st.session_state.selected_project
            if selected_project is None:
                selected_project = st.sidebar.selectbox("Project", ["Choose an option"] + list(st.session_state.unique_projects))
            else:
                selected_project = st.sidebar.selectbox("Project", ["Choose an option"] + list(st.session_state.unique_projects), index=list(st.session_state.unique_projects).index(selected_project) + 1)
            if selected_project != "Choose an option":
                st.session_state.selected_project = selected_project

                # Filter workbooks based on selected project
                project_workbooks = st.session_state.df[st.session_state.df['project'].apply(lambda x: x['name'] == selected_project)]['workbook_name'].unique()
                selected_workbook = st.session_state.selected_workbook
                if selected_workbook is None:
                    selected_workbook = st.sidebar.selectbox("Workbook", ["Choose an option"] + list(project_workbooks))
                else:
                    selected_workbook = st.sidebar.selectbox("Workbook", ["Choose an option"] + list(project_workbooks), index=list(project_workbooks).index(selected_workbook) + 1)
                if selected_workbook != "Choose an option":
                    st.session_state.selected_workbook = selected_workbook

                    # Filter dashboards based on selected workbook
                    workbook_dashboards = st.session_state.df[(st.session_state.df['project'].apply(lambda x: x['name'] == selected_project)) & (st.session_state.df['workbook_name'] == selected_workbook)]['name'].unique()
                    selected_dashboard = st.session_state.selected_dashboard
                    if selected_dashboard is None:
                        selected_dashboard = st.sidebar.selectbox("Dashboard", ["Choose an option"] + list(workbook_dashboards))
                    else:
                        selected_dashboard = st.sidebar.selectbox("Dashboard", ["Choose an option"] + list(workbook_dashboards), index=list(workbook_dashboards).index(selected_dashboard) + 1)
                    if selected_dashboard != "Choose an option":
                        st.session_state.selected_dashboard = selected_dashboard
                        
                        # Fetch and display dashboard image
                        conn, _ = connect_to_ts(st.session_state.config, st.session_state.env)
                        view_png = get_dashboard_img(selected_dashboard,dashboardfile,conn)
                        dashimg_path = os.path.join(ROOT_PATH,project_folder,'Dashboard image')
                        save_dashboard_image(view_png, selected_dashboard,dashimg_path)
                        display_dash(dashimg_path,selected_dashboard)
                        
                        if os.path.exists(filtereddashboardfile):
                            print(f'Filtered dashboard file exists')
                            os.remove(filtereddashboardfile)
                            print(f'Filtered dashboard file deleted')
                        else:
                            print(f'Filtered dashboard does not exists')
                            
                        filter_dashboard = filter_dashboard_demo(dashboardfile,selected_workbook,selected_dashboard)
                        filter_dashboard.to_csv(filtereddashboardfile)
                        
                        # get_lineage_df = get_lineage(connection,dashboardfile,portalname)
                        get_lineage_df = get_lineage(conn,filtereddashboardfile,portalname)
                        get_lineage_df.to_csv(lineagefile)
                        st.write(get_lineage_df)
                        display_analysis(lineagefile)
                        get_prompt(lineagefile)

                        
                                            
            else:
                st.session_state.selected_project = None
                st.session_state.selected_workbook = None
                st.session_state.selected_dashboard = None
        else:
            st.sidebar.write("Dashboard file could not be loaded. Please check the file path and format.")
    else:
        pass

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
            conn, response = connect_to_ts(config, env)

            # Check the response status
            if response.status_code == 200:
                st.success("Successfully signed in to Tableau Server.")
                st.session_state.authenticated = True
                st.session_state.site_name = site_name
                st.session_state.config = config
                st.session_state.env = env

                # Load dashboard file and get unique projects
                df, unique_projects = load_dashboard_file(dashboardfile)
                st.session_state.df = df
                st.session_state.unique_projects = unique_projects

                # Rerun the app to refresh the sidebar
                st.experimental_rerun()
            else:
                st.error(f"Failed to sign in to Tableau Server. Status code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    main()
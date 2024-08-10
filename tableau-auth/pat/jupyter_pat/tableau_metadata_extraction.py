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

def get_views(conn):
    views_df = querying.get_views_dataframe(conn)
    get_site_views = flatten_dict_column(views_df,keys=["name","id"],col_name="workbook")
    st.write(f'Views extracted and is passed to the main function')
    return get_site_views

def get_sites(conn):
    sites_df = querying.get_sites_dataframe(conn)
    st.write(f'Sites extracted and is passed to the main function')
    return sites_df

def get_workbook(conn):
    st.write(f'Entered workbook')
    wb_df = querying.get_workbooks_dataframe(conn)
    st.write(f'Workbook extracted and is passed to the main function')
    return wb_df

def get_dashboard(views_df):
    portal_dashdf = views_df[views_df['sheetType'] == 'dashboard']
    portal_dashdf = portal_dashdf[['name', 'id', 'contentUrl', 'sheetType', 'viewUrlName', 'workbook_name', 'workbook_id','project']]
    st.write(f'Identified dashboard and is passed to the main function')
    return portal_dashdf

def load_dashboard_file(dashboardfile):
    # Load the dashboard file CSV
    dashboarddf = pd.read_csv(dashboardfile)
    
    # Extract unique project names
    dashboarddf['project'] = dashboarddf['project'].apply(lambda x: json.loads(x.replace("'", "\"")))  # Convert string to JSON
    unique_projects = dashboarddf['project'].apply(lambda x: x['name']).unique()
    
    return dashboarddf, unique_projects

def filter_dashboard_demo(dashboardfile):
    st.write(f'Entered filter dashboard demo')
    dash_df = pd.read_csv(dashboardfile)
    desired_dashboard = ['Comparison of People Metrics', 'Executive Dashboard','Chennai Port_Demographic','Mumbai Port_Demographic']
    st.write(f'Desired dashboard is : {desired_dashboard}')
    newdashdf = dash_df.loc[dash_df['name'].isin(desired_dashboard)]
    return newdashdf

def get_lineage(conn,dashboardfile,portalname):
    dashboard_df = pd.read_csv(dashboardfile)
    st.write(f'Dashboard df read')
    luids = dashboard_df['id'].tolist()
    rows = []

    st.write(f'Dashboard id is extracted and is passed to the query')
    for index, row in dashboard_df.iterrows():
        luid = row['id']
        st.write(f"Query for the dashboard id {luid}")
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

def logout(conn):
    conn.sign_out()
    st.write(f'Connection logged out')


def main():
    # Set the title of the app
    st.title("Tableau Metadata Extraction")

    # Sidebar
    st.sidebar.title("Sidebar")

    # Main area
    st.subheader("Please enter the following details:")

    # Text inputs for Server endpoint, PAT name, PAT key, and Site name
    server_endpoint = st.text_input("Server Endpoint")
    pat_name = st.text_input("PAT Name")
    pat_key = st.text_input("PAT Key", type="password")
    site_name = st.text_input("Site Name")
    env = st.text_input("Environment name")

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

    # Sign-in button
    if st.button("Sign In"):
        config = generate_config(server_endpoint, pat_name, pat_key, site_name, env)
        connection, response = connect_to_ts(config, env)      
        # Check the response status
        if response.status_code == 200:
            st.success("Successfully signed in to Tableau Server.")
            if not os.path.exists(viewsfile):
                views_df = get_views(connection)
                views_df.to_csv(viewsfile, index=False)
                st.write(f'Views file created')
            else:
                st.write(f'Views file {viewsfile} already exists')

            if not os.path.exists(sitesfile):
                sites_info = get_sites(connection)
                sites_info.to_csv(sitesfile, index=False)
                st.write(f'Sites file created')
            else:
                st.write(f'Sites file {sitesfile} already exists')

            if not os.path.exists(workbookfile):
                wb_df = get_workbook(connection)
                wb_df.to_csv(workbookfile, index=False)
                st.write(f'Workbook file created')
            else:
                st.write(f'Workbook file {workbookfile} already exists')

            if not os.path.exists(dashboardfile):
                get_dashboard_details = get_dashboard(views_df)
                get_dashboard_details.to_csv(dashboardfile, index=False)
                st.write(f'Dashboard file created')
            else:
                st.write(f'Dashboard file {dashboardfile} already exists')

            # Always recreate the filtered dashboard file
            if os.path.exists(filtereddashboardfile):
                st.write(f'Filtered dashboard file exists')
                os.remove(filtereddashboardfile)
                st.write(f'Filtered dashboard file deleted')
            else:
                st.write(f'Filtered dashboard does not exists')
                
            dashboarddf, unique_projects = load_dashboard_file(dashboardfile)
            st.sidebar.title(site_name)
            selected_project = st.sidebar.selectbox("Project", unique_projects)
            
            # Filter workbooks based on selected project
            project_workbooks = dashboarddf[dashboarddf['project'].apply(lambda x: x['name'] == selected_project)]['workbook_name'].unique()
            selected_workbook = st.sidebar.selectbox("Workbook", project_workbooks)
            
            # Filter dashboards based on selected workbook
            workbook_dashboards = dashboarddf[(dashboarddf['project'].apply(lambda x: x['name'] == selected_project)) & (dashboarddf['workbook_name'] == selected_workbook)]['name'].unique()
            selected_dashboard = st.sidebar.selectbox("Dashboard", workbook_dashboards)
            
            logout(connection)
        
        else:
            st.error(f"Failed to sign in to Tableau Server. Status code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    main()

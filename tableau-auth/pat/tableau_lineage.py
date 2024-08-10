import pandas as pd
import os
import json
import configparser

from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying, flatten_dict_column
from typing import List
from config import config

def get_config(configfile):
    config_parser = configparser.ConfigParser()
    # Read the configuration file
    config_parser.read('configuration.txt')
    # Get the connection section
    connection_config = config_parser['connection']
    # Create the config dictionary
    config = {
        "connection": {
            "server": connection_config.get('server'),
            "api_version": connection_config.get('api_version'),
            "personal_access_token_name": connection_config.get('personal_access_token_name'),
            "personal_access_token_secret": connection_config.get('personal_access_token_secret'),
            "site_name": connection_config.get('site_name'),
            "site_url": connection_config.get('site_url')
        }
    }
    return config

def connect_to_ts(config,portalname):
    conn = TableauServerConnection(config_json=config, env=portalname)
    response = conn.sign_in()

    # Check the response status
    if response.status_code == 200:
        print("Successfully signed in to Tableau Server.")
    else:
        print(f"Failed to sign in to Tableau Server. Status code: {response.status_code}, Response: {response.text}")

    return conn

def get_views(conn):
    views_df = querying.get_views_dataframe(conn)
    get_site_views = flatten_dict_column(views_df,keys=["name","id"],col_name="workbook")
    print(f'Views extracted and is passed to the main function')
    return get_site_views

def get_sites(conn):
    sites_df = querying.get_sites_dataframe(conn)
    print(f'Sites extracted and is passed to the main function')
    return sites_df

def get_workbook(conn):
    print(f'Entered workbook')
    wb_df = querying.get_workbooks_dataframe(conn)
    print(f'Workbook extracted and is passed to the main function')
    return wb_df

def get_dashboard(views_df):
    portal_dashdf = views_df[views_df['sheetType'] == 'dashboard']
    portal_dashdf = portal_dashdf[['name', 'id', 'contentUrl', 'sheetType', 'viewUrlName', 'workbook_name', 'workbook_id','project']]
    print(f'Identified dashboard and is passed to the main function')
    return portal_dashdf

def extract_lineage(conn,dashboardfile):
    dashboard_df = pd.read_csv(dashboardfile)
    luids = dashboard_df['luid'].tolist()
    rows = []
    print(f'Dashboard id is extracted and is passed to the query')
    for luid in luids:
        print(f"Query for the dashboard id {luid}")
        query = f'''
        {{
        dashboards (filter: {{luid: "{luid}"}}) {{
            id
            index
            upstreamTables {{
            id
            }}
            upstreamFields {{
            id
            }}
            upstreamDatabases {{
            id
            name
            connectionType
            }}
            upstreamDatasources {{
            id
            name
            }}
            luid
            documentViewId
            sheets {{
            id
            name
            }}
        }}
        }}
        '''
        response = conn.metadata_graphql_query(query=query)
        print(f"Response extracted for the dashboard id {luid}")
        dashboards = response['data']['dashboards']

        for dashboard in dashboards:
            dashboard_id = dashboard['luid']
            dashboard_name = dashboard.get('name', 'N/A')  # Assuming name might be missing

            sheet_entries = dashboard['sheets']
            datasource_ids = ','.join([ds['id'] for ds in dashboard['upstreamDatasources']])
            datasource_names = ','.join([ds['name'] for ds in dashboard['upstreamDatasources']])
            database_names = ','.join([db['name'] for db in dashboard['upstreamDatabases']])
            database_connection_types = ','.join([db['connectionType'] for db in dashboard['upstreamDatabases']])

            for sheet in sheet_entries:
                rows.append({
                    'Dashboard_name': dashboard_name,
                    'Dashboard_id': dashboard_id,
                    'Sheet_name': sheet['name'],
                    'Sheet_id': sheet['id'],
                    'Datasource_id': datasource_ids,
                    'Datasource_name': datasource_names,
                    'Database_name': database_names,
                    'Database_connectionType': database_connection_types
                })

        print(f"Metadata generated for the Dashboard id {luid}")

    # Convert the collected rows into a DataFrame and save as CSV
    lineage_df = pd.DataFrame(rows)
    print(f'Metadata is generated for all the dashboard and is passed to the main function')
    return lineage_df

def get_lineage(conn,dashboardfile,portalname):
    dashboard_df = pd.read_csv(dashboardfile)
    luids = dashboard_df['id'].tolist()
    all_rows = []

    print(f'Dashboard id is extracted and is passed to the query')
    for _, row in dashboard_df.iterrows():
        luid = row['id']
        project_name = json.loads(row['project'])['name']
        workbook_name = row['workbook_name']
        
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
        print(f'The response is \n {response_json}')
        data = response_json
        dashboards = data['data']['dashboards']

        rows = []

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

                all_rows.append({
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


    lineage_df = pd.DataFrame(all_rows)
    return lineage_df

def logout(conn):
    conn.sign_out()
    print(f'Connection logged out')

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
    lineagefile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_metadata.csv")
    
    connection = connect_to_ts(config,portalname)
    views_df = get_views(connection)
    sites_info = get_sites(connection)
    wb_df = get_workbook(connection)
    get_dashboard_details = get_dashboard(views_df)
    
    views_df.to_csv(viewsfile)
    sites_info.to_csv(sitesfile)
    wb_df.to_csv(workbookfile)
    get_dashboard_details.to_csv(dashboardfile)

    get_lineage_df = get_lineage(connection,dashboardfile,portalname)
    get_lineage_df.to_csv(lineagefile)

    logout(connection)
        
if __name__ == '__main__':
    main()
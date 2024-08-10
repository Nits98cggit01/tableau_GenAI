import pandas as pd
import os
import json

from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying, flatten_dict_column
from typing import List
from config import config

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

def logout(conn):
    conn.sign_out()
    print(f'Connection logged out')

def query_and_save_views1(project_path,csv_path,conn):
    print(f'''Initial check:::
            PROJECT PATH  : {project_path}
            CSV PATH : {csv_path}
            CONN : {conn}
          ''')
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        check_view_id = row['id']
        name = row['name']
        content_url = row['contentUrl']

        # Extract page_name from contentUrl
        page_name = content_url.split('/')[0]

        try:
            # Query the data
            check_view_df = querying.get_view_data_dataframe(conn, view_id=check_view_id)

            # Create the folder if it doesn't exist
            if not os.path.exists(os.path.join(project_path,page_name)):
                os.makedirs(os.path.join(project_path,page_name))

            # Define the file path
            file_path = os.path.join(os.path.join(project_path,page_name,f"{name}.csv"))

            # Save the DataFrame to a CSV file
            check_view_df.to_csv(file_path, index=False)

            print(f"Successfully saved data for view ID {check_view_id} to {file_path}")
        except Exception as e:
            print(f"Error querying or saving data for view ID {check_view_id}: {e}")

def query_and_save_views(project_path,csv_path,conn):
    print(f'''Initial check:::
            PROJECT PATH  : {project_path}
            CSV PATH : {csv_path}
            CONN : {conn}
          ''')
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Ensure the workbook_name column exists and is populated correctly
    if 'wb_name' not in df.columns:
        def extract_project_name(project_json):
            try:
                project_dict = json.loads(project_json.replace("'", '"'))
                return project_dict.get('name', '')
            except json.JSONDecodeError:
                return ''

        df['wb_name'] = df['project'].apply(extract_project_name)

    print(f'Check for the new df')
    df.to_csv(os.path.join(project_path,f'Check.csv'))
    print(f'Check for the new df')

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        check_view_id = row['id']
        name = row['name']
        content_url = row['contentUrl']
        workbook_name = row['wb_name']

        # Extract page_name from contentUrl
        page_name = content_url.split('/')[0]

        try:
            # Query the data
            check_view_df = querying.get_view_data_dataframe(conn, view_id=check_view_id)

            # Define the folder structure
            workbook_folder = os.path.join(project_path, workbook_name)
            page_folder = os.path.join(workbook_folder, page_name)

            # Create the folders if they don't exist
            if not os.path.exists(workbook_folder):
                os.makedirs(workbook_folder)
            if not os.path.exists(page_folder):
                os.makedirs(page_folder)

            # Define the file path
            file_path = os.path.join(page_folder, f"{name}.csv")

            # Save the DataFrame to a CSV file
            check_view_df.to_csv(file_path, index=False)

            print(f"Successfully saved data for view ID {check_view_id} to {file_path}")
        except Exception as e:
            print(f"Error querying or saving data for view ID {check_view_id}: {e}")



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
    connection = connect_to_ts(config,portalname)
    views_df = get_views(connection)
    sites_info = get_sites(connection)
    views_df.to_csv(viewsfile)
    sites_info.to_csv(sitesfile)
    wb_df = get_workbook(connection)
    wb_df.to_csv(workbookfile)
    query_and_save_views(projectpath,viewsfile,connection)
    logout(connection)
        
if __name__ == '__main__':
    main()
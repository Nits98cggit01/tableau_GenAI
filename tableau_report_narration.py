import streamlit as st
import pandas as pd
import os
import json
import openai
from openai import AzureOpenAI
import re
from fpdf import FPDF
from PIL import Image
import base64

from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying, flatten_dict_column
from typing import List

# class PDF(FPDF):
#     def header(self):
#         # Draw the border
#         self.set_draw_color(0, 0, 0)
#         self.rect(5.0, 5.0, 200.0, 287.0)
        
#         # Add the title
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, 'Report narration', 0, 1, 'C')
#         self.ln(10)

#     def chapter_title(self, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, f'Dashboard : {title}', 0, 1, 'C')
#         self.ln(10)

#     def chapter_body(self, body):
#         self.set_font('Arial', '', 12)
#         self.multi_cell(0, 10, body)
#         self.ln()
    
#     def chapter_content(self, body):
#         self.set_font('Arial', '',10)
#         self.multi_cell(0, 10, body)
#         self.ln()

#     def add_image(self, img_path, x=None, y=None, w=0, h=0):
#         self.image(img_path, x, y, w, h)
#         self.ln()

#     def add_table(self, dataframe, title):
#         self.set_font('Arial', 'B', 12)
#         self.cell(0, 10, title, 0, 1, 'L')
#         self.set_font('Arial', '', 10)
#         self.ln(5)

#         # Remove 'Unnamed: 0' column if it exists
#         if 'Unnamed: 0' in dataframe.columns:
#             dataframe = dataframe.drop(columns=['Unnamed: 0'])
        
#         # Table header
#         col_width = (self.w - 20) / len(dataframe.columns)  # Adjusted for page width and margins
#         for col in dataframe.columns:
#             self.cell(col_width, 10, col, border=1)
#         self.ln()
        
#         # Table rows
#         for row in dataframe.itertuples(index=False, name=None):
#             for value in row:
#                 self.cell(col_width, 10, str(value), border=1)
#             self.ln()

class PDF(FPDF):
    def header(self):
        # Draw the border
        self.set_draw_color(0, 0, 0)
        self.rect(5.0, 5.0, 200.0, 287.0)
        if self.page_no() == 1: 
            # Add the title
            self.set_font('Arial', 'B', 12)
            self.cell(0, 5, 'Report narration', 0, 1, 'C')
            self.ln(3)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'reportnarration', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 5, f'Dashboard : {title}', 0, 1, 'C')
        self.ln(2)

    def chapter_heading(self, body):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, body.strip())
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body.strip())

    def add_image(self, img_path, x=None, y=None, w=0, h=0):
        if not x:
            x = (self.w - w) / 2  # Center the image horizontally
        if not y:
            y = self.get_y() + 10  # Add some space above the image
        self.image(img_path, x, y, w, h)
        self.rect(x - 2, y - 2, w + 4, h + 4)  # Add border to the image
        self.ln(h + 10)  # Add space below the image

    def add_table(self, dataframe, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Arial', '', 10)
        self.ln(5)

        # Remove 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in dataframe.columns:
            dataframe = dataframe.drop(columns=['Unnamed: 0'])
        
        # Table header
        col_width = (self.w - 20) / len(dataframe.columns)  # Adjusted for page width and margins
        self.set_font('Arial', 'B', 10)  # Bold for header
        for col in dataframe.columns:
            self.cell(col_width, 10, col, border=1, align='C')
        self.ln()
        
        # Table rows
        self.set_font('Arial', '', 10)  # Normal font for rows
        for row in dataframe.itertuples(index=False, name=None):
            for value in row:
                self.cell(col_width, 10, str(value), border=1)
            self.ln()

def create_pdf(selected_workbook, selected_dashboard,lineage_path,selecteddashboard_folder,dashimg_path):
    # Create instance of FPDF class
    pdf = PDF()

    # Add a page
    pdf.add_page()

    # Add title and subheading
    pdf.chapter_title(selected_dashboard)
    
    # Add dashboard image
    dashboard_image_path = os.path.join(dashimg_path,f'{selected_workbook}-{selected_dashboard}.png')
    if os.path.exists(dashboard_image_path):
        img = Image.open(dashboard_image_path)
        img_w, img_h = img.size
        aspect_ratio = img_h / img_w
        img_width = 180  # Set image width
        img_height = img_width * aspect_ratio  # Maintain aspect ratio
        pdf.add_image(dashboard_image_path, w=img_width, h=img_height)
    
    # Read lineage_path CSV
    lineage_df = pd.read_csv(lineage_path)
    sheet_names = ', '.join(lineage_df['Sheet_name'].dropna().tolist())
    
    # Add Sheet names
    # pdf.chapter_body(f'Sheet names: {sheet_names}')
    pdf.chapter_heading('Sheet name :')
    pdf.chapter_body(sheet_names)
    
    # Iterate through each sheet
    for sheet_name in lineage_df['Sheet_name'].dropna().tolist():
        sheet_path = os.path.join(selecteddashboard_folder, f'{sheet_name}.csv')
        if os.path.exists(sheet_path):
            sheet_df = pd.read_csv(sheet_path)
            
            # Add sheet name and table
            pdf.add_table(sheet_df, f'Sheet name: {sheet_name}')
            
            # Add description
            description = lineage_df.loc[lineage_df['Sheet_name'] == sheet_name, 'Description'].values[0]
            # pdf.chapter_body(f'Description: {description.strip()}')

            pdf.chapter_heading('Description :')
            pdf.chapter_body(description.strip())
    
    # Output the PDF
    output_path = f'{selected_workbook}-{selected_dashboard}_report_narration.pdf'
    pdf.output(output_path)
    print(f'PDF created successfully at {output_path}')

def display_pdf(ROOT_PATH,selected_workbook,selected_dashboard):
    filename = f'{selected_workbook}-{selected_dashboard}_report_narration.pdf'
    pdf_path = os.path.join(ROOT_PATH,filename)  
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf" zoom="100%">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.write("PDF not found for this page.")

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

def save_dashboard_image(view_png, selected_workbook,selected_dashboard,image_dir):
    filename = f'{selected_workbook}-{selected_dashboard}' + ".png"
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

def display_dash(dash_dir,selected_workbook,selected_dashboard):
    filename = f'{selected_workbook}-{selected_dashboard}'
    image_path = os.path.join(dash_dir,f'{filename}.png')
    # Check if the image file exists
    if os.path.exists(image_path):
        # st.image(image_path, caption=f"Displaying {filename}", use_column_width=True)
        st.image(image_path, width=500)
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

def display_analysis(viewtablefolder,filterdashfile):
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

                    if os.path.exists(viewtablefolder):
                        st.write(f'Writing the view analysis table to the folder')
                        if re.search(r'[\\/]', sheet_name):
                            newname = re.sub(r'[\\/]', '', sheet_name)
                            normalized_df.to_csv(os.path.join(viewtablefolder,f'{newname}.csv'))
                        else:
                            normalized_df.to_csv(os.path.join(viewtablefolder,f'{sheet_name}.csv'))
                    else:
                        st.write(f'View analysis tables already exists')
    
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

def openai_response(prompt):
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_KEY"] = "6ad08c8e58dd4de9985b86b1209d9cc2"
    os.environ["OPENAI_API_BASE"] = "https://azureopenaitext.openai.azure.com/"
    os.environ["OPENAI_API_VERSION"] = "2023-09-15-preview"
    openai.api_type = "azure"
    openai.azure_endpoint = "https://azureopenaitext.openai.azure.com/"
    openai.api_version = "2023-09-15-preview"
    openai.api_key = "6ad08c8e58dd4de9985b86b1209d9cc2"
    client = AzureOpenAI(
        api_key = openai.api_key,
        api_version = openai.api_version,
        azure_endpoint = openai.azure_endpoint
    )
    message_text = [{"role":"system","content":f"{prompt}"}]

    response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages = message_text,
    temperature=0.3
    )
    
    return response.choices[0].message.content

def get_prompt(lineagefile):
    # Read the CSV file
    try:
        df = pd.read_csv(lineagefile)
        df['Description'] = ""
        st.write(f'The df opened {lineagefile}')
        
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

            res_output = openai_response(prompt)
            df.at[index, 'Description'] = res_output
            st.write(f'Description for the {index} is fetched')

        df.to_csv(lineagefile, index=False)  
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
    lineagefile = os.path.join(ROOT_PATH,project_folder)
    view_table_folder = os.path.join(ROOT_PATH,project_folder,'view table')
    os.makedirs(view_table_folder, exist_ok=True)


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
    st.title("Tableau Report narration")

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
                        
                        if st.button('Get report'):
                        # Fetch and display dashboard image
                            conn, _ = connect_to_ts(st.session_state.config, st.session_state.env)
                            dashimg_path = os.path.join(ROOT_PATH,project_folder,'Dashboard image')
                            dashboardimgfile = os.path.join(dashimg_path,f'{selected_workbook}-{selected_dashboard}.png')
                            selecteddashboard_folder = os.path.join(view_table_folder,f"{selected_dashboard}")
                            os.makedirs(selecteddashboard_folder,exist_ok=True)
                            lineage_path = os.path.join(lineagefile,f'{portalname}-{selected_dashboard}_metadata.csv')
                            lineage_dash_path = os.path.join(lineagefile,f'{portalname}_metadata.csv')
                            report_path = os.path.join(ROOT_PATH,f'{selected_workbook}-{selected_dashboard}_report_narration.pdf')

                            if os.path.exists(dashboardimgfile):
                                st.write(f'Dashboard image exists')
                                display_dash(dashimg_path,selected_workbook,selected_dashboard)
                            else:
                                st.write(f'Dashboard image extraction')
                                view_png = get_dashboard_img(selected_dashboard,dashboardfile,conn)
                                save_dashboard_image(view_png,selected_workbook,selected_dashboard,dashimg_path)
                                display_dash(dashimg_path,selected_workbook,selected_dashboard)
                                
                            
                            if os.path.exists(filtereddashboardfile):
                                st.write(f'Filtered dashboard file exists')
                                os.remove(filtereddashboardfile)
                                filter_dashboard = filter_dashboard_demo(dashboardfile,selected_workbook,selected_dashboard)
                                filter_dashboard.to_csv(filtereddashboardfile)
                                st.write(f'Filtered dashboard file deleted')
                            else:
                                st.write(f'Filtered dashboard does not exists')
                                filter_dashboard = filter_dashboard_demo(dashboardfile,selected_workbook,selected_dashboard)
                                filter_dashboard.to_csv(filtereddashboardfile)

                            if os.path.exists(lineage_path):
                                st.write(f'Metadata for the {portalname} exists')
                                display_analysis(selecteddashboard_folder,lineage_path)
                                pass
                            else:
                                get_lineage_df = get_lineage(conn,filtereddashboardfile,portalname)
                                get_lineage_df.to_csv(lineage_path)
                                st.write(f'Metadata for the {selected_dashboard} is generated')
                                display_analysis(selecteddashboard_folder,lineage_path)
                                # write a logic to write the excel to the pdf
                                get_prompt(lineage_path)
                                # write a logic to write description to the pdf
                                st.write(f'Description generated for the chosen {selected_dashboard}')

                            if os.path.exists(lineage_dash_path):
                                st.write(f'Metadata for the {portalname} exists')
                                pass
                            else:
                                get_dash_lineage_df = get_lineage(conn,dashboardfile,portalname)
                                get_dash_lineage_df.to_csv(lineage_dash_path)
                                st.write(f'Metadata for the {portalname} is generated')
                                get_prompt(lineage_dash_path)
                                st.write(f'Description generated for the chosen {portalname}')   

                            if os.path.exists(report_path):
                                st.write(f'Report already exists')
                                display_pdf(ROOT_PATH,selected_workbook,selected_dashboard)                   
                            else:
                                st.write('Report creation in progress')
                                create_pdf(selected_workbook,selected_dashboard,lineage_path,selecteddashboard_folder,dashimg_path)  
                                display_pdf(ROOT_PATH,selected_workbook,selected_dashboard)                   
                                            
            else:
                st.session_state.selected_project = None
                st.session_state.selected_workbook = None
                st.session_state.selected_dashboard = None
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
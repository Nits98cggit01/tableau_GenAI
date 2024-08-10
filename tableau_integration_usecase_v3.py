import streamlit as st

import pandas as pd
import os
import json
import openai
import ast
from openai import AzureOpenAI
import re
from fpdf import FPDF
from PIL import Image
import base64

from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying, flatten_dict_column
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI

# Import sklearn for similarity score - Template matching
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class PDF(FPDF):
    def header(self):
        # Draw the border on every page
        self.set_draw_color(0, 0, 0)
        self.rect(5.0, 5.0, 200.0, 287.0)
        
        # Add the title only on the first page
        if self.page_no() == 1:
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Report narration', 0, 1, 'C')
            self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Tableau report narration', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f'Dashboard : {title}', 0, 1, 'C')
        self.ln(10)

    def chapter_heading(self, body):
        self.set_font('Arial', 'B', 12)
        self.multi_cell(0, 10, body.strip())

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 5, body.strip())
        self.ln()

    def add_image(self, img_path, x=None, y=None, w=0, h=0):
        if not x:
            x = (self.w - w) / 2  # Center the image horizontally
        if not y:
            y = self.get_y() + 10  # Add some space above the image
        self.image(img_path, x, y, w, h)
        self.rect(x - 2, y - 2, w + 4, h + 4)  # Add border to the image
        self.ln(h + 10)  # Add space below the image

    # Can be removed later
    # def add_table(self, dataframe, title):
    #     self.set_font('Arial', 'B', 12)
    #     self.cell(0, 10, title, 0, 1, 'L')
    #     self.set_font('Arial', '', 10)
    #     self.ln(5)

    #     # Remove 'Unnamed: 0' column if it exists
    #     if 'Unnamed: 0' in dataframe.columns:
    #         dataframe = dataframe.drop(columns=['Unnamed: 0'])
        
    #     # Table header
    #     col_width = (self.w - 20) / len(dataframe.columns)  # Adjusted for page width and margins
    #     self.set_font('Arial', 'B', 8)  # Smaller font for header
    #     for col in dataframe.columns:
    #         self.cell(col_width, 10, col, border=1, align='C')
    #     self.ln()
        
    #     # Table rows
    #     self.set_font('Arial', '', 8)  # Smaller font for rows
    #     for row in dataframe.itertuples(index=False, name=None):
    #         # Prepare cell contents considering both comma and character length wrapping
    #         cell_contents = []
    #         for value in row:
    #             text = str(value)
    #             wrapped_text = []
    #             for segment in text.split(','):
    #                 while len(segment) > 30:
    #                     wrapped_text.append(segment[:30])
    #                     segment = segment[30:]
    #                 wrapped_text.append(segment)
    #             cell_contents.append(wrapped_text)
            
    #         max_lines = max(len(lines) for lines in cell_contents)
    #         max_height = max_lines * 5
            
    #         # Check if the row fits on the current page, add a new page if needed
    #         if self.get_y() + max_height > self.h - 30:  # 30 is a margin for the footer
    #             self.add_page()
    #             self.set_font('Arial', 'B', 8)  # Repeat table header on new page
    #             for col in dataframe.columns:
    #                 self.cell(col_width, 10, col, border=1, align='C')
    #             self.ln()
    #             self.set_font('Arial', '', 8)  # Set font back to normal for rows
            
    #         # Save the current position
    #         x_before = self.get_x()
    #         y_before = self.get_y()
            
    #         # Write each cell with calculated max_height
    #         for cell_lines in cell_contents:
    #             cell_text = "\n".join(cell_lines)
    #             self.multi_cell(col_width, 5, cell_text, border=1, align='L')
    #             # Move the cursor to the right for the next cell
    #             self.set_xy(x_before + col_width, y_before)
    #             x_before += col_width
            
    #         # Move to the next line after the row
    #         self.ln(max_height)

    # def add_table(self, dataframe, title):
    #     self.set_font('Arial', 'B', 12)
    #     self.cell(0, 10, title, 0, 1, 'L')
    #     self.set_font('Arial', '', 10)
    #     self.ln(5)

    #     if 'Unnamed: 0' in dataframe.columns:
    #         dataframe = dataframe.drop(columns=['Unnamed: 0'])
        
    #     col_width = (self.w - 20) / len(dataframe.columns)  
    #     self.set_font('Arial', 'B', 8)  
    #     for col in dataframe.columns:
    #         self.cell(col_width, 10, col, border=1, align='C')
    #     self.ln()
        
    #     self.set_font('Arial', '', 8)  
    #     for row in dataframe.itertuples(index=False, name=None):
    #         cell_contents = []
    #         for value in row:
    #             text = str(value)
    #             wrapped_text = []
    #             for segment in text.split(','):
    #                 while len(segment) > 30:
    #                     wrapped_text.append(segment[:30])
    #                     segment = segment[30:]
    #                 wrapped_text.append(segment)
    #             cell_contents.append(wrapped_text)
            
    #         max_lines = max(len(lines) for lines in cell_contents)
    #         max_height = max_lines * 5
            
    #         if self.get_y() + max_height > self.h - 30:  
    #             self.add_page()
    #             self.set_font('Arial', 'B', 8)  
    #             for col in dataframe.columns:
    #                 self.cell(col_width, 10, col, border=1, align='C')
    #             self.ln()
    #             self.set_font('Arial', '', 8)  
            
    #         x_before = self.get_x()
    #         y_before = self.get_y()
            
    #         for cell_lines in cell_contents:
    #             cell_text = "\n".join(cell_lines)
                
    #             # Adjust font size only for wrapped cells
    #             if len(cell_lines) > 1:
    #                 self.set_font('Arial', '', 6)
    #             else:
    #                 self.set_font('Arial', '', 8)
                
    #             self.multi_cell(col_width, 5, cell_text, border=1, align='L')
    #             self.set_xy(x_before + col_width, y_before)
    #             x_before += col_width
            
    #         self.ln(max_height)

    def add_table(self, dataframe, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Arial', '', 10)
        self.ln(5)

        if 'Unnamed: 0' in dataframe.columns:
            dataframe = dataframe.drop(columns=['Unnamed: 0'])
        
        col_width = (self.w - 20) / len(dataframe.columns)  
        self.set_font('Arial', 'B', 8)  
        for col in dataframe.columns:
            self.cell(col_width, 10, col, border=1, align='C')
        self.ln()
        
        self.set_font('Arial', '', 8)  
        for row in dataframe.itertuples(index=False, name=None):
            cell_contents = []
            for value in row:
                text = str(value)
                wrapped_text = []
                for segment in text.split(','):
                    while len(segment) > 30:
                        wrapped_text.append(segment[:30])
                        segment = segment[30:]
                    wrapped_text.append(segment)
                cell_contents.append("\n".join(wrapped_text))
            
            max_lines = max(len(lines.split('\n')) for lines in cell_contents)
            max_height = max_lines * 5
            
            if self.get_y() + max_height > self.h - 30:  
                self.add_page()
                self.set_font('Arial', 'B', 8)  
                for col in dataframe.columns:
                    self.cell(col_width, 10, col, border=1, align='C')
                self.ln()
                self.set_font('Arial', '', 8)  
            
            x_before = self.get_x()
            y_before = self.get_y()
            
            for cell_text in cell_contents:
                
                # Adjust font size only for wrapped cells
                if '\n' in cell_text:
                    self.set_font('Arial', '', 6)
                else:
                    self.set_font('Arial', '', 8)
                
                self.multi_cell(col_width, 5, cell_text, border=1, align='L')
                self.set_xy(x_before + col_width, y_before)
                x_before += col_width
            
            self.ln(max_height)

def create_pdf(cg_logo_path,selected_workbook, selected_dashboard, lineage_path, selecteddashboard_folder,dashimg_path):
    # Create instance of FPDF class
    pdf = PDF()

    # Add a page
    pdf.add_page()
    pdf.image(cg_logo_path, 10, 10, 30)
        
    # Add title and subheading
    pdf.chapter_title(selected_dashboard)
    
    # Add dashboard image
    dashimg_path = f'{selected_workbook}-{selected_dashboard}.png'
    if os.path.exists(dashimg_path):
        img = Image.open(dashimg_path)
        img_w, img_h = img.size
        aspect_ratio = img_h / img_w
        img_width = 180  # Set image width
        img_height = img_width * aspect_ratio  # Maintain aspect ratio
        pdf.add_image(dashimg_path, w=img_width, h=img_height)
    
    # Read lineage_path CSV
    lineage_df = pd.read_csv(lineage_path)
    sheet_names = ', '.join(lineage_df['Sheet_name'].dropna().tolist())
    
    # Add Sheet names
    # pdf.chapter_body(f'Sheet names: {sheet_names}')
    pdf.chapter_heading('Sheet names :')
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

def generate_csvs(metadata_file,portalname,portalfolder):
    # Read the metadata.csv file
    df = pd.read_csv(metadata_file)
    # Remove the 'Unnamed: 0' column if it exists
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Master_metadata.csv
    df_master_metadata = df.copy()
    df_master_metadata['Table_name'] = df_master_metadata['Table_name'].str.replace("'", "")
    print(f'Master data : {df_master_metadata}')
    masterfile_path = os.path.join(portalfolder,f'{portalname}_Master_metadata.csv')
    df_master_metadata.to_csv(masterfile_path, index=False)

    # Table_schema.csv
    table_schema_rows = []
    try:
        for idx, row in df.iterrows():
            table_names = row['Table_name'].replace("'", "").split(',')
            column_names = row['Column_name'].split(',')
            dashboard_name = row['Dashboard_name']
            
            table_column_mapping = {}
            for table_name, column_name in zip(table_names, column_names):
                if table_name in table_column_mapping:
                    table_column_mapping[table_name].append(column_name)
                else:
                    table_column_mapping[table_name] = [column_name]
            
            for table_name, columns in table_column_mapping.items():
                table_schema_rows.append([dashboard_name, table_name, ','.join(columns)])
    except:
        print(f'Exception occured')

    df_table_schema = pd.DataFrame(table_schema_rows, columns=['Dashboard_name', 'Table_name', 'Column_name'])
    tableschemapath = os.path.join(portalfolder,f'{portalname}_Table_schema.csv')
    df_table_schema.to_csv(tableschemapath, index=False)

    # MasterData_links.csv
    df_master_data_links = df.copy()
    df_master_data_links['Table_name'] = df_master_data_links['Table_name'].str.replace("'", "")
    df_master_data_links['link'] = df_master_data_links.apply(lambda x: f"{x['Site_name']}\\{x['Project_name']}\\{x['Workbook_name']}\\{x['Dashboard_name']}", axis=1)
    datalinkpath = os.path.join(portalfolder,f'{portalname}_MasterData_links.csv')
    df_master_data_links.to_csv(datalinkpath, index=False)

    print(f'All 3 csvs created')

def create_list_of_strings_from_a_DF(input_df):
    combined_strings = []

    for index, row in input_df.iterrows():
        combined_string = '-'.join(row.astype(str))  # Combine values in the row with '-'
        combined_strings.append(combined_string)

    return combined_string

def create_list_of_strings_of_a_column(df_column):
    combined_strings = [f"{index}: {value}" for index, value in enumerate(df_column)]#, start=1)]
    #combined_strings = df_column.tolist()
    return combined_strings

def result_df(result):
    data_dict = {'Index': [], 'Score': []}  # Initialize an empty dictionary to store data

    # Assuming 'results' contains the list of tuples you've provided

    for item in result:
        # Extract index number and score
        index = int(item[0].page_content.split(':')[0])
        score = item[1]
        
        # Append index number and score to the dictionary
        data_dict['Index'].append(index)
        data_dict['Score'].append(score)
        
    scores_df = pd.DataFrame(data_dict)
    scores_df = scores_df.sort_values('Index')
    return scores_df

def openai_response_1(query,table):
    client = AzureOpenAI(
        api_key = openai.api_key,
        api_version = openai.api_version,
        azure_endpoint = openai.azure_endpoint
    )
    response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[
        {"role": "system", "content": f"""Given a DataFrame: {table} and a query: {query}, 
        Return the top 5 row numbers in {table} that best match {query}.
        The output should be a Python list containing only the matching row numbers.
        The row numbers should be aligned with the {table} index.
        Do not include any explanation or text in the output other than a python list of matching rows.
        """},
        ],
    temperature=0.3
    )
    
    return response.choices[0].message.content

def openai_response_2(query,table):
    client = AzureOpenAI(
        api_key = openai.api_key,
        api_version = openai.api_version,
        azure_endpoint = openai.azure_endpoint
    )
    response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[{"role": "system", "content": f"""Given a DataFrame: {table} and a query: {query}, 
        Return the top 3 row numbers in {table} that best match {query}.
        The output should be a Python list containing only the matching row numbers.
        If there is no match with any row of {table} with {query}, return empty list.
        Do not include any explanation or text in the output other than a python list of matching rows.
        """},
        
        ],
    temperature=0.3
    )
    
    return response.choices[0].message.content

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

# New function added 160724 - Metadata extraction
def filter_workbook_demo(dashboardfile,selected_workbook):
    dash_df = pd.read_csv(dashboardfile)
    desired_workbook = selected_workbook
    st.subheader(f'Desired dashboard is : {desired_workbook}')
    newwbdf = dash_df[(dash_df['workbook_name'] == selected_workbook)]
    return newwbdf

# New function added 160724 - Metadata extraction

def display_dash(dash_dir,selected_workbook,selected_dashboard):
    filename = f'{selected_workbook}-{selected_dashboard}'
    image_path = os.path.join(dash_dir,f'{filename}.png')
    # Check if the image file exists
    if os.path.exists(image_path):
        # st.image(image_path, caption=f"Displaying {filename}", use_column_width=True)
        st.image(image_path, width=1000)
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

def display_analysis1(filterdashfile):
    # Read the CSV file
    df = pd.read_csv(filterdashfile)
    
    # Display the Sheet names in a comma-separated manner
    sheet_names = ', '.join(df['Sheet_name'].unique())
    st.subheader(f"Sheet Names: {sheet_names}")
    
    # Display the unique Datasource names in a comma-separated manner
    datasource_names = ', '.join(df['Datasource_name'].unique())
    st.subheader(f"Unique Datasource Names: {datasource_names}")

    unique_sheet_names = df['Sheet_name'].unique()
    try:
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
    except:
        st.write('Exception occured, few entries are empty')

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
    try:
        for i in range(0, len(unique_sheet_names), 2):
            cols = st.columns(2)
            
            for j in range(2):
                if i + j < len(unique_sheet_names):
                    sheet_name = unique_sheet_names[i + j]
                    sheet_df = df[df['Sheet_name'] == sheet_name]
                    
                    # Normalize the data for the current sheet
                    normalized_data = []
                    for index, row in sheet_df.iterrows():
                        column_names = row['Column_name'].split(',') if pd.notna(row['Column_name']) else ['NaN']
                        table_names = row['Table_name'].split(',') if pd.notna(row['Column_name']) else ['NaN']
                        for column_name, table_name in zip(column_names, table_names):
                            # normalized_data.append({
                            #     'Column Name': column_name.strip(),
                            #     'Table Name': table_name.strip(),
                            #     'Database Name': row['Database_name'],
                            #     'Database_connection': row['Database_connection']
                            # })
                            normalized_data.append({
                                'Database_connection': row['Database_connection'] if pd.notna(row['Database_connection']) else 'NaN',
                                'Database Name': row['Database_name'] if pd.notna(row['Database_name']) else 'NaN',
                                'Table Name': table_name.strip() if pd.notna(table_name) else 'NaN',
                                'Column Name': column_name.strip() if pd.notna(column_name) else 'NaN'
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
    except:
        st.write('Exception occured, few entries are empty')

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

# New function added 030724 - Template matching
def get_dashboard_imgwithid(dashid,conn):
    st.write(f'Inside get_dashboard_imgwithid function : {dashid}')
    try:
        # dashid = f'{dashid}'
        view_png = conn.query_view_image(view_id=f'{dashid}')
        st.write(f'Fetched dashboard {dashid}')
        return view_png
    except:
        st.write('Error processing dash id')
    return None

def get_relavance(query, context_df):
    client = AzureOpenAI(
        api_key = openai.api_key,
        api_version = openai.api_version,
        azure_endpoint = openai.azure_endpoint
    )
    response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[{"role": "system", "content": f"""Given a DataFrame: {context_df} and a query: {query}, 
        Return the rows in {context_df} that best match {query}.
        The output should be a list containing the row numbers of which rows from the input dataframe matches the condition, make sure that the row number should be index value like 1,2,3,... and not the number given in the dataframe.
        There are only 5 rows in the input dataframe and so the row numbers should be between 1 to 5, ensure that the row number starts with 1.
        If there is no match with any row of {context_df} with {query}, return empty list.
        Do not include any explanation or text in the output other than a python list of matching rows.
        """},
        
        ],
    temperature=0.3
    )
    
    return response.choices[0].message.content

def get_dashboard_link(portalname,dashid,dashboard_df):
    with open('serverendpoint.txt', 'r') as file:
        server_endpoint = file.read()
    matching_dashboard = dashboard_df[dashboard_df['id'] == dashid]
    if not matching_dashboard.empty:
        getlink = matching_dashboard.iloc[0]['contentUrl']
        workbookname = getlink.split('/')[0]
        dashboardname = getlink.split('/')[-1]
        dashboardlink = f'{server_endpoint}/#/site/{portalname}/views/{workbookname}/{dashboardname}'
        return dashboardlink
    else:
        st.write('Error occured')
    return None

# New functions added 030724 - Template matching

# New function - Similarity score added 050724 - Template matching
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

# New functions added 170724 - Views, Dashboard, Sites

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

# New functions added 170724

def logout(conn):
    conn.sign_out()
    st.write(f'Connection logged out')

def main():
    st.set_page_config(layout="wide")

    # Set the title and center align it
    st.markdown("<h1 style='text-align: center;'>Tableau Integration with Generative AI</h1>", unsafe_allow_html=True)
    
    portalname = "890Portal"
    project_folder = portalname
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(ROOT_PATH, project_folder)):
        os.makedirs(os.path.join(ROOT_PATH, project_folder))

    portalfolder = os.path.join(ROOT_PATH, project_folder)
    metadatafile = os.path.join(portalfolder, f"{portalname}_metadata.csv")

    projectpath = os.path.join(ROOT_PATH, project_folder)
    viewsfile = os.path.join(ROOT_PATH, project_folder, f"{portalname}_Views.csv")
    sitesfile = os.path.join(ROOT_PATH, project_folder, f"{portalname}_Sites.csv")
    workbookfile = os.path.join(ROOT_PATH, project_folder, f"{portalname}_Workbooks.csv")
    dashboardfile = os.path.join(ROOT_PATH, project_folder, f"{portalname}_Dashboard.csv")
    filtereddashboardfile = os.path.join(ROOT_PATH, project_folder, f"{portalname}_Demo_Dashboards.csv")
    filteredworkbookfile = os.path.join(ROOT_PATH, project_folder, f"{portalname}_Demo_Workbook.csv")
    
    lineagefile = os.path.join(ROOT_PATH, project_folder)
    

    # Placeholder for generating CSVs
    # generate_csvs(metadatafile, portalname, portalfolder)

    masterfile = os.path.join(portalfolder, f'{portalname}_Master_metadata.csv')
    linkfile = os.path.join(portalfolder, f'{portalname}_MasterData_links.csv')
    tableschemafile = os.path.join(portalfolder, f'{portalname}_Table_schema.csv')
    # st.title("Report Matching Assistant")
    view_table_folder = os.path.join(ROOT_PATH,project_folder,'view table')
    os.makedirs(view_table_folder, exist_ok=True)

    cg_logo_path = os.path.join(ROOT_PATH,'capgemini_logo.png')

    cg_logo = Image.open(cg_logo_path)
    if cg_logo is not None:
        st.image(cg_logo, width=100)

    # Initialize session state variables
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'service_selected' not in st.session_state:
        st.session_state.service_selected = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'unique_projects' not in st.session_state:
        st.session_state.unique_projects = []
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None
    if 'selected_workbook' not in st.session_state:
        st.session_state.selected_workbook = None
    if 'selected_dashboard' not in st.session_state:
        st.session_state.selected_dashboard = None
    if 'site_name' not in st.session_state:
        st.session_state.site_name = None

    def go_back():
        st.session_state.service_selected = None
        st.experimental_rerun()

    # Main page
    if not st.session_state.authenticated:
        # Main area
        st.title("Tableau Server Authentication")
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
            with open('serverendpoint.txt','w') as file:
                file.write(server_endpoint)
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
        
    else:
        if st.session_state.service_selected == 'Dashboard metadata extraction' and st.session_state.authenticated:
            if 'selected_project' not in st.session_state:
                st.session_state.selected_project = None
            if 'selected_workbook' not in st.session_state:
                st.session_state.selected_workbook = None
            if 'selected_dashboard' not in st.session_state:
                st.session_state.selected_dashboard = None
            if 'site_name' not in st.session_state:
                st.session_state.site_name = None
            
            st.sidebar.title(st.session_state.site_name)
            st.title("Metadata Extraction")

            tabs = st.sidebar.radio('Metadata', ['Metadata by workbook', 'Metadata by dashboard'])

            if tabs == 'Metadata by dashboard':
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
                                
                                if st.button('Get metadata'):
                                    # Fetch and display dashboard image
                                    conn, _ = connect_to_ts(st.session_state.config, st.session_state.env)
                                    view_png = get_dashboard_img(selected_dashboard,dashboardfile,conn)
                                    dashimg_path = os.path.join(ROOT_PATH,project_folder,'Dashboard image')
                                    save_dashboard_image(view_png,selected_workbook,selected_dashboard,dashimg_path)
                                    display_dash(dashimg_path,selected_workbook,selected_dashboard)
                                    selecteddashboard_folder = os.path.join(view_table_folder,f"{selected_dashboard}")
                                    os.makedirs(selecteddashboard_folder,exist_ok=True)
                                    
                                    if os.path.exists(filtereddashboardfile):
                                        print(f'Filtered dashboard file exists')
                                        os.remove(filtereddashboardfile)
                                        print(f'Filtered dashboard file deleted')
                                    else:
                                        print(f'Filtered dashboard does not exists')

                                        
                                    filter_dashboard = filter_dashboard_demo(dashboardfile,selected_workbook,selected_dashboard)
                                    filter_dashboard.to_csv(filtereddashboardfile)

                                    lineage_path = os.path.join(lineagefile,f'{portalname}_{selected_workbook}-{selected_dashboard}_metadata.csv')
                                    lineage_dash_path = os.path.join(lineagefile,f'{portalname}_metadata.csv')

                                    if os.path.exists(lineage_path):
                                        st.write(f'Metadata for the {portalname} exists')
                                        display_analysis(selecteddashboard_folder,lineage_path)
                                        get_lineage_df = pd.read_csv(lineage_path)
                                        st.table(get_lineage_df)
                                        pass
                                    else:
                                        get_lineage_df = get_lineage(conn,filtereddashboardfile,portalname)
                                        get_lineage_df.to_csv(lineage_path)
                                        st.write(f'Metadata for the {selected_dashboard} is generated')
                                        display_analysis(selecteddashboard_folder,lineage_path)
                                        get_prompt(lineage_path)
                                        st.write(f'Description generated for the chosen {selected_dashboard}')
                                        st.table(get_lineage_df)

                                    
                                    if os.path.exists(lineage_dash_path):
                                        st.write(f'Metadata for the {portalname} exists')
                                        pass
                                    else:
                                        st.write(f'Metadata for the {portalname} does not exist, click the button below to get the metadata of all dashboards')
                                        st.warning("This might take a lot of time...")
                                        if st.button('Get portal metadata'):
                                            get_dash_lineage_df = get_lineage(conn,dashboardfile,portalname)
                                            get_dash_lineage_df.to_csv(lineage_dash_path)
                                            st.write(f'Metadata for the {portalname} is generated')
                                            get_prompt(lineage_dash_path)
                                            st.write(f'Description generated for the chosen {portalname}')                        
                                                        
                    else:
                        st.session_state.selected_project = None
                        st.session_state.selected_workbook = None
                        st.session_state.selected_dashboard = None
                else:
                    st.sidebar.write("Dashboard file could not be loaded. Please check the file path and format.")
            
            elif tabs == 'Metadata by workbook':
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
                            if st.button('Get metadata'):
                                conn, _ = connect_to_ts(st.session_state.config, st.session_state.env)
                                st.write(f'The selected workbook is : {selected_workbook}')
                                filter_workbook = filter_workbook_demo(dashboardfile,selected_workbook)

                                if os.path.exists(filteredworkbookfile):
                                    print(f'Filtered workbook file exists')
                                    os.remove(filteredworkbookfile)
                                    print(f'Filtered workbook file deleted')
                                else:
                                    print(f'Filtered workbook does not exists')
                                # filter_workbook.to_csv(filtereddashboardfile)
                                filter_workbook.to_csv(filteredworkbookfile)

                                workbook_lineage_path = os.path.join(lineagefile,f'{portalname}_{selected_workbook}_metadata.csv')
                                if os.path.exists(workbook_lineage_path):
                                    st.write(f'Metadata for the {portalname} exists')
                                    get_wb_lineage_df = pd.read_csv(workbook_lineage_path)
                                    st.write(get_wb_lineage_df)
                                    pass
                                else:
                                    get_wb_lineage_df = get_lineage(conn,filteredworkbookfile,portalname)
                                    get_wb_lineage_df.to_csv(workbook_lineage_path)
                                    st.write(f'Metadata for the {selected_workbook} is generated')
                                    get_prompt(workbook_lineage_path)

                                    st.table(get_wb_lineage_df)
                                    st.write(f'Description generated for the chosen {selected_workbook}')

                    else:
                        st.session_state.selected_project = None
                        st.session_state.selected_workbook = None
                        st.session_state.selected_dashboard = None
                else:
                    st.sidebar.write("Workbook file could not be loaded. Please check the file path and format.")
            

            if st.button('Back'):
                go_back()

        elif st.session_state.service_selected == 'Template matching' and st.session_state.authenticated:
            st.title('Check for template matching')
            conn, _ = connect_to_ts(st.session_state.config, st.session_state.env)
            query = st.text_input('Query')
            submit_button = st.button('Submit')
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_KEY"] = "6ad08c8e58dd4de9985b86b1209d9cc2"
            os.environ["OPENAI_API_BASE"] = "https://azureopenaitext.openai.azure.com/"
            os.environ["OPENAI_API_VERSION"] = "2023-09-15-preview"
            openai.api_type = "azure"
            openai.azure_endpoint = "https://azureopenaitext.openai.azure.com/"
            openai.api_version = "2023-09-15-preview"
            openai.api_key = "6ad08c8e58dd4de9985b86b1209d9cc2"
            # generate_csvs(metadatafile,masterfile,linkfile,tableschemafile)
            dashimg_path = os.path.join(portalfolder,'Dashboard image')
            cache_file = os.path.join(portalfolder,f'{portalname}_templatematching_cache.csv')
            if os.path.exists(cache_file):
                st.write(f'Template matching file exists')
                cache_df = pd.read_csv(cache_file)
            else:
                cache_df = pd.DataFrame(columns=['query', 'response_df'])
            matched_cache = None

            # Commented this block - 050724
            # if submit_button:
            #     combined_df = pd.read_csv(masterfile)
            #     links_df = pd.read_csv(linkfile)
            #     tables_df = pd.read_csv(tableschemafile)

            #     with st.status("Template matching..."):
            #         st.write("Embeddings creation process started...")
            #         embeddings = AzureOpenAIEmbeddings()
            #         embeddings_input = create_list_of_strings_of_a_column(combined_df['Description'])
            #         st.write("Embeddings created...")
            #         st.write("Document search process started")
            #         document_search = FAISS.from_texts(embeddings_input,embeddings)
            #         st.write("Document search completed")

            #         st.write("Score evaluation process started")
            #         df_list = []
            #         new_df = combined_df.copy()#[['Report_Name', 'Page_Name', 'Title']].copy()
            #         result = document_search.similarity_search_with_score(query,len(embeddings_input))
            #         new_df['Score'] = result_df(result)['Score'].values
            #         df_list.append(new_df)
            #         context_df = df_list[0].nsmallest(10,'Score')
            #         st.write(f"Score evaluation completed")
            #         # st.write(f"\nContext df : {context_df}")
            #         # context_df.to_csv('context_df.csv')

            #         # Existing code 020724 - Nitin S
            #         # st.write(f"Result extraction in progress")
            #         # context_df = context_df.drop('Score',axis='columns')
            #         # context_df = context_df.head(5)
            #         # matching_df = context_df[["Project_name","Workbook_name","Dashboard_name","Sheet_name","Column_name","Table_name","Database_name"]]
            #         # matching_df = matching_df.loc[:, ~matching_df.columns.str.contains('^Unnamed')]
            #         # st.write(f"Result extraction completed")
            #         # End of existing code 020724 - Nitin S

            #         # New code added 030724 - Nitin S
            #         st.write(f"Result extraction in progress")
            #         context_df = context_df.drop('Score',axis='columns')
            #         context_df = context_df.head(5)
            #         # context_df = context_df.rename(columns={'Unnamed: 0': 'Row id'}, inplace=True)
            #         # st.write(context_df)
            #         st.write(f"Result extraction completed")
            #         st.write(f"Fetching final response")
            #         matching_rows = get_relavance(query, context_df)
            #         if len(matching_rows)>0:
            #             result_list = ast.literal_eval(matching_rows)
            #             result_list = [int(x) - 1 for x in result_list]
            #             response_df = context_df.iloc[result_list]
            #             response_df.reset_index(inplace=True)
            #             response_df = response_df[["Project_name","Workbook_name","Dashboard_name","Sheet_name","Column_name","Table_name","Database_name"]]
            #             response_df = response_df.loc[:, ~response_df.columns.str.contains('^Unnamed')]
            #             st.markdown("Based on the given requirement, below are the existing matching templates")
            #             st.write(response_df)
            #             dashboard_names = response_df['Dashboard_name'].tolist()
            #             workbook_names = response_df['Dashboard_name'].tolist()
            #             unique_dashboard_names = list(set(dashboard_names))
            #             unique_workbook_names = list(set(workbook_names))
            #             dashboard_df = pd.read_csv(dashboardfile)
            #             # Iterate through each unique dashboard name and fetch the corresponding ID
                        
            #             for dashboard_name in unique_dashboard_names:
            #                 try:
            #                     matching_dashbd = dashboard_df[dashboard_df['name'] == dashboard_name]
            #                     if not matching_dashbd.empty:
            #                         workbook_name = matching_dashbd.iloc[0]['workbook_name']
            #                         dashboard_id = matching_dashbd.iloc[0]['id']
            #                         # dashimg_path = os.path.join(ROOT_PATH,project_folder,'Dashboard image')
            #                         dash_img_file = os.path.join(dashimg_path,f'{workbook_name}-{dashboard_name}.png')
            #                         if os.path.exists(dash_img_file):
            #                             # st.write(f'Dashboard image exists')
            #                             display_dash(dashimg_path,workbook_name,dashboard_name)
            #                             dashboardlink = get_dashboard_link(portalname,dashboard_id,dashboard_df)
            #                             st.subheader("Dashboard [link](%s)" % dashboardlink)
            #                         else:
            #                             # st.write(f'Dashboard image extraction')
            #                             view_png = get_dashboard_imgwithid(dashboard_id,conn)
            #                             # st.write('Extracted, moving to save it')
            #                             save_dashboard_image(view_png,workbook_name,dashboard_name,dashimg_path)
            #                             display_dash(dashimg_path,workbook_name,dashboard_name)
            #                             # st.write(server_endpoint)
            #                             dashboardlink = get_dashboard_link(portalname,dashboard_id,dashboard_df)
            #                             st.subheader("Dashboard [link](%s)" % dashboardlink)
            #                     else:
            #                         print(f"No ID found for Dashboard Name: {dashboard_name}")
            #                 except IndexError:
            #                     print(f"No ID found for Dashboard Name: {dashboard_name}")

            #             # For loop modified 040724
            #             # for dashboard_name, workbook_name in zip(unique_dashboard_names, unique_workbook_names):
            #             #     try:
            #             #         matching_dashbd = dashboard_df[
            #             #             (dashboard_df['name'] == dashboard_name) & (dashboard_df['workbook_name'] == workbook_name)
            #             #         ]
            #             #         if not matching_dashbd.empty:
            #             #             dashboard_id = matching_dashbd.iloc[0]['id']
            #             #             dash_img_file = os.path.join(dashimg_path, f'{workbook_name}-{dashboard_name}.png')
            #             #             if os.path.exists(dash_img_file):
            #             #                 display_dash(dashimg_path, workbook_name, dashboard_name)
            #             #                 dashboardlink = get_dashboard_link(portalname, dashboard_id, dashboard_df)
            #             #                 st.subheader("Dashboard [link](%s)" % dashboardlink)
            #             #             else:
            #             #                 view_png = get_dashboard_imgwithid(dashboard_id, conn)
            #             #                 save_dashboard_image(view_png, workbook_name, dashboard_name, dashimg_path)
            #             #                 display_dash(dashimg_path, workbook_name, dashboard_name)
            #             #                 dashboardlink = get_dashboard_link(portalname, dashboard_id, dashboard_df)
            #             #                 st.subheader("Dashboard [link](%s)" % dashboardlink)
            #             #         else:
            #             #             print(f"No ID found for Dashboard Name: {dashboard_name}")
            #             #     except IndexError:
            #             #         print(f"No ID found for Dashboard Name: {dashboard_name}")
            
            # End of comment - 050724

            # New logic implemented - 050724
            if submit_button:
                combined_df = pd.read_csv(masterfile)
                links_df = pd.read_csv(linkfile)
                tables_df = pd.read_csv(tableschemafile)

                # changes added 040724
                for index, row in cache_df.iterrows():
                    similarity = calculate_similarity(query, row['query'])
                    st.write(f'Similarity calculated... the score is {similarity}')
                    if similarity > 0.45:  # Assuming a similarity threshold of 0.9
                        matched_cache = row['response_df']
                        break

                if matched_cache is not None:
                    st.write(f'Check for matching entry')
                    response_df = pd.read_json(matched_cache)
                    st.write("Found matching query in cache. Displaying cached response_df:")
                    st.write(response_df)
                    dashboard_names = response_df['Dashboard_name'].tolist()
                    workbook_names = response_df['Dashboard_name'].tolist()
                    unique_dashboard_names = list(set(dashboard_names))
                    unique_workbook_names = list(set(workbook_names))
                    dashboard_df = pd.read_csv(dashboardfile)
                    # Iterate through each unique dashboard name and fetch the corresponding ID
                    for dashboard_name in unique_dashboard_names:
                    # for dashboard_name, workbook_name in zip(unique_dashboard_names, unique_workbook_names):
                        try:
                            matching_dashbd = dashboard_df[dashboard_df['name'] == dashboard_name]
                            # matching_dashbd = dashboard_df[
                            #     (dashboard_df['name'] == dashboard_name) & (dashboard_df['workbook_name'] == workbook_name)
                            # ]
                            if not matching_dashbd.empty:
                                workbook_name = matching_dashbd.iloc[0]['workbook_name']
                                dashboard_id = matching_dashbd.iloc[0]['id']
                                # dashimg_path = os.path.join(ROOT_PATH,project_folder,'Dashboard image')
                                dash_img_file = os.path.join(dashimg_path,f'{workbook_name}-{dashboard_name}.png')
                                if os.path.exists(dash_img_file):
                                    st.write(f'Dashboard image exists')
                                    display_dash(dashimg_path,workbook_name,dashboard_name)
                                    dashboardlink = get_dashboard_link(portalname,dashboard_id,dashboard_df)
                                    st.subheader("Dashboard [link](%s)" % dashboardlink)
                                else:
                                    st.write(f'Dashboard image extraction')
                                    view_png = get_dashboard_imgwithid(dashboard_id,conn)
                                    st.write('Extracted, moving to save it')
                                    save_dashboard_image(view_png,workbook_name,dashboard_name,dashimg_path)
                                    display_dash(dashimg_path,workbook_name,dashboard_name)
                                    dashboardlink = get_dashboard_link(portalname,dashboard_id,dashboard_df)
                                    st.subheader("Dashboard [link](%s)" % dashboardlink)
                                    # get_link(server_endpoint,selected_dashboard,selected_workbook,dashboard_df)
                            else:
                                print(f"No ID found for Dashboard Name: {dashboard_name}")
                        except IndexError:
                            print(f"No ID found for Dashboard Name: {dashboard_name}")
                # changes done 040724
                else:
                    with st.status("Template matching..."):
                        st.write("Embeddings creation process started...")
                        embeddings = AzureOpenAIEmbeddings()
                        embeddings_input = create_list_of_strings_of_a_column(combined_df['Description'])
                        st.write("Embeddings created...")
                        st.write("Document search process started")
                        st.write("This process might take few minutes, stay tuned...")
                        document_search = FAISS.from_texts(embeddings_input,embeddings)
                        st.write("Document search completed")
                        st.write("Score evaluation process started")
                        df_list = []
                        new_df = combined_df.copy()#[['Project_name', 'Workbook_name', 'Title']].copy()
                        result = document_search.similarity_search_with_score(query,len(embeddings_input))
                        new_df['Score'] = result_df(result)['Score'].values
                        df_list.append(new_df)
                        context_df = df_list[0].nsmallest(10,'Score')
                        st.write(f"Score evaluation completed")
                        # st.write(f"\nContext df : {context_df}")
                        # context_df.to_csv('context_df.csv')
                        st.write(f"Result extraction in progress")
                        context_df = context_df.drop('Score',axis='columns')
                        context_df = context_df.head(5)
                        # context_df = context_df.rename(columns={'Unnamed: 0': 'Row id'}, inplace=True)
                        # st.write(context_df)
                        st.write(f"Result extraction completed")
                        st.write(f"Fetching final response")
                        matching_rows = get_relavance(query, context_df)
                        # st.write("Based on the given query and input dataframe the below are the matching rows :")
                        # st.write(matching_rows)
                    
                    if len(matching_rows)>0:
                        result_list = ast.literal_eval(matching_rows)
                        result_list = [int(x) - 1 for x in result_list]
                        response_df = context_df.iloc[result_list]
                        response_df.reset_index(inplace=True)
                        st.markdown("Based on the given requirement, below are the existing matching templates")
                        st.write(response_df)
                        dashboard_names = response_df['Dashboard_name'].tolist()
                        workbook_names = response_df['Dashboard_name'].tolist()
                        unique_dashboard_names = list(set(dashboard_names))
                        unique_workbook_names = list(set(workbook_names))
                        dashboard_df = pd.read_csv(dashboardfile)
                        # Iterate through each unique dashboard name and fetch the corresponding ID
                        for dashboard_name in unique_dashboard_names:
                        # for dashboard_name, workbook_name in zip(unique_dashboard_names, unique_workbook_names):
                            try:
                                matching_dashbd = dashboard_df[dashboard_df['name'] == dashboard_name]
                                # matching_dashbd = dashboard_df[
                                #     (dashboard_df['name'] == dashboard_name) & (dashboard_df['workbook_name'] == workbook_name)
                                # ]
                                if not matching_dashbd.empty:
                                    workbook_name = matching_dashbd.iloc[0]['workbook_name']
                                    dashboard_id = matching_dashbd.iloc[0]['id']
                                    # dashimg_path = os.path.join(ROOT_PATH,project_folder,'Dashboard image')
                                    dash_img_file = os.path.join(dashimg_path,f'{workbook_name}-{dashboard_name}.png')
                                    if os.path.exists(dash_img_file):
                                        st.write(f'Dashboard image exists')
                                        display_dash(dashimg_path,workbook_name,dashboard_name)
                                    else:
                                        st.write(f'Dashboard image extraction')
                                        view_png = get_dashboard_imgwithid(dashboard_id,conn)
                                        st.write('Extracted, moving to save it')
                                        save_dashboard_image(view_png,workbook_name,dashboard_name,dashimg_path)
                                        display_dash(dashimg_path,workbook_name,dashboard_name)
                                        # get_link(server_endpoint,selected_dashboard,selected_workbook,dashboard_df)
                                else:
                                    print(f"No ID found for Dashboard Name: {dashboard_name}")
                            except IndexError:
                                print(f"No ID found for Dashboard Name: {dashboard_name}")

                        # response_df.to_csv('template_response.csv')
                        # st.write(dashboard_names)

                    else:
                        st.write(f"Based on the given requirement, No matching reports, tables and columns found")
                    
                    # changes added 040724
                    new_cache_entry = {
                        'query': query,
                        'response_df': response_df.to_json()
                    }
                    cache_df = pd.concat([cache_df, pd.DataFrame([new_cache_entry])], ignore_index=True)
                    cache_df.to_csv(cache_file, index=False)
            # End of submit_button block - 050724

            if st.button('Back'):
                go_back()

        elif st.session_state.service_selected == 'Report narration' and st.session_state.authenticated:
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
                                
                                if st.button('Get narration'):
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
                                        # st.write(f'Metadata for the {portalname} exists')
                                        # display_analysis(selecteddashboard_folder,lineage_path)
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
                                        create_pdf(cg_logo_path,selected_workbook,selected_dashboard,lineage_path,selecteddashboard_folder,dashimg_path)  
                                        display_pdf(ROOT_PATH,selected_workbook,selected_dashboard)                   
                                                    
                    else:
                        st.session_state.selected_project = None
                        st.session_state.selected_workbook = None
                        st.session_state.selected_dashboard = None
                else:
                    st.sidebar.write("Dashboard file could not be loaded. Please check the file path and format.")
            

            if st.button('Back'):
                go_back()

        else:
            with st.container():
                st.markdown("### Our Services")
                service_selected = st.selectbox("Choose a service", 
                                                 ['Choose an option', 'Dashboard metadata extraction', 'Template matching', 'Report narration'])
                if service_selected != 'Choose an option':
                    st.session_state.service_selected = service_selected
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
import openai
import pandas as pd 
import os
from config import *

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

def get_prompt(dashboardfile,lineagefile):
    # Read the CSV file
    try:
        dash_df = pd.read_csv(dashboardfile)
        print(f'File read \n {dash_df}')
        # description = []
        # Create a list to store the prompts
        prompts = []
        
        # Iterate through each row and create the prompt
        for index, row in dash_df.iterrows():
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
            metadatadf = dash_df.copy()
            res_output = call_openai(prompt)
            metadatadf['Description'] = res_output
        
        # Save the updated DataFrame back to the CSV file
        metadatadf.to_csv(lineagefile, index=False)

        prompts.append(prompt)
        print(f'The description is updated for the metadata')
    except:
        print(f'Error occured...Cannot update the description')
    return prompts

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
    lineagefile = os.path.join(ROOT_PATH,project_folder,f"{portalname}_Master_metadata.csv")

    get_prompt(dashboardfile,lineagefile)

if __name__ == "__main__":
    main()

import pandas as pd
import os
import ast
import openai
from openai import AzureOpenAI
import pandas as pd
import streamlit as st
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from tableau_api_lib import TableauServerConnection

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# from config import *

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

def save_dashboard_image(view_png, selected_workbook,selected_dashboard,image_dir):
    filename = f'{selected_workbook}-{selected_dashboard}' + ".png"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    image_path = os.path.join(image_dir, filename)
    with open(image_path, 'wb') as v:
        v.write(view_png.content)

def display_dash(dash_dir,selected_workbook,selected_dashboard):
    st.write(f'Inside the displaydash function')
    filename = f'{selected_workbook}-{selected_dashboard}'
    image_path = os.path.join(dash_dir,f'{filename}.png')
    st.write(f'Img path is : {image_path}')
    # Check if the image file exists
    if os.path.exists(image_path):
        # st.image(image_path, caption=f"Displaying {filename}", use_column_width=True)
        st.image(image_path, width=1000)
    else:
        st.write(f"Image file '{filename}' not found in '{dash_dir}'.")

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

def openai_response(query,table):
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

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]


def main():
    st.set_page_config(layout="wide")
    config = {
        '890Portal': {
            "server": "http://10.100.252.218/",
            "api_version": "3.19",
            "personal_access_token_name": 'rest_token',
            "personal_access_token_secret": 'zL6zdQRgS9WQYH24aGR4JQ==:7yDTI0BCA6XlNxOm24YDN6S3B6673Nyn',
            "site_name": '890Portal',
            "site_url": '890Portal'
        }
    }
    conn = TableauServerConnection(config_json=config, env='890Portal')
    conn.sign_in()

    st.write("Server connected")
    
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_KEY"] = "6ad08c8e58dd4de9985b86b1209d9cc2"
    os.environ["OPENAI_API_BASE"] = "https://azureopenaitext.openai.azure.com/"
    os.environ["OPENAI_API_VERSION"] = "2023-09-15-preview"
    openai.api_type = "azure"
    openai.azure_endpoint = "https://azureopenaitext.openai.azure.com/"
    openai.api_version = "2023-09-15-preview"
    openai.api_key = "6ad08c8e58dd4de9985b86b1209d9cc2"

    portalname = "890Portal"
    project_folder = portalname
    root = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = os.path.dirname(root)
    st.write(f'ROOT PATH : {root}')
    portalfolder = os.path.join(root,project_folder)
    st.write(f'Portal folder : {portalfolder}')
    metadatafile = os.path.join(portalfolder,f"{portalname}_metadata.csv")
    cache_file = os.path.join(portalfolder,f'{portalname}_templatematching_cache.csv')


    generate_csvs(metadatafile,portalname,portalfolder)

    masterfile = os.path.join(portalfolder,f'{portalname}_Master_metadata.csv')
    linkfile = os.path.join(portalfolder,f'{portalname}_MasterData_links.csv')
    tableschemafile = os.path.join(portalfolder,f'{portalname}_Table_schema.csv')
    dashboardfile = os.path.join(portalfolder, f"{portalname}_Dashboard.csv")
    dashimg_path = os.path.join(portalfolder,'Dashboard image')

    st.write(f'All support files loaded')
    st.title("Report Matching Assistant")
    #left_column, right_column = st.columns(2)
    query = st.text_input('Enter a Query')
    submit_button = st.button('Submit')
    # generate_csvs(metadatafile,masterfile,linkfile,tableschemafile)
    # changes added 040724
    if os.path.exists(cache_file):
        st.write(f'Template matching file exists')
        cache_df = pd.read_csv(cache_file)
    else:
        cache_df = pd.DataFrame(columns=['query', 'response_df'])
    matched_cache = None
    # changes done 040724

    response_df = []
    
    if submit_button:
        combined_df = pd.read_csv(masterfile)
        links_df = pd.read_csv(linkfile)
        tables_df = pd.read_csv(tableschemafile)

        # changes added 040724
        for index, row in cache_df.iterrows():
            similarity = calculate_similarity(query, row['query'])
            st.write(f'Similarity calculated... the score is {similarity}')
            if similarity > 0.4:  # Assuming a similarity threshold of 0.9
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
            # changes incorporated 040724

    conn.sign_out()   
                
if __name__ == '__main__':
    main()

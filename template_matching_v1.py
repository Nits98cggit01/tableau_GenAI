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

# from config import *

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

def main():
    st.set_page_config(layout="wide")
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
    
    generate_csvs(metadatafile,portalname,portalfolder)

    masterfile = os.path.join(portalfolder,f'{portalname}_Master_metadata.csv')
    linkfile = os.path.join(portalfolder,f'{portalname}_MasterData_links.csv')
    tableschemafile = os.path.join(portalfolder,f'{portalname}_Table_schema.csv')
    st.write(f'All support files loaded')
    st.title("Report Matching Assistant")
    #left_column, right_column = st.columns(2)
    query = st.text_input('Enter a Query')
    submit_button = st.button('Submit')
    # generate_csvs(metadatafile,masterfile,linkfile,tableschemafile)
    if submit_button:
        combined_df = pd.read_csv(masterfile)
        links_df = pd.read_csv(linkfile)
        tables_df = pd.read_csv(tableschemafile)

        embeddings = AzureOpenAIEmbeddings()
        embeddings_input = create_list_of_strings_of_a_column(combined_df['Description'])
        document_search = FAISS.from_texts(embeddings_input,embeddings)

        df_list = []
        new_df = combined_df.copy()#[['Project_name', 'Workbook_name', 'Title']].copy()
        result = document_search.similarity_search_with_score(query,len(embeddings_input))
        new_df['Score'] = result_df(result)['Score'].values
        df_list.append(new_df)
        context_df = df_list[0].nsmallest(10,'Score')

        if context_df["Score"].iloc[0] <= 0.45:
            print("Context_df", context_df)
            response_rows = openai_response(query, context_df.drop('Score',axis='columns'))
            try:
                result_list = ast.literal_eval(response_rows)
                matching_rows = context_df.loc[result_list]#.reset_index(drop=True)
                if len(matching_rows)>0:
                    matching_rows_with_links = pd.merge(matching_rows, links_df[['Project_name', 'Workbook_name', 'Dashboard_name', 'Sheet_name' ,'Links']], 
                            on=['Project_name', 'Workbook_name', 'Dashboard_name', 'Sheet_name'], how='left')
                    matching_rows_with_links.drop(["Number of columns", "Column_name", "Table_name", "Description"], axis = 1, inplace = True)
                    st.write("")
                    st.markdown("Based on the given requirement, below are the existing matching templates")
                    matching_rows_with_links["Score"] = 1 - matching_rows_with_links["Score"]
                    #left_area = left_column.write(matching_rows_with_links.drop("Links", axis = 1))
                    st.write(matching_rows_with_links)
                else:
                    st.write("Based on the given requirement, No matching reports, tables and columns found")
                #right_area = right_column.write(matching_rows_with_links["Links"].iloc[0])
            except:
                st.write("Based on the given requirement, No matching reports, tables and columns found")
        else:
            response_rows_2 = openai_response_2(query, tables_df)
            try:
                result_list_2 = ast.literal_eval(response_rows_2)
                matching_rows = tables_df.loc[result_list_2]#.reset_index(drop=True)
                if len(matching_rows)>0:
                    st.write("Based on the given requirement, No matching report found and below are the few suggestion tables")
                    st.write(matching_rows.reset_index(drop = True))
                else:
                    st.write("Based on the given requirement, No matching reports, tables and columns found")
            except:
                st.write("Based on the given requirement, No matching reports, tables and columns found")


if __name__ == '__main__':
    main()
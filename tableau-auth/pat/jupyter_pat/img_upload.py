import streamlit as st
import os
import pandas as pd


def display_img(img_dir, filename):
    image_path = os.path.join(img_dir,f'{filename}.png')
    # Check if the image file exists
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Displaying {filename}", use_column_width=True)
        # st.image(image_path, width=300)
    else:
        st.write(f"Image file '{filename}' not found in '{img_dir}'.")

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


# Example usage:
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    portalname = "890Portal"
    project_folder = portalname
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    dashboardname = 'Comparison of People Metrics'
    dash = 'Comparison of People Metrics.png'
    st.title(f'{dashboardname}')
    dashboarddf = os.path.join(ROOT_PATH,portalname,f'{portalname}_metadata.csv')
    # filename = f'{dash}.png'
    img_dir = os.path.join(ROOT_PATH,portalname,'Dashboard image')  # Replace with the actual path to your image folder
    # display_img(img_dir, dashboardname)
    display_analysis(dashboarddf)
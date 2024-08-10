import streamlit as st

# server_endpoint = st.text_input("Server Endpoint")
# pat_name = st.text_input("PAT Name")
# pat_key = st.text_input("PAT Key", type="password")
# site_name = st.text_input("Site Name")
# env = st.text_input("Environment name")
# signin_button = st.form_submit_button("Sign In")

def main():
    # Set the layout to wide
    st.set_page_config(layout="wide")
    # Title
    st.markdown("<h1 style='text-align: center;'>Tableau Integration with Generative AI</h1>", unsafe_allow_html=True)

    # Session state management
    if 'auth' not in st.session_state:
        st.session_state.auth = False
    if 'page' not in st.session_state:
        st.session_state.page = 'signin'
    if 'selected_service' not in st.session_state:
        st.session_state.selected_service = None

    # Signin Page
    if st.session_state.page == 'signin':
        with st.form(key='signin_form'):
            server_endpoint = st.text_input("Server Endpoint")
            pat_name = st.text_input("PAT Name")
            pat_key = st.text_input("PAT Key", type="password")
            site_name = st.text_input("Site Name")
            env = st.text_input("Environment name")
            signin_button = st.form_submit_button("Sign In")

            if signin_button:
                # Authentication logic here
                st.session_state.auth = True
                st.session_state.page = 'services'
                st.success("Authentication successful!")
            
        
    # Services Page
    if st.session_state.page == 'services' and st.session_state.auth:
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum vestibulum.")
        with st.form(key='services_form'):
            st.header("Our services")
            service = st.selectbox("Choose a service", ["Choose an option", "Dashboard metadata extraction", "Template matching", "Report narration"])
            service_button = st.form_submit_button("Submit")

            if service_button and service != "Choose an option":
                st.session_state.selected_service = service
                st.session_state.page = service

    # Dashboard Metadata Extraction Page
    if st.session_state.page == 'Dashboard metadata extraction':
        with st.sidebar:
            project_name = st.selectbox("Project Name", ["Choose an option", "Project A", "Project B", "Project C"])
            workbook_name = st.selectbox("Workbook Name", ["Choose an option", "Workbook A", "Workbook B", "Workbook C"])
            dashboard_name = st.selectbox("Dashboard Name", ["Choose an option", "Dashboard A", "Dashboard B", "Dashboard C"])

        if project_name != "Choose an option" and workbook_name != "Choose an option" and dashboard_name != "Choose an option":
            st.write(f"You have chosen the Project Name - {project_name}; the Workbook Name - {workbook_name} and the Dashboard - {dashboard_name}")

    # Template Matching Page
    if st.session_state.page == 'Template matching':
        query = st.text_area("Query")
        check_button = st.button("Check")

        if check_button:
            st.write(f"Query: {query}")

    # Report Narration Page
    if st.session_state.page == 'Report narration':
        with st.sidebar:
            project_name = st.selectbox("Project Name", ["Choose an option", "Project A", "Project B", "Project C"])
            workbook_name = st.selectbox("Workbook Name", ["Choose an option", "Workbook A", "Workbook B", "Workbook C"])
            dashboard_name = st.selectbox("Dashboard Name", ["Choose an option", "Dashboard A", "Dashboard B", "Dashboard C"])

        if project_name != "Choose an option" and workbook_name != "Choose an option" and dashboard_name != "Choose an option":
            st.write(f"You have chosen the Project Name - {project_name}; the Workbook Name - {workbook_name} and the Dashboard - {dashboard_name}")

if __name__ == "__main__":
    main()
import streamlit as st
import tempfile
from preprocessing.phi import extract_text_from_pdf, detect_phi_entities, scrub_phi_from_text
from models.deep_learning_summarizer import summarize_deep_learning

st.set_page_config(page_title="MediScrub", page_icon="", layout="wide")

def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'summarization_results' not in st.session_state:
        st.session_state.summarization_results = {}

    if not st.session_state.authenticated:
        login_page()
    else:
        main_dashboard()

# Taking user login and verifying credentials
def login_page():
    st.title("MediScrub")
    st.markdown("---")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            valid_credentials = {"doctor": "doc123", "frontdesk": "front123"}
            
            if username in valid_credentials and valid_credentials[username] == password:
                st.session_state.authenticated = True
                st.session_state.user_role = "Doctor" if username == "doctor" else "Front Desk Staff"
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

# Main UI to upload the pdf docs and generate the summary
def main_dashboard():
    st.title("MediScrub")
    
    st.header("Document Upload and Summarization")
    
    uploaded_files = st.file_uploader("Upload documents",type=['pdf'],accept_multiple_files=True,help="Select PDF files to process")
    
    if uploaded_files:
        st.write(f"{len(uploaded_files)} document(s) uploaded")
        
        if st.button("Deep Learning Summaries", type="primary"):
            with st.spinner("Processing documents"):
                results = []
                
                for uploaded_file in uploaded_files:
                    if uploaded_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_file_path = tmp_file.name
                        
                        full_text = extract_text_from_pdf(temp_file_path)
                        
                        if st.session_state.user_role == "Front Desk Staff":
                            phi_entities = detect_phi_entities(full_text)
                            processed_text = scrub_phi_from_text(full_text, phi_entities)
                            summary = summarize_deep_learning(processed_text)
                        else:
                            summary = summarize_deep_learning(full_text)
                        
                        results.append({'filename': uploaded_file.name,'summary': summary, 'role': st.session_state.user_role, 'phi_removed': st.session_state.user_role == "Front Desk Staff"})
                
                st.session_state.summarization_results = results
                
                if results:
                    st.success("Summaries generated successfully!")
                    display_results()
                else:
                    st.error("No documents were successfully processed.")

# Display the summary result in UI
def display_results():
    results = st.session_state.summarization_results
    st.header("Deep Learning Summarization Results")
    
    for result in results:
        with st.expander(f"{result['filename']}", expanded=True):
            st.subheader("Summary")
            st.write(result['summary'])
            st.markdown("---")

if __name__ == "__main__":
    main()

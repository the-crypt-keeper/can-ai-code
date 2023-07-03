import json
import streamlit as st
import sys

def load_analysis_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_analysis_data(data):
    tests = data['tests']
    models = data['models']
   
    for test_name, test_data in tests.items():
        task = test_data['task']
        language = test_data['language']

        with st.expander(f"{test_name}: {task}"):
            columns = st.columns(len(models))
            if 'summary' in test_data:
                st.markdown("**Analysis**: "+test_data['summary'])
            
        for model_id, model_result in test_data['results'].items():
            model_info = models[int(model_id)]

            model_result['passing_tests'] = '\n\n'.join([f":blue[{x}]" for x in model_result['passing_tests'].split('\n') if x.strip() != ''])
            model_result['failing_tests'] = '\n\n'.join([f":red[{x}]" for x in model_result['failing_tests'].split('\n') if x.strip() != ''])

            with columns[int(model_id)]:
                st.subheader(f"{model_info['model']}")
                st.markdown(f"**Summary:** {model_result['check_summary']}")

                st.code(model_result['code'], language=language)
                
                passcol,failcol=st.columns(2)
                passcol.markdown(f"**Passing Tests:**\n\n{model_result['passing_tests']}")
                failcol.markdown(f"**Failing Tests:**\n\n{model_result['failing_tests']}")
                    
st.set_page_config(page_title='Analysis Explorer', layout="wide")
st.markdown("""
        <style>
            .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 3.5rem;
                }
        </style>
        """, unsafe_allow_html=True)

data = load_analysis_file(sys.argv[1])
st.header(sys.argv[2])
display_analysis_data(data)

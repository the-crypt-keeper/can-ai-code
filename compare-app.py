import json
import streamlit as st
import sys
import glob

def load_analysis_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_analysis_data(data):
    tests = data['tests']
    models_list = data['models']
    models = {}
    for idx, model_info in enumerate(models_list):
        models[model_info['id']] = model_info

    # summary table
    summary_cols = st.columns(len(models_list))
    for model_id, model_info in models.items():
        with summary_cols[model_info['idx']]:
            st.subheader(f"{model_info['short_name']}")
            st.progress(model_info['passed']/model_info['total'], f"{model_info['passed']}/{model_info['total']}")
   
    for test_name, test_data in tests.items():
        task = test_data['task']
        language = test_data['language']

        with st.expander(f"{test_name}: {task}"):
            columns = st.columns(len(models))
            if 'summary' in test_data:
                st.markdown("**Analysis**: "+test_data['summary'])
            
        for model_id, model_result in test_data['results'].items():
            model_info = models[model_id]

            model_result['passing_tests'] = '\n\n'.join([f":blue[{x}]" for x in model_result['passing_tests'].split('\n') if x.strip() != ''])
            model_result['failing_tests'] = '\n\n'.join([f":red[{x}]" for x in model_result['failing_tests'].split('\n') if x.strip() != ''])

            with columns[model_info['idx']]:
                st.subheader(f"{model_info['short_name']}")
                st.markdown(f"**Summary:** {model_result['check_summary']}")
                st.write(model_result['answer'])
                st.write('---')
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

files = glob.glob('compare/*.json')
data = [json.load(open(file,'r')) for file in files]
titles = [x['config']['title'] for x in data]
options = st.selectbox('Select Analysis', titles)
idx = titles.index(options)
display_analysis_data(data[idx])

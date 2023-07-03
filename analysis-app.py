import json
import streamlit as st

def load_analysis_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_analysis_data(data):
    tests = data['tests']
    models = data['models']
   
    for test_name, test_data in tests.items():
        task = test_data['prompt'].split('\n')[2]

        with st.expander(f"{test_name}: {task}"):
            columns = st.columns(len(models))
            st.markdown("**Analysis**: "+test_data['summary'])
            
        for model_id, model_result in test_data['models'].items():
            model_info = models[int(model_id)]

            model_result['passing_tests'] = '\n'.join([f"* :blue[{x}]" for x in model_result['passing_tests'].split('\n') if x.strip() != ''])
            model_result['failing_tests'] = '\n'.join([f"* :red[{x}]" for x in model_result['failing_tests'].split('\n') if x.strip() != ''])

            with columns[int(model_id)]:
                st.subheader(f"{model_info['model']}")
                st.markdown(f"**Summary:** {model_result['check_summary']}")

                st.code(model_result['code'], language='javascript')
                
                passcol,failcol=st.columns(2)
                passcol.markdown(f"**Passing Tests:**\n\n{model_result['passing_tests']}")
                failcol.markdown(f"**Failing Tests:**\n\n{model_result['failing_tests']}")
                    
st.set_page_config(page_title='AutoAnalysis Explorer', layout="wide")
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
st.header('Orca-Mini 3B vs 7B vs 13B [JavaScript] Auto-Analysis')
# Load analysis data from analysis.json
data = load_analysis_file('autoanalysis-orca-mini-javascript.json')

# Display the analysis data
display_analysis_data(data)

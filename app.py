import streamlit as st
import pandas as pd
import glob
import json
import os

from prepare import load_questions

def read_ndjson(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data

def load_data():
    files = glob.glob('results/eval*.ndjson')
    data = {}
    for file in files:
        tags = os.path.basename(file).replace('.ndjson', '').split('_')

        if len(tags) == 9:
            tags = tags[0:8] + tags[10:10]
        elif len(tags) == 7:
            tags += [0]
        elif len(tags) != 8:
            print('Skipping', file)
            continue

        results = read_ndjson(file)

        langs = tags[2].split('-')
        for lang in langs:
            new_tags = tags.copy()
            new_tags[2] = lang
            data[file+'-'+lang] = {
                'tags': new_tags,
                'results': list(filter(lambda x: x.get('language') == lang, results))
            }

    return data

def calculate_summary(data):
    summary = []
    for file, info in data.items():
        res = info['results']
        passed = sum(x['passed'] for x in res)
        total = sum(x['total'] for x in res)
        summary.append(info['tags'] + [passed, total])
    sumdf = pd.DataFrame(summary, columns=['Eval', 'Interview', 'Languages', 'Template', 'TemplateOut', 'Params', 'Model', 'Timestamp', 'Passed', 'Total'])
    sumdf = sumdf[['Languages','Model','Params','Template','Passed','Total']]
    sumdf['Score'] = sumdf['Passed'] / sumdf['Total']
    sumdf.drop('Total', axis=1, inplace=True)
    return sumdf.sort_values(by='Passed', ascending=False)

@st.cache_data
def load_and_prepare_data():
    data = load_data()
    summary = calculate_summary(data)
    return data, summary

def main():
    st.set_page_config(page_title='CanAiCode Explorer', layout="wide")
    st.markdown("""
            <style>
                .block-container {
                        padding-top: 1rem;
                        padding-bottom: 0rem;
                        padding-left: 3rem;
                        padding-right: 3.5rem;
                    }
            </style>
            """, unsafe_allow_html=True)
    
    data, summary = load_and_prepare_data()

    #st.sidebar.title('CanAiCode? 🤔')
    #st.sidebar.markdown('A visual tool to explore the results of [CanAiCode](https://github.com/the-crypt-keeper/can-ai-code)')

    tabs = ['Summary', 'Explore', 'Compare']
    selected_tab = 'Summary' #st.sidebar.radio('', tabs)

    if selected_tab == 'Summary':
        st.title('CanAiCode Leaderboard 🏆')
        st.markdown('A visual tool to explore the results of [CanAiCode](https://github.com/the-crypt-keeper/can-ai-code)')
        
        column_config={
            "Score": st.column_config.ProgressColumn(
                label="Score",
                help="Can it code?",
                format="%.3f",
                min_value=0,
                max_value=1,
            )
        }
        column_order=("Model", "Params", "Template", "Passed", "Score")

        mode = st.radio(label='View',options=['Side by Side','Python','JavaScript'], horizontal=True, label_visibility='hidden')
        if mode == 'Side by Side':
            pyct, jsct = st.columns(2)
        else:
            pyct = st.container() if mode == 'Python' else None
            jsct = st.container() if mode == 'JavaScript' else None

        if pyct is not None:
            with pyct:
                st.subheader('Python')
                st.dataframe(summary[summary['Languages'] == 'python'], use_container_width=True, column_config=column_config, column_order=column_order, hide_index=True, height=700)

        if jsct is not None:
            with jsct:
                st.subheader('JavaScript')
                st.dataframe(summary[summary['Languages'] == 'javascript'], use_container_width=True, column_config=column_config, column_order=column_order, hide_index=True, height=700)

    elif selected_tab == 'Compare':
        st.title('🚧 CanAiCode Compare')

        filenames = list(data.keys())
        left_file = st.selectbox('Select the left result', filenames)
        right_file = st.selectbox('Select the right result', filenames)
        left_data = data[left_file]['results']
        right_data = data[right_file]['results']
        for left, right in zip(left_data, right_data):
            expander = st.expander(f'{left["name"]} - {left["language"]}')
            expander.write('Left: ', left)
            expander.write('Right: ', right)

    elif selected_tab == 'Explore':
        st.title('🚧 CanAiCode Explore')

        filenames = list(data.keys())
        filename = st.selectbox('Select the result', filenames)
        data = data[filename]
        results = data['results']
        st.dataframe(results, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

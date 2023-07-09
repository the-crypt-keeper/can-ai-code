import streamlit as st
import pandas as pd
import glob
import json
import os
import sys
import yaml

from prepare import load_questions

def read_ndjson(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data

def load_data():
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = ['results/eval*.ndjson',
                 'results/orca-mini-v2/eval*.ndjson',
                 'results/salesforce/eval*.ndjson',
                 'results/falcon/eval*.ndjson']
    files = []
    for path in paths:
        files += glob.glob(path)

    print('Loading', len(files), 'files')
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
                'results': list(filter(lambda x: x.get('language') == lang, results)),
                'runtime': results[0].get('runtime')
            }

    return data

def load_models():
    with open('models/models.yaml') as f:
        model_list = yaml.safe_load(f)
    model_df = pd.DataFrame(model_list)
    return model_df

def calculate_summary(data):
    summary = []
    for file, info in data.items():
        res = info['results']
        passed = sum(x['passed'] for x in res)
        total = sum(x['total'] for x in res)
        summary.append(info['tags'] + [passed, total] + [info['runtime']])
    sumdf = pd.DataFrame(summary, columns=['Eval', 'Interview', 'Languages', 'Template', 'TemplateOut', 'Params', 'Model', 'Timestamp', 'Passed', 'Total', 'Runtime'])
    sumdf = sumdf[['Languages','Model','Params','Template','Runtime','Passed','Total']]
    sumdf['Score'] = sumdf['Passed'] / sumdf['Total']
    sumdf.drop('Total', axis=1, inplace=True)

    merged_df = pd.merge(sumdf, load_models(), left_on='Model', right_on='id', how='left')
    merged_df['name'].fillna(merged_df['Model'], inplace=True)
    merged_df['size'].fillna('', inplace=True)
    merged_df.drop('Model', axis=1, inplace=True)

    return merged_df

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
                .row-widget {
                    padding-top: 0rem;
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
        
        settings_col, mode_col = st.columns((1,5))
        with mode_col:
            mode = st.radio(label='View', options=['Side by Side','Python','JavaScript'], horizontal=True, label_visibility='collapsed')
        with settings_col:
            best_of = st.checkbox(label='Show best result from each Model', value=True)
            
        tag_list = sorted(summary['tags'].explode().dropna().unique())
        if len(tag_list) > 0:
            tag_cols = st.columns(len(tag_list))
            tag_checks = []
            for i, tag in enumerate(tag_list):
                with tag_cols[i]:
                    tag_checks.append(st.checkbox(tag))

            tags_selected = [tag_list[i] for i, tag in enumerate(tag_list) if tag_checks[i]]
            if len(tags_selected) == 0:
                tags_selected = tag_list
            filtered = summary[summary['tags'].apply(lambda x: x != x or any(elem in x for elem in tags_selected))]
        else:
            filtered = summary

        if best_of:
            idx = filtered.groupby(['name','size','Languages'])['Score'].idxmax()
            filtered = filtered.loc[idx]

        filtered = filtered.sort_values(by='Passed', ascending=False)
            
        column_config={
            "Score": st.column_config.ProgressColumn(
                label="Score",
                help="Can it code?",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            "url": st.column_config.LinkColumn(
                label="URL",
                width=50
            ),
            "quant": st.column_config.TextColumn(
                label="Quant",
                width=30
            ),
            "size": st.column_config.TextColumn(
                label="Size",
                width=30
            )  
        }
        column_order=("name", "size", "url", "Params", "Template", "Score")
        column_order_detail=("name", "size", "quant", "url", "Params", "Template", "Runtime", "Passed", "Score")
        
        if mode == 'Side by Side':
            pyct, jsct = st.columns(2)
        else:
            pyct = st.container() if mode == 'Python' else None
            jsct = st.container() if mode == 'JavaScript' else None
            column_config['url']['width'] = 300

        if pyct is not None:
            with pyct:
                st.subheader('Python')
                st.dataframe(filtered[filtered['Languages'] == 'python'], use_container_width=True, column_config=column_config, column_order=column_order if mode == 'Side by Side' else column_order_detail, hide_index=True, height=700)

        if jsct is not None:
            with jsct:
                st.subheader('JavaScript')
                st.dataframe(filtered[filtered['Languages'] == 'javascript'], use_container_width=True, column_config=column_config, column_order=column_order if mode == 'Side by Side' else column_order_detail, hide_index=True, height=700)

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

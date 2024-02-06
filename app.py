import streamlit as st
import pandas as pd
import glob
import json
import os
import sys
import yaml
import requests
from copy import copy

def read_ndjson(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data

@st.cache_data
def load_data(paths):
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
                
        data[file] = {
            'tags': tags,
            'results': results,
            'runtime': results[0].get('runtime')
        }

    return data

def load_models():
    with open('models/models.yaml') as f:
        models = yaml.safe_load(f)

    model_list = []
    for id, model in models.items():
        if not 'size'in model:
            model['size'] = None
        elif model['size'][-1] == 'M':
            model['size'] = str(int(model['size'][:-1])/1000)
        elif model['size'][-1] == 'B':
            model['size'] = model['size'][:-1]
        else:
            raise Exception('bad model size '+model['size'])
        
        if not 'url' in model:
            model['url'] = 'https://huggingface.co/' + id.replace('-','/',1).replace('-fp16','').replace('-main','')
            
        if not 'tags' in model:
            model['tags'] = []

        model['id'] = id
        if 'alias' in model:
            alias_list = [model['alias']] if not isinstance(model['alias'], list) else model['alias']
            base_copy = copy(model)
            del base_copy['alias']
            model_list.append(base_copy)

            for id in alias_list:
                new_copy = copy(base_copy)
                new_copy['id'] = id
                model_list.append(new_copy)
        else:
            model_list.append(model)

    model_df = pd.DataFrame(model_list)
    return model_df

def verify_urls():
    models = load_models()
    for _, model in models.iterrows():
        try:
            response = requests.head(model['url'], allow_redirects=True)
            if response.status_code != 200:
                print(f"Broken URL for model {model['id']}: {model['url']} (Status code: {response.status_code})")
        except requests.RequestException as e:
            print(f"Request failed for model {model['id']}: {model['url']} (Error: {e})")

def calculate_summary(data, language = None):
    summary = []
    for file, info in data.items():
        res = info['results']
        passed = sum(x['passed'] for x in res if language is None or x['language'] == language)
        total = sum(x['total'] for x in res if language is None or x['language'] == language)
        if total == 0: continue
        summary.append(info['tags'] + [passed, total] + [info['runtime']])
    sumdf = pd.DataFrame(summary, columns=['Eval', 'Interview', 'Languages', 'Template', 'TemplateOut', 'Params', 'Model', 'Timestamp', 'Passed', 'Total', 'Runtime'])
    sumdf = sumdf[['Interview','Model','Params','Template','Runtime','Passed','Total']]
    sumdf['Score'] = sumdf['Passed'] / sumdf['Total']
    sumdf.drop('Total', axis=1, inplace=True)

    merged_df = pd.merge(sumdf, load_models(), left_on='Model', right_on='id', how='left')
    merged_df['name'].fillna(merged_df['Model'], inplace=True)
    merged_df['size'].fillna('', inplace=True)
    merged_df.drop('Model', axis=1, inplace=True)

    return merged_df

def load_and_prepare_data():
    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = ['results/**/eval*.ndjson']
    data = load_data(paths)    
    return data

def dump_csv():
    paths = ['results/**/eval*.ndjson']
    data = load_data(paths)
    summary = calculate_summary(data)
    summary.to_csv('can-ai-code-results.csv')
    print(f'Wrote {len(summary)} rows to can-ai-code-results.csv')
        
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
    
    data = load_and_prepare_data()    

    lb_tab, faq_tab, explore_tab = st.tabs(['Leaderboard 🏆', 'FAQ ❓', 'Explore 🔍'])

    with faq_tab:
        st.markdown(
        """
        ## What is this?

        This application explores the results of [CanAiCode](https://github.com/the-crypt-keeper/can-ai-code), a test suite specifically designed for testing small text-to-code LLMs.
        
        ## Why not HumanEval?

        These are complex interviews with hundreds of questions and the evaluation harness is python-specific.  See [llm-humaneval-benchmarks](https://github.com/my-other-github-account/llm-humaneval-benchmarks) and [code-eval](https://github.com/abacaj/code-eval) for projects large lists of Humaneval LLM benchmark results.

        ## What is the difference between `junior-v2` and `junior-dev` interviews?

        The v2 interview fixes a number of bugs in the prompt, self-checking, and evaluation harness.  It also focuses on code-generation models and avoids quantization where possible.

        ## Who are you?

        This leaderboard is maintained by [the-crypt-keeper](https://github.com/the-crypt-keeper) aka [kryptkpr](https://www.reddit.com/user/kryptkpr)

        ## How can I add a model?

        Open an issue tagged model request, or submit a PR!
        """)

    with explore_tab:
        st.markdown('Under construction.')
    
    with lb_tab:
        st.markdown('## CanAiCode Leaderboard 🏆 <sub>A visual tool to explore the results of [CanAiCode](https://github.com/the-crypt-keeper/can-ai-code)</sub>', unsafe_allow_html=True)

        view_col, interview_col, model_col, size_col = st.columns(4)

        with view_col:            
            mode_col, note_col = st.columns(2)
            with mode_col:
                mode = st.radio(label='View', options=['Both','Python','JavaScript'], label_visibility='collapsed')
            with note_col:
                best_of = st.checkbox(label='Best Result Only', value=True)
                show_quants = st.checkbox(label='Show All Quants', value=True, disabled=not best_of)
        
        mode_to_language={ 'Both': None, 'Python': 'python', 'JavaScript': 'javascript' }
        summary = calculate_summary(data, mode_to_language[mode])

        with interview_col:
            interview_list = sorted(summary['Interview'].unique())
            default_interview = interview_list.index('junior-v2') if 'junior-v2' in interview_list else 0
            selected_interview = st.selectbox('Interview', interview_list, index=default_interview)
            filtered = summary[ summary['Interview'] == selected_interview ]

        with model_col:
            tag_list = ["all"] + sorted(filtered['tags'].explode().dropna().unique())
            selected_tag = st.selectbox('Model Group',tag_list,index=0)
            
            if selected_tag != 'all':
                filtered = filtered[filtered['tags'].apply(lambda x: x != x or selected_tag in x)]

        with size_col:
            size_list = list(filtered['size'].dropna().unique())
            if '' in size_list: size_list.remove('')
            size_list.sort(key=lambda x: float(x) if x else 0)
            size_list = ['all'] + size_list
            selected_size = st.selectbox('Size', size_list, format_func=lambda x: 'all' if x == 'all' else '%dM'%(float(x)*1000) if float(x)<1 else x+'B')
            if selected_size != 'all':
                filtered = filtered[filtered['size'] == selected_size]

        if best_of:
            if not show_quants:
                idx = filtered.groupby(['name','size'], dropna=False)['Score'].idxmax()
            else:
                idx = filtered.groupby(['name','size','quant'], dropna=False)['Score'].idxmax()
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
                width=200
            ),
            "quant": st.column_config.TextColumn(
                label="Quant",
                width=30
            ),
            "params": st.column_config.TextColumn(
                label="Params",
                width=30
            ),            
            "size": st.column_config.NumberColumn(
                label="Size",
                format="%fB",
                width=30
            )  
        }
        column_order_detail=("name", "size", "quant", "url", "Params", "Template", "Runtime", "Passed", "Score")
        
        st.subheader("Python + JavaScript" if mode == "Both" else mode)
        st.dataframe(filtered, use_container_width=True, column_config=column_config, column_order=column_order_detail, hide_index=True, height=700)


    #elif selected_tab == 'Compare':
    #    st.title('🚧 CanAiCode Compare')
    #
    #    filenames = list(data.keys())
    #    left_file = st.selectbox('Select the left result', filenames)
    #    right_file = st.selectbox('Select the right result', filenames)
    #    left_data = data[left_file]['results']
    #    right_data = data[right_file]['results']
    #    for left, right in zip(left_data, right_data):
    #        expander = st.expander(f'{left["name"]} - {left["language"]}')
    #        expander.write('Left: ', left)
    #        expander.write('Right: ', right)

    #elif selected_tab == 'Explore':
    #    st.title('🚧 CanAiCode Explore')

    #    filenames = list(data.keys())
    #    filename = st.selectbox('Select the result', filenames)
    #    data = data[filename]
    #    results = data['results']
    #    st.dataframe(results, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_urls()
    elif len(sys.argv) > 1 and sys.argv[1] == '--csv':
        dump_csv()
    else:
        main()

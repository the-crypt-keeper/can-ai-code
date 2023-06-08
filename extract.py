from bs4 import BeautifulSoup
import re

# Useful functions for extracting code from LLM responces
def extract_code_markdown(answer):
    # Fallback if the model forgot to use block quotes or used a single quote instead.
    simple_answer = answer.replace('`','').strip()
    if simple_answer[0:3] == 'def' or simple_answer[0:8] == 'function':
        return simple_answer

    # Look for start tokens   
    match = re.search(r'```(\w*)', answer)
    start_token = match.group(0) if match else None
    start_index = match.start() if match else -1

    # If we didn't find a start token, return None
    if start_index == -1:
        return None

    # Find the index of the end token, starting from the end of the start token.
    # if not found, assume we're taking the whole thing.
    end_token = "```"
    end_index = answer.find(end_token, start_index + len(start_token))
    if end_index == -1:
        end_index = len(answer)

    # Extract the text between the tokens
    code_text = answer[start_index + len(start_token):end_index].strip()

    return code_text

def remove_indentation(code_block):
    lines = code_block.split('\n')
    if not lines:
        return code_block
    
    first_line_indent = len(lines[0]) - len(lines[0].lstrip())
    modified_lines = [line[first_line_indent:] for line in lines]
    modified_code = '\n'.join(modified_lines)
    return modified_code

def extract_code_html(answer):
    soup = BeautifulSoup(answer, "html.parser")

    longest_code = None
    for item in soup.find_all('code'):
        if longest_code is None or len(item.get_text()) > len(longest_code):
            #print("Found candidate code: ", item)
            longest_code = remove_indentation(item.get_text())

    return longest_code

def extract_code(answer):
    if answer.find('<code>') != -1:
        return extract_code_html(answer)
    
    if answer.find('```') != -1:
        return extract_code_markdown(answer)
    
    return None
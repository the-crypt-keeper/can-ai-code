from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional, Tuple

class CodeBlock:
    def __init__(self, code: str, source_type: str, line_count: int, char_count: int, 
                 has_def: bool, has_function: bool, has_class: bool):
        self.code = code
        self.source_type = source_type  # 'markdown', 'html', 'codellama', 'fallback'
        self.line_count = line_count
        self.char_count = char_count
        self.has_def = has_def
        self.has_function = has_function
        self.has_class = has_class
    
    def score(self) -> Tuple[bool, int, int]:
        """Return a tuple for sorting priority: (has_definition, line_count, char_count)"""
        has_definition = self.has_def or self.has_function or self.has_class
        return (has_definition, self.line_count, self.char_count)

def remove_indentation(code_block: str) -> str:
    """Remove common indentation from a code block."""
    if not code_block:
        return code_block
        
    lines = code_block.split('\n')
    if not lines:
        return code_block
    
    # Find first non-empty line
    first_non_empty = None
    for i, line in enumerate(lines):
        if line.strip():
            first_non_empty = i
            break
    
    if first_non_empty is None:
        return code_block
    
    # Calculate indentation from first non-empty line
    first_line = lines[first_non_empty]
    indent_size = len(first_line) - len(first_line.lstrip())
    
    # Remove the indentation from all lines
    modified_lines = []
    for line in lines:
        if line.strip():  # Only process non-empty lines for indentation
            # Make sure we don't remove more characters than exist in the line
            remove_count = min(indent_size, len(line) - len(line.lstrip()))
            modified_lines.append(line[remove_count:])
        else:
            modified_lines.append(line)  # Keep empty lines as is
            
    return '\n'.join(modified_lines)

def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from text."""
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)

def find_markdown_code_blocks(text: str) -> List[str]:
    """Find all markdown code blocks in the text."""
    # Regex to match code blocks with triple backticks
    pattern = r'```(?:\w*\n)?(.*?)```'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    code_blocks = []
    for match in matches:
        code = match.group(1).strip()
        if code:
            code_blocks.append(code)
    
    return code_blocks

def find_html_code_blocks(text: str) -> List[str]:
    """Find all HTML code blocks in the text."""
    soup = BeautifulSoup(text, "html.parser")
    code_blocks = []
    
    for item in soup.find_all('code'):
        code = remove_indentation(item.get_text())
        if code.strip():
            code_blocks.append(code)
            
    return code_blocks

def find_codellama_blocks(text: str) -> List[str]:
    """Find all CodeLlama style [PYTHON]...[/PYTHON] blocks."""
    start_token = '[PYTHON]'
    end_token = '[/PYTHON]'
    
    code_blocks = []
    start_pos = 0
    
    while True:
        start_index = text.find(start_token, start_pos)
        if start_index == -1:
            break
            
        end_index = text.find(end_token, start_index + len(start_token))
        if end_index == -1:
            break
            
        code = text[start_index + len(start_token):end_index].strip()
        if code:
            code_blocks.append(code)
            
        start_pos = end_index + len(end_token)
        
    return code_blocks

def analyze_code_block(code: str, source_type: str) -> CodeBlock:
    """Analyze a code block for various properties."""
    line_count = len(code.split('\n'))
    char_count = len(code)
    
    has_def = bool(re.search(r'\bdef\b', code))
    has_function = bool(re.search(r'\bfunction\b', code))
    has_class = bool(re.search(r'\bclass\b', code))
    
    return CodeBlock(
        code=code,
        source_type=source_type,
        line_count=line_count,
        char_count=char_count,
        has_def=has_def,
        has_function=has_function,
        has_class=has_class
    )

def extract_code(answer: str, stop_at_prefix: List[str] = []) -> Optional[str]:
    """
    Main function to extract code from LLM answers.
    Uses a two-pass approach:
    1. Identify all potential code blocks
    2. Select the best block based on heuristics
    """
    # Remove thinking tags
    answer = remove_think_tags(answer)
    
    # Pass 1: Find all code blocks
    all_blocks = []
    
    # Look for CodeLlama style blocks
    for code in find_codellama_blocks(answer):
        all_blocks.append(analyze_code_block(code, 'codellama'))
    
    # Look for HTML code blocks
    for code in find_html_code_blocks(answer):
        all_blocks.append(analyze_code_block(code, 'html'))
    
    # Look for Markdown code blocks
    for code in find_markdown_code_blocks(answer):
        all_blocks.append(analyze_code_block(code, 'markdown'))
    
    # Pass 2: Select the best block
    if all_blocks:
        # Sort blocks by: (has_definition, line_count, char_count)
        all_blocks.sort(key=lambda block: block.score(), reverse=True)
        selected_code = all_blocks[0].code
    else:
        # Fallback: treat entire answer as code
        selected_code = answer.strip()
    
    # Handle stop_at_prefix if provided
    if selected_code and stop_at_prefix:
        lines = selected_code.split('\n')
        for i, line in enumerate(lines):
            for prefix in stop_at_prefix:
                if line.startswith(prefix):
                    selected_code = '\n'.join(lines[:i])
                    break
            if len(selected_code.split('\n')) < i:
                break
    
    return selected_code
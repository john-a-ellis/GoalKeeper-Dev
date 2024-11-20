from langchain_core.messages import SystemMessage
# from pydantic import BaseModel
from typing import List
import os

def read_prompt(prompt_name: str) -> List[SystemMessage]:
    """
    Reads a prompt file and returns it as a SystemMessage.
    
    Args:
        prompt_name (str): Name of the prompt file to read (without .txt extension)
        
    Returns:
        List[SystemMessage]: A list containing a single SystemMessage
    """
    try:
        with open(f'assets/prompts/{prompt_name}.txt', 'r') as file:
            prompt_text = file.read()
            # Remove "system," prefix if it exists in the text file
            if prompt_text.startswith("system,"):
                prompt_text = prompt_text[7:]
            return [SystemMessage(content=prompt_text)]
    except FileNotFoundError:
        print(f'The {prompt_name} does not exist')
        # Return empty system message or handle error as needed
        return [SystemMessage(content="")]
    except IOError:
        print(f'An error occurred trying to read the {prompt_name}')
        # Return empty system message or handle error as needed
        return [SystemMessage(content="")]
    
def get_user_id(auth_data):
    if os.getenv('DEPLOYED', 'False').lower() == 'true':
        user_id = auth_data.get('user_info', {}).get('email', 'User')
        pass
    else: 
        user_id='default'
        pass
    return user_id
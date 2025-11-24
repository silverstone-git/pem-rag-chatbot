import re
import random

def make_it_an_id(file_name):
    """
    Input: A file name mixed with spaces, periods, etc.
    Output: '_' separated smallcase alphabetic id, with random filler if less than 5 chars
    """
    # 1. Convert to lowercase
    file_name = file_name.lower()

    # 2. Replace non-alphanumeric characters (except periods and spaces) with underscores
    # Keep periods for now to split by later, and spaces for initial conversion
    cleaned_name = re.sub(r'[^a-z0-9\s\.]', '_', file_name)
    
    # 3. Replace spaces and periods with underscores
    cleaned_name = re.sub(r'[\s\.]+', '_', cleaned_name)

    # 4. Remove leading/trailing underscores and multiple consecutive underscores
    cleaned_name = re.sub(r'_{2,}', '_', cleaned_name).strip('_')

    # Ensure it only contains alphabetic characters (after previous cleaning)
    # If the file_name was something like "123.pdf", this step ensures we only keep alphabetic parts.
    # We will filter out non-alphabetic parts after initial cleaning to retain some structure.
    # Let's refine this to ensure we only keep alphabetic parts before padding.
    alphabetic_parts = re.findall(r'[a-z]+', cleaned_name)
    
    # Join alphabetic parts with underscores
    result_id = '_'.join(alphabetic_parts)

    # 5. Add random filler if less than 5 chars
    if len(result_id) < 5:
        # Generate random lowercase alphabetic characters
        filler_length = 5 - len(result_id)
        random_filler = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(filler_length))
        
        # Append filler with an underscore if result_id is not empty
        if result_id:
            result_id += '_' + random_filler
        else: # If the result_id is empty (e.g., from "123.txt" or "$.*"), just use the filler
            result_id = random_filler
            
    return result_id



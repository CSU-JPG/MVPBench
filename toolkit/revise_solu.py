import os
import json
import glob
import re

def clean_json_files(directory_path):
    # Obtain all the json files in the directory
    json_files = glob.glob(os.path.join(directory_path, "**/*.json"), recursive=True)
    
    for file_path in json_files:
        try:
            # Read JSON file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # First check if the file is already in clean JSON format
            try:
                data = json.loads(content)
                if isinstance(data, list) and all(isinstance(item, list) and len(item) > 0 and isinstance(item[0], str) for item in data):
                    print(f"File is already in correct format: {file_path}")
                    continue
            except json.JSONDecodeError:
                pass
            
            # Find the actual JSON array part
            # Ignore any potential [JSON array] markers
            json_pattern = r'(\[\s*\[\s*"key_step_.*?\]\s*\])'
            matches = re.search(json_pattern, content, re.DOTALL)
            
            if matches:
                json_array_str = matches.group(1)
                try:
                    # Validate that the extracted part is valid JSON
                    cleaned_data = json.loads(json_array_str)
                    
                    # Write back to file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(cleaned_data, file, indent=2)
                    print(f"Cleaned file: {file_path}")
                except json.JSONDecodeError as e:
                    print(f"Unable to parse extracted JSON array {file_path}: {str(e)}")
                    
                    # Try to manually fix common issues
                    try:
                        # Remove potential comments or non-JSON text
                        cleaned_str = re.sub(r'//.*?[\r\n]|/\*.*?\*/', '', json_array_str, flags=re.DOTALL)
                        # Ensure quotes are correct
                        cleaned_str = cleaned_str.replace("'", '"')
                        cleaned_data = json.loads(cleaned_str)
                        
                        # Write back to file
                        with open(file_path, 'w', encoding='utf-8') as file:
                            json.dump(cleaned_data, file, indent=2)
                        print(f"Fixed and cleaned file: {file_path}")
                    except Exception:
                        print(f"Fix failed: {file_path}")
            else:
                # If regex didn't find anything, try a more traditional approach
                try:
                    # Find the start and end of the first JSON array
                    start_bracket = content.find('[')
                    if start_bracket != -1:
                        # Skip potential [JSON array] markers
                        if content[start_bracket:].strip().startswith('[JSON'):
                            start_bracket = content.find('[', start_bracket + 1)
                        
                        # Find the matching closing bracket
                        bracket_count = 0
                        end_bracket = -1
                        
                        for i in range(start_bracket, len(content)):
                            if content[i] == '[':
                                bracket_count += 1
                            elif content[i] == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_bracket = i
                                    break
                        
                        if end_bracket != -1:
                            # Extract the JSON array part
                            json_array = content[start_bracket:end_bracket+1]
                            try:
                                # Validate that the extracted part is valid JSON
                                cleaned_data = json.loads(json_array)
                                
                                # Write back to file
                                with open(file_path, 'w', encoding='utf-8') as file:
                                    json.dump(cleaned_data, file, indent=2)
                                print(f"Cleaned file (traditional method): {file_path}")
                            except json.JSONDecodeError:
                                print(f"Unable to parse extracted JSON array (traditional method): {file_path}")
                    else:
                        print(f"Could not find JSON array in file: {file_path}")
                except Exception as e:
                    print(f"Traditional method processing failed {file_path}: {str(e)}")
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

# Specify the directory paths to process
directory_paths = [
    "Result/output_single_img/solu_p-t/PhyTest"
]

for directory_path in directory_paths:
    print(f"Processing directory: {directory_path}")
    clean_json_files(directory_path)
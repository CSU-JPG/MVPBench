import os
import yaml
import json
import glob
from datasets import Dataset

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return None


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True

def load_local_dataset(data_dir, subject=None):
    # Ensure data_dir is an absolute path
    data_dir = os.path.abspath(data_dir)
    
    # Get all JSON files in the specified directory
    json_files = glob.glob(f"{data_dir}/**/*.json", recursive=True)
    
    # Read all JSON files
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Process JSON in list format
                if isinstance(data, list):
                    for item in data:
                        # Process single item
                        processed_item = process_item(item, data_dir, subject, json_file)
                        if processed_item:
                            all_data.append(processed_item)
                
                # Process JSON in dictionary format (single question)
                elif isinstance(data, dict):
                    processed_item = process_item(data, data_dir, subject, json_file)
                    if processed_item:
                        all_data.append(processed_item)
        except Exception as e:
            print(f"reading file {json_file} occurred error: {e}")
            continue
    
    print(f"successfully loaded {len(all_data)} data")
    # Convert to Dataset object
    return Dataset.from_list(all_data)

def process_item(item, data_dir, subject=None, json_file=None):
    """process single data item"""
    if not isinstance(item, dict):
        return None
        
    # Check subject filter condition
    if subject is not None and (not item.get('subject') or item.get('subject') != subject):
        return None
    
    # Add json_path field (if needed)
    if 'json_path' not in item and json_file:
        item['json_path'] = json_file
    
    # Process image path
    process_image_path(item, 'original_scene', data_dir)
    
    # Process final_scene field
    # process_image_path(item, 'final_scene', data_dir)
    
    # Process key_step fields
    for i in range(1, 10):  # Assume maximum of 9 key_steps
        step_key = f'key_step_{i}'
        if step_key in item:
            process_image_path(item, step_key, data_dir)
        else: break
    return item

def process_image_path(item, field, data_dir):
    """process image path in data item"""
    if field in item and 'path' in item[field]:
        path = item[field]['path']
        
        # If path is empty, don't process
        if not path or path.strip() == "":
            return
        
        # Standardize path format
        data_dir = os.path.normpath(data_dir)
        path = os.path.normpath(path)
        
        # Try various possible path combinations
        possible_paths = [
            # Use provided path directly
            os.path.join(data_dir, path),
            
            # If path contains PhyTest and data_dir contains Data/PhyTest, remove PhyTest prefix
            os.path.join(data_dir, path.replace('PhyTest/', '', 1)) if path.startswith('PhyTest/') else None,
            
            # Try to find in Data directory
            os.path.join(os.path.dirname(data_dir), path) if 'Data' in data_dir else None,
            
            # Absolute path
            path if os.path.isabs(path) else None
        ]
        
        # Filter out None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        # Check each possible path
        found = False
        for full_path in possible_paths:
            if os.path.isfile(full_path):
                item[field]['full_path'] = full_path
                item[field]['file_exists'] = True
                #print(f"Found image file: {full_path}")
                found = True
                break
        
        if not found:
            # If all attempts fail, use the first path as default
            item[field]['full_path'] = possible_paths[0] if possible_paths else os.path.join(data_dir, path)
            item[field]['file_exists'] = False
            print(f"Image file not found: {item[field]['full_path']}")
            # Output all attempted paths for debugging
            # print(f"Attempted paths: {possible_paths}")

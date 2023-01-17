'''
Introduction:
This script reads a ipynb file and extracts all cells marked with ###EXTRACT###
to a py-file. It is ment to be used for converting a ipynb-file into a classic
python module.

Use:
python3 ipynb_to_py.py /path/to/input/my.ipynb /path/to/output/my.py
'''

import sys
import json

def read_json(path: str) -> dict:
    """
    Description:
        Reads a json-file and returns a dictionary
    Inputs:
        path: Path to jupyter notebook (.ipynb)
    Outputs:
        dictionary representation of notebook
    """
    file = open(path, mode= "r", encoding= "utf-8")
    myfile = file.read()
    myjson = json.loads(myfile)
    file.close()
    return myjson

def get_code_cells(dictionary: dict) -> list:
    """
    Description
        Finds cells of ipynb with code
    Inputs:
        dictionary: Dictionary from importing a ipynb notebook
    Output:
        List of code cells
    """
    code_cells = [cell for cell in dictionary['cells'] if cell['cell_type']=='code']
    return code_cells

def get_labeled_cells(code_cells: dict, label="###EXPORT###") -> dict:
    """
    Description
        Gets cells with the specified label
    Inputs:
        code_cells: Dictionary with code cells
    Output:
        Dictionary with labeled cells
    """
    label = label + "\n"
    labeled_cells = [cell['source'] for cell in code_cells if cell['source'][0]==label]
    return labeled_cells

def write_to_file(labeled_cells: dict, output_file: str) -> None:
    """
    Description:
        Writes the labeled cells to a file
    Inputs:
        labeled_cells: Dictionary with cells that should be written to a file
    """
    flattened_lists = '\n\n'.join([''.join(labeled_cell[1:]) for labeled_cell in labeled_cells])
    file = open(output_file, 'w')
    file.write(flattened_lists)
    file.close()

if __name__ == "__main__":
    print(f'INFO: Cells with label {sys.argv[2]} extracted from {sys.argv[1]}')
    json_file = read_json(sys.argv[1])
    code_cells = get_code_cells(json_file)
    labeled_cells = get_labeled_cells(code_cells,sys.argv[2])
    write_to_file(labeled_cells, sys.argv[3])

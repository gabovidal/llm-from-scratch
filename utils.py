import os
# TODO: functions to pre-process Tolkien's texts


def get_data(url, save_file):
    """snippet function to get the txt files from public github repositories **if** it hasn't yet done it previously"""
    try:
        with open(save_file, 'r', encoding='latin-1') as f:
            return f.read()
    except:
        os.system(f"curl {url} -o {save_file}")
        with open(save_file, 'r', encoding='latin-1') as f:
            return f.read()

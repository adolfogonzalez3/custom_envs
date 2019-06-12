'''Utilities for the manipulation of files.'''
import json
import subprocess
from pathlib import Path


def load_json(path):
    '''
    Load a json file from the given path.

    :param path: (str or Path) The path to the json file.
    :return: (dict) The dictionary from the json file.
    '''
    with Path(path).open('rt') as json_file:
        return json.load(json_file)


def save_json(dictionary, path):
    '''
    Save a dictionary to a json file.

    :param path: (str or Path) The path to save the json file.
    '''
    with Path(path).with_suffix('.json').open('wt') as json_file:
        return json.dump(dictionary, json_file)


def get_commit_hash(path=None):
    '''
    Get the current commit hash.

    :param path: (str or None) If str then `path` is the path to the git
                               repository else if None then is current working
                               directory.
    :return: (str) The git commit hash.
    '''
    path = Path() if path is None else Path(path)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            cwd=path).stdout
    if len(commit) == 0:
        raise RuntimeError("Path doesn't point to a repository.")
    return commit.rstrip().decode()

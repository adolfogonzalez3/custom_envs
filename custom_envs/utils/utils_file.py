'''Utilities for the manipulation of files.'''
import json
import shutil
import subprocess
from pathlib import Path


def load_json(path, line_no=None):
    '''
    Load a json file from the given path.

    :param path: (str or Path) The path to the json file.
    :param line_no: (int or None) If int then deserialize the `line_no` line
                                  otherwise if None then deserialize
                                  everything.
    :return: (dict) The dictionary from the json file.
    '''
    with Path(path).open('rt') as json_file:
        if line_no is None:
            dictionary = json.load(json_file)
        else:
            for _ in range(line_no - 1):
                json_file.readline()
            dictionary = json.loads(json_file.readline())
    return dictionary


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
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                            stdout=subprocess.PIPE).stdout
    if not commit:
        raise RuntimeError("Path doesn't point to a repository.")
    return commit.rstrip().decode()

def remove_suffixes(path):
    '''
    Retrieve path without any suffixes.

    :param path: (pathlike) A pathlike object that may have suffixes.
    :return: (Path) A Path object without suffixes.
    '''
    path = Path(path)
    for _ in range(len(path.suffixes)):
        path = path.with_suffix('')
    return path

def decompress(source, destination=None, remove=False):
    '''
    Decompress a file pointed to by the file_path.

    :param source: (pathlike) The path to the compressed file.
    :param destination: (pathlike) The destination path.
    :param remove: (bool) If True then remove compressed file.
    :return: (str) Path to decompressed directory.
    '''
    source = Path(source)
    source_name = remove_suffixes(source)
    assert source.is_file()
    destination = Path(destination) if destination else source_name
    if destination.is_dir():
        destination = destination / source_name.name
    shutil.unpack_archive(source, destination)
    if remove:
        source.unlink()
    return destination.with_name(source_name.name)


def compress(source, destination=None, remove=False):
    '''
    Compress a file pointed to by the file_path.

    :param source: (pathlike) The path to the directory.
    :param destination: (pathlike) The path to the file.
    :param remove: (bool) If True then remove source directory.
    :return: (str) Path to compressed file.
    '''
    source = Path(source)
    assert source.is_dir()
    if destination is None:
        destination = source.with_suffix('.zip')
    if destination.is_dir():
        destination = destination / remove_suffixes(source).name
        destination = destination.with_suffix('.zip')
    destination = Path(destination) 
    compress_fmt = ''.join(reversed(destination.suffixes)).replace('.', '')
    shutil.make_archive(remove_suffixes(destination), compress_fmt, source)
    if remove:
        shutil.rmtree(source)
    return destination

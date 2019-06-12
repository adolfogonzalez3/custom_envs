'''Tests utils_file module.'''

import json
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

import custom_envs.utils.utils_file as utils_file


def create_test_file(path, version=0):
    '''Create a json file for testing.'''
    if version == 0:
        path = Path(path, 'test.json')
        data = {chr(97 + i): i for i in range(26)}
        with path.open('wt') as test:
            json.dump(data, test)
    else:
        path = Path(path, 'test_line.json')
        with path.open('wt') as test:
            data = [{chr(97 + i): i for i in range(26-j)}
                    for j in range(1, 11)]
            for i in range(10):
                test.write('\n'.join(json.dumps(d) for d in data))
    return path


@pytest.fixture(scope="module")
def save_path():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_load_json(save_path):
    '''Test load_json function.'''
    path = create_test_file(save_path)
    dictionary = utils_file.load_json(path)
    for i, (name, value) in enumerate(dictionary.items()):
        assert i == value
        assert name == chr(97 + i)
    path = create_test_file(save_path, version=1)
    dictionary = utils_file.load_json(path, line_no=5)
    assert len(dictionary) == 26 - 5
    for i, (name, value) in enumerate(dictionary.items()):
        assert i == value
        assert name == chr(97 + i)


def test_save_json(save_path):
    '''Test save_json function.'''
    data = {chr(97 + i): i for i in range(26)}
    path = save_path / 'test_save.json'
    utils_file.save_json(data, path)
    dictionary = utils_file.load_json(path)
    for i, (name, value) in enumerate(dictionary.items()):
        assert i == value
        assert name == chr(97 + i)


def test_get_commit_hash(save_path):
    '''Test get_commit_hash function.'''
    with pytest.raises(RuntimeError):
        utils_file.get_commit_hash(save_path)
    subprocess.run(["git", "init"], cwd=save_path)
    commit_no = utils_file.get_commit_hash(save_path)
    assert len(commit_no) > 0

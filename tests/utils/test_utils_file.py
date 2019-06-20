'''Tests utils_file module.'''

import json
import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

import custom_envs.utils.utils_file as utils_file


def create_test_file(path, name=None, version=0):
    '''Create a json file for testing.'''
    name = name if name else 'test.json'
    path = Path(path, name).with_suffix('.json')
    if version == 0:
        data = json.dumps({chr(97 + i): i for i in range(26)})
    else:
        data = [{chr(97 + i): i for i in range(26-j)}
                for j in range(1, 11)]
        data = '\n'.join([json.dumps(d) for d in data]*10)
    with path.open('wt') as test:
        test.write(data)
    return path


@pytest.fixture(scope="module")
def save_path():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_load_json(save_path):
    '''Test load_json function.'''
    path = create_test_file(save_path, name='test_load_json0')
    dictionary = utils_file.load_json(path)
    for i, (name, value) in enumerate(dictionary.items()):
        assert i == value
        assert name == chr(97 + i)
    path = create_test_file(save_path, name='test_load_json1', version=1)
    dictionary = utils_file.load_json(path, line_no=5)
    assert len(dictionary) == 26 - 5
    for i, (name, value) in enumerate(dictionary.items()):
        assert i == value
        assert name == chr(97 + i)


def test_save_json(save_path):
    '''Test save_json function.'''
    data = {chr(97 + i): i for i in range(26)}
    path = save_path / 'test_save_json.json'
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
    assert commit_no


def test_remove_suffixes(save_path):
    '''Test remove_suffixes function.'''
    assert utils_file.remove_suffixes('test') == Path('test')
    assert utils_file.remove_suffixes('test.tar') == Path('test')
    assert utils_file.remove_suffixes('test.tar.tar.tar') == Path('test')


def test_decompress(save_path):
    '''Test decompress function.'''
    target_dir = save_path / 'test_decompress'
    target_dir.mkdir()
    for i in range(10):
        create_test_file(target_dir, name=str(i))
    shutil.make_archive(target_dir, 'xztar', target_dir)
    shutil.rmtree(target_dir)
    decompress_path = utils_file.decompress(target_dir.with_suffix('.tar.xz'),
                                            save_path)
    assert decompress_path.is_dir()
    shutil.rmtree(decompress_path)
    decompress_path = utils_file.decompress(target_dir.with_suffix('.tar.xz'),
                                            save_path, remove=True)
    assert decompress_path.is_dir()
    assert not target_dir.with_suffix('.tar.xz').exists()


def test_compress(save_path):
    '''Test decompress function.'''
    target_dir = save_path / 'test_compress'
    target_dir.mkdir()
    for i in range(10):
        create_test_file(target_dir, name=str(i))
    compressed_file = utils_file.compress(target_dir, save_path)
    assert compressed_file.is_file()
    compressed_file.unlink()
    compressed_file = utils_file.compress(target_dir, save_path, remove=True)
    assert compressed_file.is_file()
    assert not target_dir.exists()

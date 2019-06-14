'''A script for decompressing files.'''
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def decompress(file_path):
    '''
    Decompress a file pointed to by the file_path.

    :param file_path: (pathlike) The path to the file.
    '''
    file_path = Path(file_path)
    assert file_path.is_file()
    shutil.unpack_archive(file_path, file_path.with_suffix(''))


def main():
    '''Decompress files given a path.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to the files.")
    args = parser.parse_args()
    all_compressed_files = list(Path().glob(args.path))

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(decompress, all_compressed_files),
                  total=len(all_compressed_files)))


if __name__ == '__main__':
    main()

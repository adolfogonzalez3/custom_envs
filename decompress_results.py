
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

def decompress(path):
    shutil.unpack_archive(str(path), str(path).rstrip('.zip'))

def main():
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("path", help="The path to the files.")
    ARGS = PARSER.parse_args()
    PATH = Path(ARGS.path)

    with ThreadPoolExecutor() as executor:
        all_zip_folders = list(PATH.glob('*.zip'))
        list(tqdm(executor.map(decompress, all_zip_folders),
                  total=len(all_zip_folders)))

if __name__ == '__main__':
    main()

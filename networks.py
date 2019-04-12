
from pathlib import Path
from custom_envs import load_data

if __name__ == '__main__':
    print(__file__)
    path = Path(__file__)
    print(path.resolve())
    data = load_data()

import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def get_num_files(path) -> None:
    """
    Walks through a directory and prints its contents
    :param path: Path to directory.
    """

    for path, _, filenames in os.walk(path):
        print(f"There are {len(filenames)} images in '{path}'.")

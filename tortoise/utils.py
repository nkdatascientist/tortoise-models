
import requests, os, time
from loguru import logger
from tqdm import tqdm  # Import the tqdm library for progress bar

def download_file(url, filename, dest_folder):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))  # Get the total file size

    if response.status_code == 200:
        file_path = os.path.join(dest_folder, filename)
        with open(file_path, "wb") as file, tqdm(
            desc=filename,  # Use the filename as the description for the progress bar
            total=total_size,
            unit="B",  # Unit to display in the progress bar
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                bar.update(len(chunk))  # Update the progress bar with the chunk size

        print(f"Downloaded file {filename} to {dest_folder}")
    else:
        print(f"Failed to download file {url}"); exit()


# https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
from functools import wraps
def timetaken(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

# https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html
import hashlib
def get_sha256_value(filename):
    with open(filename, "rb") as f:
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# https://github.com/Lornatang/ResNet-PyTorch/blob/main/utils.py#L134
from enum import Enum
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

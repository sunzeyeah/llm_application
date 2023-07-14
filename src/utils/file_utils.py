
import shutil
import os
import random
import numpy as np
import torch

from src.utils.logger import logger


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def list_dir(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def rmdir(folder):
    ''' remove all the sub-directory and files within a folder, but does not remove the folder itself

    :param folder: directory path
    :return:
    '''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warn('Failed to delete %s. Reason: %s' % (file_path, e))
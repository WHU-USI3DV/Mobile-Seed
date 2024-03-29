'''
This file is modified from:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/utils/__init__.py
'''
from .collect_env import collect_env
from .logger import get_root_logger
from .misc import find_latest_checkpoint
from .set_env import setup_multi_processes

__all__ = [
    'get_root_logger', 'collect_env', 'find_latest_checkpoint',
    'setup_multi_processes'
]

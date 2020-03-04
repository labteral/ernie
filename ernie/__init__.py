#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .ernie import *
import logging

__version__ = '0.0.21b0'

logging.getLogger().setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)-15s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def _get_cpu_name():
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = f"{cpu_info['brand']}, {cpu_info['count']} vCores"
    return cpu_name


def _get_gpu_name():
    from tensorflow.python.client import device_lib
    gpu_name = device_lib.list_local_devices()[3].physical_device_desc.split(',')[1].split('name:')[1].strip()
    return gpu_name


device_name = _get_cpu_name()
device_type = 'CPU'

try:
    device_name = _get_gpu_name()
    device_type = 'GPU'
except IndexError:
    # Detect TPU
    pass

logging.info(f'ernie v{__version__}')
logging.info(f'target device: [{device_type}] {device_name}\n')

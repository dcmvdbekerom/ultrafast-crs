# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:34:50 2024

@author: bekeromdcmvd
"""

#%% funtion definitions and import
import sys
import requests
import os
import numpy as np

def download_file(file_name, url, fpath='./', barsize=35):
    
    link = url + file_name
        
    with open(fpath + file_name, "wb") as f:

        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')
    
        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(barsize * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (barsize-done)) )    
                sys.stdout.flush()
            print('')


#%% run

fnames = [
    'nu_data.npz',
    'sigma_gRmin_data.npz',
    'E0_data.npz',
    'J_clip_data.npz',
    'EvJ_data.npz',
    ]
url = 'https://dcmvdbekerom.github.io/ultrafast-crs-data/data/CH4_v2/'
fpath = './data/CH4_v2/'

arr_dict = {}

os.makedirs(fpath, exist_ok=True)

for i,fname in enumerate(fnames):
    
    print('({:d}/{:d}) Downloading {:s}:'.format(i+1, len(fnames), fname))
    download_file(fname, url, fpath)
    basename, ext = os.path.splitext(fname)
    
    print('Decompressing... ', end='')
    arr = np.load(fpath + fname)['arr_0']    
    arr_dict[basename] = arr
    os.remove(fpath + fname)
    print('Done!\n')

np.savez(fpath + 'database.npz', **arr_dict)
print('\nAll downloads finished!')

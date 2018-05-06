import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/kimby/eccodes/lib/python2.7/site-packages')
from eccodes import *

GRIB_FOLDER = '/media/kimby/Data/dev/deep-vortices/data/ecmwf_era_interim'
GRID_SIZE = 0.75

def load_data(file_name, year):
    """ 
        Loads data from the grib file 
        and returns it
    """
    f = open(os.path.join(GRIB_FOLDER, file_name+'.grib'))
    
    # while 1:
    #     gid = codes_grib_new_from_file(f)
    #     if gid is None:
    #         break
 
    #     iterid = codes_keys_iterator_new(gid, 'ls')
 
    #     # Different types of keys can be skipped
    #     # codes_skip_computed(iterid)
    #     # codes_skip_coded(iterid)
    #     # codes_skip_edition_specific(iterid)
    #     # codes_skip_duplicates(iterid)
    #     # codes_skip_read_only(iterid)
    #     # codes_skip_function(iterid)
 
    #     while codes_keys_iterator_next(iterid):
    #         keyname = codes_keys_iterator_get_name(iterid)
    #         keyval = codes_get_string(iterid, keyname)
    #         print("%s = %s" % (keyname, keyval))
 
    #     codes_keys_iterator_delete(iterid)
    #     codes_release(gid)
 
    # f.close()

    frame_it = 0
    images = []
    while True:
        frame_it = frame_it + 1
        gid = codes_grib_new_from_file(f)
        
        if gid is None:
            break
        
        date = codes_get(gid, 'dataDate')
        if str(date)[:4] != year:
            codes_release(gid)
            continue

        time = codes_get(gid, 'dataTime')
        stepRange = codes_get(gid, 'stepRange')
        print(date, time, stepRange, frame_it)
        
        nx = codes_get(gid, 'Ni')
        ny = codes_get(gid, 'Nj')

        missingVal = codes_get_double(gid, 'missingValue')

        img = np.empty([ny,nx], dtype=float)
        iterid = codes_grib_iterator_new(gid, 0)

        while 1:
            result = codes_grib_iterator_next(iterid)
            if not result:
                break
 
            [lat, lon, value] = result
            r, c = int((-lat + 90) / GRID_SIZE), int(lon / GRID_SIZE)
            img[r,c] = value
            # print r, c
         
            if value == missingVal:
                print("missing")

        images.append(img)
        if frame_it % 2 == 0:
            fields = np.asarray(images).transpose([1,2,0]).astype('float32')
            t = 0
            if time == 1200:
                t = 1
            field_path = os.path.join(GRIB_FOLDER, 'fc/{}_{}_{}.npy'.format(date, t, stepRange))
            np.save(field_path, fields)
            images = []
        
        codes_grib_iterator_delete(iterid)
        codes_release(gid)

        # if frame_it == 16: break

    f.close()
    return

    fields = []
    for i in range(len(images)/2):
        fields.append(np.stack((images[i*2],images[i*2+1])))
    fields = np.asarray(fields).transpose([0,2,3,1]).astype('float32')
    print(fields.shape)

    field_path = os.path.join(GRIB_FOLDER, file_name+'.npy')
    np.save(field_path, fields)

    plt.figure()
    # plt.subplot(311)
    # plt.imshow(fields[0,:,:,0])
    # plt.subplot(312)
    # plt.imshow(fields[0,:,:,1])
    # plt.subplot(313)
    # plt.imshow(fields[0,:,:,0]**2+fields[0,:,:,1]**2)

    plt.subplot(221)
    plt.imshow(fields[0,:,:,0])
    plt.subplot(222)
    plt.imshow(fields[0,:,:,1])
    plt.subplot(223)
    plt.imshow(fields[1,:,:,0])
    plt.subplot(224)
    plt.imshow(fields[1,:,:,1])
    plt.show()

    return

# load_data('uv-2013-2014-fc', '2013')
load_data('uv-2013-2014-fc', '2014')
# load_data('uv-2013-2017-fc', '2015')
# load_data('uv-2013-2017-fc', '2016')
# load_data('uv-2013-2017-fc', '2017')
# load_data('uv-2013-2017-an')
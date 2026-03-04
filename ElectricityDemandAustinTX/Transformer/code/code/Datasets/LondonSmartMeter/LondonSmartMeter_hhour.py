# -*- coding: utf-8 -*-
import datetime

import torch
import pickle
import zipfile
import pgzip
import os

import sys
from multiprocessing import Pool

def p_(path,zcsv_fnames):
    block_householdseries = []
    zf = zipfile.ZipFile(path)
    
    for fname in zcsv_fnames:
        #f = open(os.path.join(path,fname))
        f = zf.open(fname)
        lines = f.readlines()
        
        #Exclude header
        lines = lines[1:]
        
        #For some reason zipfile reads out bytes instead of strings
        for i in range(len(lines)):
            if type(lines[i]) != str:
                lines[i] = lines[i].decode()
    
        #Do 1 pass through the lines to bucket records by household LCLids
        households = {}
        for line in lines:
            splitline = line.split(',')
            
            LCLid = int(splitline[0].strip('MAC')) #LCLids are prefixed by 'MAC'
            
            tstp = datetime.datetime.fromisoformat(splitline[1][:-1])

            #Reject timestamps that are not aligned at 0min or 30min
            if not ((tstp.minute == 0) or (tstp.minute == 30)):
                continue
            
            value = -1
            try:
                value = float(splitline[2])
            except ValueError:
                value = float('NaN')
                
            try:
                households[LCLid].append([tstp,value])
            except KeyError:
                households[LCLid] = []
                households[LCLid].append([tstp,value])
        
        #block_householdseries.append(households)
        f.close()
        del(lines)
        
        skips = {}
        for key in households:
            #households[key] = radixsort_datetime(households[key])
            households[key].sort(key = lambda le: le[0])
            
            skips_ = []
            for i in range(len(households[key])-1):
                dif = households[key][i+1][0] - households[key][i][0]
        
                if dif.total_seconds() != (30*60):
                    skips_.append((i,dif))
            
            for i in range(len(skips_)):
                #Convert the time difference into multiples of 30min to count the extra spaces
                tskip = skips_[i][1]
                slots = tskip//datetime.timedelta(minutes=30) - 1
                skips_[i] = (skips_[i][0],slots)
            
            skips[key] = skips_
        
        for key in households:
            extra_slots = 0
            for skip in skips[key]:
                extra_slots += skip[1]
            
            entries = households[key]
            start_time = entries[0][0]
            ser = [float('NaN')]*(len(entries) + extra_slots)
            
            j = 0
            skip__ = skips[key]
            for i in range(len(entries)):
                ser[j] = entries[i][1]
                
                if len(skip__) > 0:                  
                    if i == skip__[0][0]:
                        j = j + skip__[0][1]
                        skip__ = skip__[1:]
                
                j = j + 1 #j is not automatically incremented
            
            #Store only the series values and the starting timestamp
            households[key] = (start_time,torch.tensor(ser,dtype=torch.float32))
        block_householdseries.append(households)
    
    cmb = {}
    for dc in block_householdseries:
        cmb = cmb | dc
    return cmb


def dispatch(dset_root,nworkers):
    # joined_path = os.path.join(dset_root,'halfhourly_dataset')
    # csv_fnames = os.listdir(joined_path)
    
    zj_path = os.path.join(dset_root,'halfhourly_dataset.zip')
    zfmod = zipfile.ZipFile(zj_path)
    zcsv_fnames = []
    for name in zfmod.namelist():
        if name[-1] != '/':
            zcsv_fnames.append(name)
    
    #par_files = [csv_fnames[x*len(csv_fnames)//nworkers : (x+1)*len(csv_fnames)//nworkers] for x in range(nworkers)]
    par_files = [zcsv_fnames[x*len(zcsv_fnames)//nworkers : (x+1)*len(zcsv_fnames)//nworkers] for x in range(nworkers)]
    pool = Pool(nworkers)
    #results = pool.starmap(p_,[(joined_path,f) for f in par_files])
    results = pool.starmap(p_,[(zj_path,f) for f in par_files])
    #results = p_(zj_path,zcsv_fnames)
    pool.close()
    pool.join()
    
    cmb = {}
    for dc in results:
        cmb = cmb | dc
    
    return cmb

if __name__ == '__main__':
    args = sys.argv
    
    dset_root = ""
    output_filename = ""
    try:
        dset_root = args[1]
        output_filename = args[2]
    except IndexError:
        dset_root = "."
        output_filename = "lsm_dict.pkl"
        
    series_dict = dispatch(dset_root,16)


    #series_dict is a dictionary with house LCLids for keys and the 
    #corresponding Tensor containing the household energy consumption series
    #as the value
    with pgzip.open(os.path.join(dset_root,output_filename + '.pgz'), "wb",thread=16,blocksize=64*10**6) as ff:
    #ff = open(os.path.join(dset_root,output_filename),'wb')
        pickle.dump(series_dict,ff)
    
    #pickle.dump(series_dict,ff)
    #ff.close()
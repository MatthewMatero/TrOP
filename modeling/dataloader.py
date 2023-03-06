import logging
logging.disable(logging.CRITICAL)
import torch
import pandas as pd
from random import shuffle
from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

import sys
import numpy as np
from collections import defaultdict, Counter

import time

def open_data(tables, mode='train', fields=['*'], group_field='group_id'):
    """
        Opens connection to DB and generates a pd.DataFrame
        from selected tables
    """

    print('Table: ', tables[0])
    print('Filter: ', tables[1])
    
    myDB = URL(drivername='mysql', host='localhost',
                database='county_opioids', query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='utf8')
    conn = engine.connect()

    main_table = tables[0]
    filter_table = tables[1] # for training/validation this table holds heldout ids
    
    select = 'select ' + ', '.join(fields) + ' from '
    if mode == 'train': # remove data for heldout set
        select += main_table 
    elif mode == 'val': # only retain data from heldout ids
        select += main_table + f" a where a.{group_field} in (select * from {filter_table})"
    elif mode == 'test':
        select += main_table + f" a where a.{group_field} not in (select * from {filter_table})"
    
    select = conn.execute(select)

    df = pd.DataFrame(select.fetchall())
    df.columns = select.keys()
    
    # we don't actually need the date field since we're already ordered by it
    df = df.drop(['year'], axis=1) 

    num_unique_groups = df[group_field].nunique()
    print(f"Loaded: {num_unique_groups} counties")
    
    try:
        print(df[df['group_id'] == 1097])
    except:
        print(df[df['cnty'] == 1097])

    conn.close()
    return df

def build_grp_dict(df, multi_var=False):
    try:
        grp_id_idx = df.columns.get_loc('group_id')
    except:
        grp_id_idx = df.columns.get_loc('cnty')
    
    try:
        opioid_idx = df.columns.get_loc('opioid_death_rate')
    except:
        opioid_idx = df.columns.get_loc('prev_death_rate')

    grp_data = df.to_numpy()
    grp_dict = defaultdict(list)
    eps_list = np.full((20,), sys.float_info.epsilon) # 20feats

    for record in grp_data:
        if multi_var:
            grp_dict[record[grp_id_idx]].append(record[grp_id_idx+1:])
        else:
            grp_dict[record[grp_id_idx]].append(record[opioid_idx])
    
    return grp_dict

def adj_diff(lst, n=1):
    for i in range(n):
        new_list = []
        adj_pairs = zip(lst[0::], lst[1::])
        for x,y in adj_pairs:
            
            new_list.append(y-x)

        lst = new_list

    return lst

def transform_data(df, history_len, num_diffs, multi_var=False, lang_only=False, nowcasting=False):
    sequences, labels = [],[]
    
    # record = (cnty, deaths, pop, crude_rate, death_rate, year)
    seq_by_cnty = build_grp_dict(df, multi_var) 
    
    for k,v in seq_by_cnty.copy().items():
        seq_by_cnty[k] = adj_diff(v, num_diffs)

    seq_lens = len(seq_by_cnty[list(seq_by_cnty.keys())[0]])
    
    max_history = seq_lens - 3  # 7yrs of data, 2 saved for test (+1 to account for data overlap)

    print('max len: ' , max_history, seq_lens)
    print('curr len: ', history_len)
    print('num exp: ', max_history - history_len + 1)
    cnty_seqs = []
    cnty_seq_labels = []
    cnties = []
    for cnty,yearly_vals in seq_by_cnty.items():
        cnty_data = []
        cnty_data = []
        cnty_labels = []
        for i in range(max_history - history_len + 1): # total number of examples per cnty
            curr_example = []
            curr_label = []
            for j in range(history_len): # accumulate history into sequence
                cnties.append(cnty)
                if lang_only:
                    curr_example.append(yearly_vals[j+i][:20]) # [:20] to debug only language feats as input
                else:
                    curr_example.append(yearly_vals[j+i])

                if j == history_len - 1:
                    if nowcasting and lang_only:
                        offset = 0
                    else:
                        offset = 1
                        
                    label = yearly_vals[j+i+offset]
                    if isinstance(label, np.ndarray):
                        label = label[-1] # target needs to be last element of list

                    curr_label.append(label)

            cnty_data.append(curr_example)
            cnty_labels.append(curr_label)

        cnty_seqs.append(cnty_data)
        cnty_seq_labels.append(cnty_labels)
    
    sequences = torch.tensor(cnty_seqs, dtype=torch.float32)
    sequences = torch.reshape(sequences, (sequences.shape[0]*sequences.shape[1], history_len, -1))
    
    labels = torch.tensor(cnty_seq_labels, dtype=torch.float32)
    labels = torch.reshape(labels, (labels.shape[0]*labels.shape[1], -1))

    print('Train Seqs: ', sequences.shape)
    print('Train labels: ', labels.shape)
    return sequences, labels, cnties

def transform_data_test(df, history_len, num_diffs=1, multi_var=False, lang_only=False, nowcasting=False):
    sequences, labels = [],[]
    seq_by_cnty = build_grp_dict(df, multi_var)
    

    for k,v in seq_by_cnty.copy().items():
        seq_by_cnty[k] = adj_diff(v, num_diffs)
    
    max_history = 2 # 7yrs of data, 2 saved for test
    cnty_seqs = []
    cnty_seq_labels = []
    cnties = []

    if nowcasting and lang_only:
        offset = 2 # we don't want to look at 2017 at all so go back to 15/16 only
    else:
        offset = 1

    for k,v in seq_by_cnty.items():
        cnty_data = []
        cnty_labels = []
        for i in reversed(range(2)):
            cnties.append(k)
            data = v[len(v) - i - history_len-1 : -i-1]
            if lang_only:
                for idx,d in enumerate(data):
                    data[idx] = data[idx][:20] # lang feats
            if nowcasting:
                for idx,d in enumerate(data):
                    data[idx] = data[idx][:21] # lang feats + prev_year
            
            label = v[-i-offset]
            if isinstance(label, np.ndarray):
                label = label[-1] # target needs to be last element of list
            
            cnty_data.append(data)
            cnty_labels.append(label)
        
        cnty_seqs.append(cnty_data)
        cnty_seq_labels.append(cnty_labels)

    sequences = torch.tensor(cnty_seqs, dtype=torch.float32)
    sequences = torch.reshape(sequences, (sequences.shape[0]*sequences.shape[1], history_len, -1))

    labels = torch.tensor(cnty_seq_labels, dtype=torch.float32)
    labels = torch.reshape(labels, (labels.shape[0]*labels.shape[1], -1))

    print('Test seq: ', sequences.shape)
    print('Test Label: ', labels.shape)
    return sequences, labels, cnties

def temporal_dataloader(table, history_len, batch_size, mode, num_diffs, multi_var, grp_field, lang_only=False, nowcasting=False, model_type=None, use_ddp=False):
    df = open_data(table, mode, group_field=grp_field)

    if mode == 'train':
        data, labels, cnties = transform_data(df, history_len, num_diffs, multi_var, lang_only, nowcasting)
    else:
        data, labels, cnties = transform_data_test(df, history_len, num_diffs, multi_var, lang_only, nowcasting)

    dataset = TensorDataset(data, labels)


    if use_ddp:
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), sampler=train_sampler, num_workers=24)

    return dataloader, dataset, cnties

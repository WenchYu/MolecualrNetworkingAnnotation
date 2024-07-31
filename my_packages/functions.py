# -*- coding: utf-8 -*-
# @Time :2022/6/17 23:03
# @Auther :Yuwenchao
# @Software : PyCharm
'''

Basic Functions:
    ex_content()
    ex_startswith()
    mirror_plotting()
    calculate_ppm()
    is_between()
'''

import os
import re
import ast
import time
import math
import json
import linecache
import pandas as pd
import numpy as np
import spectrum_utils.spectrum as sus
from collections import namedtuple



TopK=namedtuple('topk',['index','number'])
def ex_startswith(file, start_txt):
    '''
    因为原来是为了提取mgf中所有文件的信息，所以返回的是一个列表
    使用了列表推到式，Extracting information after keyword as **float** or **str**
    :param filename: Path including suffix of the **text** file you intend to slice
    :param start_txt: Starting keyword
    :return: A list containing content after keywords
    '''
    with open(file, 'r') as f:
        content=[line[len(start_txt):].rstrip() for line in f if line.startswith(start_txt)]
        return content

def ex_content(filename,start_txt,end_txt):
    '''
        根据关键词提取二级质谱 **float**

        :param filename: Path including suffix of the **text** file you intend to slice
        :param start_txt: Starting keyword
        :param end_txt: Ending Keyword
        :return: A list contain lists of sliced content, like[[],[],...,[]]
        '''
    start_num = []
    end_num = []
    dest_content=[]
    linenumber = 1
    with open ('{}'.format(filename),'r') as f:
        for eachline in f:
            s=re.match(start_txt,eachline)# 使用正则表达式匹配每行内容
            e=re.search(end_txt,eachline)
            if s is not None:# 如果不为None，将当前行号添加到 start_num 列表中
                start_num.append(linenumber)
            elif e is not None:
                end_num.append(linenumber)
            linenumber+=1
        index=list(zip(start_num,end_num))
        index_size=len(index)
        for i in range(index_size):
            start=index[i][0]# 获取元组中的起始行行号
            end=index[i][1]# 获取元组中的结束行行号
            destlines=[]# 创建一个空列表 destlines，用来存储提取出的目标行的内容
            try:# 尝试使用行号获取对应的行内容
                if 'MERGED' in linecache.getlines(filename)[start]:# 判断当前行是否包含字符串'MERGED'
                    cache_destlines =linecache.getlines(filename)[start + 1:end - 1]
                    for destline in cache_destlines:
                        destlines.append([float(destline.rstrip().split(' ')[0]),float(destline.rstrip().split(' ')[1])])
                else:
                    cache_destlines = linecache.getlines(filename)[start:end - 1]
                    for destline in cache_destlines:
                        destlines.append([float(destline.rstrip().split(' ')[0]), float(destline.rstrip().split(' ')[1])])
            except:
                cache_destlines = linecache.getlines(filename)[start:end - 1]
                for destline in cache_destlines:
                    destlines.append([float(destline.rstrip().split('\t')[0]), float(destline.rstrip().split('\t')[1])])

            dest_content.append(destlines)
        return  dest_content

def ex_spectra(file, start_txt, end_txt, skip_words=None):
    '''
    Horizontal and vertical coordinates of tandem mass
    :param file:
    :param start_txt:
    :param end_txt:
    :param skip_words:
    :return: A list contain lists of sliced content, like[[],[],...,[]],and converting to an array
    '''
    if skip_words == None:
        skip_words = []
    spectra = []
    with open(file, 'r') as f:
        lines = f.readlines()
        start_idx = 0
        for i in range(len(lines)):
            if start_txt in lines[i]:
                if any(word in lines[i + 1] for word in skip_words):
                    start_idx = i+2
                else:
                    start_idx = i+1
            elif end_txt in lines[i]:
                spectrum = ''.join(lines[start_idx:i])
                spectra_list = spectrum.split('\n')[:-1]
                temp=[]
                for s in spectra_list:
                    m_z, intensity = s.split()
                    temp.append([float(m_z), float(intensity)])
                temp = np.array(temp,dtype=np.float64)
                spectra.append(temp)
    return spectra

def mgf_process(mgf_file):
    '''
    提取mgf文件中的'FEATURE_ID=', 'PEPMASS=', 和二级质谱
    :param mgf_file:
    :return: id<str> pepmass<str>, ms2<np array>
    '''
    id_txt = 'FEATURE_ID='
    # id_txt = 'TITLE='
    id = ex_startswith(mgf_file, id_txt)

    pepmass_txt = 'PEPMASS='
    pepmass = ex_startswith(mgf_file, pepmass_txt)

    charge_txt = 'CHARGE='
    charge = ex_startswith(mgf_file, charge_txt)
    charge = [s.replace('+', '') for s in charge]

    start_txt = 'MSLEVEL=2'
    # start_txt = 'CHARGE'
    end_txt = 'END'
    ms2 = ex_spectra(mgf_file, start_txt, end_txt, skip_words=['MERGED'])

    # 储存成dataframe，读取更快
    exp_info = pd.DataFrame({
        'id': id
        ,'pepmass': pepmass
        ,'charge' : charge
        ,'ms2': ms2
    })
    # 使用apply方法检查每个列表是否为空
    exp_info = exp_info[exp_info['ms2'].apply(len) > 1]  # 删除空列表
    exp_info = exp_info.reset_index(drop=True)  # 重新编写索引

    return exp_info

def get_mgf_info(mgf_info,mgf_id):
    '''

    :param mgf_info:
    :param id:
    :return:pepmass<float>, spec<np.adarray>, spectrum<spectrum_utils object>
    '''
    if not mgf_info.empty:
        pepmass = float(mgf_info[mgf_info['id'] == mgf_id]['pepmass'].iloc[0])
        charge = int(mgf_info[mgf_info['id'] == mgf_id]['charge'].iloc[0])
        spec = mgf_info[mgf_info['id'] == mgf_id]['ms2'].iloc[0]
        mz = np.array(spec[:, 0])
        spectrum = sus.MsmsSpectrum(identifier=mgf_id
                                    , precursor_mz=pepmass
                                    , precursor_charge=charge
                                    , mz=mz
                                    , intensity=spec[:, 1])
        return {'pepmass': pepmass, 'spec': spec, 'spectrum': spectrum, 'charge': charge, 'id': mgf_id}
    else:
        raise ValueError(f"No data found for mgf_id: {mgf_id}")

def get_gnps_info(gnps_info, gnps_id):
    '''

    :param isdb_info:
    :param id:
    :return:
    '''
    keys_to_retrieve = ['smiles', 'pepmass', 'ms2','charge']
    values = [gnps_info[gnps_id][key] for key in keys_to_retrieve]
    smiles, pepmass, spec, charge = values
    # string convertion
    pepmass = float(pepmass)
    charge = int(charge)
    spec = np.asarray(ast.literal_eval(spec))
    mz = np.array(spec[:, 0])
    spectrum = sus.MsmsSpectrum(identifier=f'{gnps_id}'
                                 , precursor_mz=pepmass
                                 , precursor_charge=charge
                                 , mz=mz
                                 , intensity=spec[:, 1])

    return {'smiles': smiles, 'pepmass': pepmass
        , 'spec': spec, 'spectrum': spectrum,'charge': charge}

def get_isdb_info(isdb_info, is_id):
    '''

    :param isdb_info:
    :param id:
    :return:
    '''
    keys_to_retrieve = ['smiles', 'pepmass', 'energy0_ms2', 'energy1_ms2', 'energy2_ms2']
    values = [isdb_info[is_id][key] for key in keys_to_retrieve]
    smiles, pepmass, e0spec, e1spec, e2spec = values
    # string convertion
    pepmass = float(pepmass)
    e0spec = np.asarray(ast.literal_eval(e0spec))
    e1spec = np.asarray(ast.literal_eval(e1spec))
    e2spec = np.asarray(ast.literal_eval(e2spec))

    mz0 = np.array(e0spec[:, 0])
    spectrum0 = sus.MsmsSpectrum(identifier=f'e0_{is_id}'
                                 , precursor_mz=pepmass
                                 , precursor_charge=1
                                 , mz=mz0
                                 , intensity=e0spec[:, 1])
    mz1 = np.array(e1spec[:, 0])
    spectrum1 = sus.MsmsSpectrum(identifier = f'e1_{is_id}'
                                 , precursor_mz=pepmass
                                 , precursor_charge=1
                                 , mz=mz1
                                 , intensity=e1spec[:, 1])
    mz2 = np.array(e2spec[:, 0])
    spectrum2 = sus.MsmsSpectrum(identifier=f'e2_{is_id}'
                                 , precursor_mz=pepmass
                                 , precursor_charge=1
                                 , mz=mz2
                                 , intensity=e2spec[:, 1])

    return {'smiles': smiles, 'pepmass': pepmass
        , 'e0spec': e0spec, 'e1spec': e1spec, 'e2spec': e2spec
        , 'e0spectrum': spectrum0, 'e1spectrum': spectrum1, 'e2spectrum': spectrum2}

def df_preprocess(filename):
    '''
    标准化dataframe ： 删除空列，标准化index
    :param filename:
    :return:
    '''
    # 读取CSV文件
    df = pd.read_csv(filename,low_memory=False)

    # 删除空列
    # df = df.dropna(axis=1, how='all',subset = None)

    if  df.index[-1] != len(df)-1:
        df.index.name = ''
        df.reset_index(inplace=True)

    # 返回DataFrame
    return df

def calculate_ppm(query_mass_value: float, reference_mass_value: float) -> float:
    '''
    Calculate parts per million (ppm) for query and reference mass values.

    :param query_mass_value: The mass value of the query
    :param reference_mass_value: The mass value of the reference
    :return: The ppm value
    '''
    if not isinstance(query_mass_value, (int, float)) or not isinstance(reference_mass_value, (int, float)):
        raise TypeError('Input parameters must be numbers.')

    if reference_mass_value != 0:
        ppm = abs((query_mass_value - reference_mass_value) / reference_mass_value * 1e6)
    else:
        ppm = math.inf

    return ppm

def db_parsing():
    '''
    默认方式解析数据库
    :return:
    '''
    isdb_file = './msdb/isdb_info.json'
    gnps_file = './msdb/edb_info.json'
    with open(isdb_file, 'r') as f:
        isdb_info = json.load(f)

    with open(gnps_file, 'r') as f1:
        gnps_info = json.load(f1)
    return isdb_info, gnps_info
if __name__ == '__main__':
    t = time.time()
    os.chdir('/Users/hehe/desktop')


    print(f'居然花了:{(time.time() - t)/60:.2f}s')
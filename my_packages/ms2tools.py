# -*- coding: utf-8 -*-
# @Time :2023/3/18 23:54
# @Auther :Yuwenchao
# @Software : PyCharm
'''
docstrings
'''
import os
import time
import json
import ast
import heapq
import pandas as pd
import numpy as np
import networkx as nx
import spectrum_utils.spectrum as sus
import spectral_entropy
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from spectral_entropy import similarity, calculate_entropy
from my_packages import functions
from my_packages import spectrum_alignment
from my_packages.spectrum_alignment import find_match_peaks_efficient, convert_to_peaks
from my_packages.similarity import modified_cosine, neutral_loss
from my_packages.config import arg_parse


def spectral_entropy_calculating(args):
    """
    计算样本中每个谱图的光谱熵，并将结果保存到新的 CSV 文件中。
    args.mgf_file: MGF 文件路径。
    args.quant_file: 待计算MS2 信息熵的 CSV 文件路径。
    args.output_path: 结果文件保存路径
    """
    # 处理 MGF 文件
    exp_info = functions.mgf_process(args.mgf_file)
    # 加载待处理的 CSV 文件
    quant_df = functions.df_preprocess(args.quant_file)

    # 添加 spectral_entropy 列
    quant_df['spectral_entropy'] = np.nan

    # 计算每个谱图的光谱熵，并将结果保存到 quant_df 中
    for i in range(len(quant_df)):
        try:
            id = str(quant_df.loc[i,'row ID'])
            result = functions.get_mgf_info(exp_info,id)
            pepmass = result['pepmass']
            spec = result['spec']
            SE = calculate_entropy(spectrum = spec, max_mz = pepmass)
            quant_df.loc[i,'spectral_entropy'] = SE
        except:
            quant_df.loc[i, 'spectral_entropy'] = 0


    # 保存结果到新的 CSV 文件
    quant_name = os.path.splitext(os.path.basename(args.quant_file))[0]
    result_dir = os.path.join(args.output, f'{quant_name}_result')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, os.path.basename(args.quant_file))
    quant_df.to_csv(result_path,index=None)
    # print('Spectral entropy calculation finished!')

def match_mz(quant_df_row, msdb_df, mz_column='row m/z',np_ms1_match_threshld = 5):
    '''
    根据quant csv搜索ms1db，如果有match的
    :param quant_df_row: mzmine导出的quant.csv文件
    :param msdb_df: 读数据库csv文件成dataframe
    :param mz_column: m/z列名
    :param np_ms1_match_threshld: MS1匹配
    :return:一个元组([...],[...],[...]) ，不符合条件则返回元组中的列表为空([],[],[])
    '''

    hits_id = []
    hits_smiles = []
    for j in range(len(msdb_df.index)):
        ppm_H = functions.calculate_ppm(quant_df_row[mz_column], msdb_df.loc[j,'m+h'])
        ppm_Na = functions.calculate_ppm(
                quant_df_row[mz_column], msdb_df.loc[j,'m+na'])
        if ppm_H < np_ms1_match_threshld or ppm_Na < np_ms1_match_threshld:
            hits_id.append(msdb_df.id[j])
            hits_smiles.append(msdb_df.smiles[j])
    if not hits_id or not hits_smiles:
        hits_id.append(None)
        hits_smiles.append(None)
    return hits_id, hits_smiles

def match_edb_mz(quant_df_row,edb_df,mz_column='row m/z',edb_ms1_match_threshold = 5):
    '''

    :param quant_df_row:
    :param edb_df:
    :param mz_column:
    :return:
    '''
    hits_id = []
    hits_smiles = []
    for j in range(len(edb_df.index)):
        ppm = functions.calculate_ppm(quant_df_row[mz_column], edb_df['pepmass'][j])
        if ppm < edb_ms1_match_threshold:
            hits_id.append(edb_df.id[j])
            hits_smiles.append(str(edb_df.smiles[j]))
    if not hits_id or not hits_smiles:
        hits_id.append(None)
        hits_smiles.append(None)
    return hits_id, hits_smiles

def ms1_match(args):
    '''


    args.npms1_file:
    args.edbms1_file:
    args.quant_file:
    args.output_path:
    :return:
    '''
    np_ppm = args.pepmass_match_tolerance
    edb_ppm = args.pepmass_match_tolerance # parallel需要np_ppm, edb_ppm两个
    np_msdb_df = pd.read_csv(args.npms1_file,low_memory=False)
    edb_df = functions.df_preprocess(args.edbms1_file)
    quant_df = functions.df_preprocess(args.quant_file)

    n_jobs = os.cpu_count()
    np_results = Parallel(n_jobs=n_jobs)(
        delayed(match_mz)(quant_df_row, np_msdb_df,np_ms1_match_threshld = np_ppm) for quant_df_row in tqdm(quant_df.to_dict('records')))

    np_match_rows = []
    edb_match_rows = []
    for i, (hits_id, hits_smiles) in enumerate(np_results):
        for j in range(len(hits_id)):
            np_match_row = {'row ID': quant_df.at[i, 'row ID'], 'row m/z': quant_df.at[i, 'row m/z'],
                            'match_id': hits_id[j], 'match_smiles': hits_smiles[j]}
            np_match_rows.append(np_match_row)
    np_match_df = pd.DataFrame(np_match_rows)

    edb_results = Parallel(n_jobs=n_jobs)(
        delayed(match_edb_mz)(quant_df_row, edb_df, edb_ms1_match_threshold = edb_ppm) for quant_df_row in tqdm(quant_df.to_dict('records')))
    for i, (hits_id, hits_smiles) in enumerate(edb_results):
        for j in range(len(hits_id)):
            edb_match_row = {'row ID': quant_df.at[i, 'row ID'], 'row m/z': quant_df.at[i, 'row m/z'],
                              'match_id': hits_id[j], 'match_smiles': hits_smiles[j]}
            edb_match_rows.append(edb_match_row)
    edb_match_df = pd.DataFrame(edb_match_rows)

    quant_name = os.path.splitext(os.path.basename(args.quant_file))[0]
    result_dir = os.path.join(args.output, f'{quant_name}_result')
    os.makedirs(result_dir, exist_ok=True)
    np_result_path = os.path.join(result_dir, f'npMS1match_{os.path.basename(args.quant_file)}')
    edb_result_path = os.path.join(result_dir, f'edbMS1match_{os.path.basename(args.quant_file)}')

    np_match_df.to_csv(np_result_path, index=False)
    edb_match_df.to_csv(edb_result_path, index=False)

    quant_df['npms1_id'] = np.nan
    quant_df['noms1_smiles'] = np.nan
    quant_df['edbms1_id'] = np.nan
    quant_df['edbms1_smiles'] = np.nan
    for i, (hits_id, hits_smiles) in enumerate(np_results):
        if hits_id is not None and all(isinstance(x, str) for x in hits_id):
            quant_df.at[i, 'npms1_id'] = ';'.join([x or '' for x in hits_id])
            quant_df.at[i, 'npms1_smiles'] = ';'.join(hits_smiles)

    for i, (hits_id, hits_smiles) in enumerate(edb_results):
        if hits_id is not None and all(isinstance(x, str) for x in hits_id):
            quant_df.at[i, 'edbms1_id'] = ';'.join([x or '' for x in hits_id])
            quant_df.at[i, 'edbms1_smiles'] = ';'.join(hits_smiles)

    ms1_result_path = os.path.join(result_dir, f'MS1match_{os.path.basename(args.quant_file)}')
    quant_df.to_csv(ms1_result_path, index=False)
    print('MS1 matching finished!')

def ISDB_MS2_match(args):
    '''

    args.isdb_file:
    args.quant_file:
    args.mgf_file:
    args.sim_method:
    :return:
    '''
    with open(args.isdb_file) as f:
        isdb_info = json.load(f)

    exp_info = functions.mgf_process(args.mgf_file)
    quant_name = os.path.splitext(os.path.basename(args.quant_file))[0]
    parent_dir = os.path.join(args.output, f'{quant_name}_result')
    os.makedirs(parent_dir, exist_ok=True)

    np_result_path = os.path.join(parent_dir, f'npMS1match_{os.path.basename(args.quant_file)}')
    np_ms1_match_df = functions.df_preprocess(np_result_path)

    np_ms1_match_df['mps0'] = np.nan
    np_ms1_match_df['pp0'] = np.nan
    np_ms1_match_df['pair_similarity0'] = np.nan
    np_ms1_match_df['mps1'] = np.nan
    np_ms1_match_df['pp1'] = np.nan
    np_ms1_match_df['pair_similarity1'] = np.nan
    np_ms1_match_df['mps2'] = np.nan
    np_ms1_match_df['pp2'] = np.nan
    np_ms1_match_df['pair_similarity2'] = np.nan

    for i in trange(len(np_ms1_match_df)):
        row_id = str(np_ms1_match_df.loc[i,'row ID']) # quant.csv的row ID是int，提取数据库获得的row ID是str
        match_id = str(np_ms1_match_df.loc[i,'match_id'])
        if match_id != 'nan':
            try:  # quant.csv中的有些feature，没有二级,先用try:except顶着
                exp_pm = float(exp_info[exp_info['id'] == row_id].pepmass.iloc[0])
                exp_charge = int(exp_info[exp_info['id'] == row_id].charge.iloc[0])
                exp_ms2 = np.asarray(exp_info[exp_info['id'] == row_id].ms2.iloc[0])
                exp_ms2 = spectral_entropy.clean_spectrum(exp_ms2, max_mz=exp_pm + 0.01)
                exp_mz = np.array(exp_ms2[:, 0], dtype=np.float64)
                exp_intensity = np.array(exp_ms2[:, 1], dtype=np.float64)
                exp_spectrum = sus.MsmsSpectrum(identifier=row_id, precursor_mz=exp_pm + 0.01, precursor_charge=exp_charge,
                                                mz=exp_mz,intensity=exp_intensity)

                is_pm = exp_pm
                is_smile = isdb_info[match_id]['smiles']

                e0_ms2 = np.asarray(ast.literal_eval(isdb_info[match_id]['energy0_ms2']))
                e0_mz = np.array(e0_ms2[:,0],dtype=np.float64)
                e0_intensity = np.array(e0_ms2[:, 0], dtype=np.float64)
                e0_spectrum = sus.MsmsSpectrum(identifier=f'e0_{match_id}', precursor_mz=is_pm + 0.01, precursor_charge=1,
                                                mz=e0_mz,intensity=e0_intensity)
                e1_ms2 = np.asarray(ast.literal_eval(isdb_info[match_id]['energy1_ms2']))
                e1_mz = np.array(e1_ms2[:, 0], dtype=np.float64)
                e1_intensity = np.array(e1_ms2[:, 0], dtype=np.float64)
                e1_spectrum = sus.MsmsSpectrum(identifier=f'e1_{match_id}', precursor_mz=is_pm + 0.01, precursor_charge=1,
                                               mz=e1_mz,intensity=e1_intensity)

                e2_ms2 = np.asarray(ast.literal_eval(isdb_info[match_id]['energy2_ms2']))
                e2_mz = np.array(e2_ms2[:, 0], dtype=np.float64)
                e2_intensity = np.array(e2_ms2[:, 0], dtype=np.float64)
                e2_spectrum = sus.MsmsSpectrum(identifier=f'e2_{match_id}', precursor_mz=is_pm + 0.01, precursor_charge=1,
                                               mz=e2_mz,
                                               intensity=e2_intensity)

                shift = abs(is_pm-exp_pm)
                exp_peaks = len(exp_ms2)
                sim0, sim1, sim2 = 0.0, 0.0, 0.0
                mps0, mps1, mps2 = 0, 0, 0
                pp0, pp1, pp2 = 0.0, 0.0, 0.0
                if args.library_matching_method == 'modified_cosine_similarity':
                    try:
                        result0 = modified_cosine(exp_spectrum, e0_spectrum, fragment_mz_tolerance=0.05)
                        sim0 = result0.score
                        mps0 = result0.matches
                        pp0 = mps0/exp_peaks
                        result1 = modified_cosine(exp_spectrum, e1_spectrum, fragment_mz_tolerance=0.05)
                        sim1 = result1.score
                        mps1 = result1.matches
                        pp1 = mps1 / exp_peaks
                        result2 = modified_cosine(exp_spectrum, e2_spectrum, fragment_mz_tolerance=0.05)
                        sim2 = result2.score
                        mps2 = result2.matches
                        pp2 = mps2 / exp_peaks
                    except:
                        pass

                elif args.library_matching_method == 'netural_loss':
                    try:
                        result0 = neutral_loss(exp_spectrum, e0_spectrum, fragment_mz_tolerance=0.05)
                        sim0 = result0.score
                        mps0 = result0.matches
                        pp0 = mps0 / exp_peaks
                        result1 = neutral_loss(exp_spectrum, e1_spectrum, fragment_mz_tolerance=0.05)
                        sim1 = result1.score
                        mps1 = result1.matches
                        pp1 = mps1 / exp_peaks
                        result2 = neutral_loss(exp_spectrum, e2_spectrum, fragment_mz_tolerance=0.05)
                        sim2 = result2.score
                        mps2 = result2.matches
                        pp2 = mps2 / exp_peaks
                    except:
                        pass
                else:
                    mps0 = len(find_match_peaks_efficient(convert_to_peaks(exp_ms2)
                                                          , convert_to_peaks(e0_ms2), shift, 0.05))
                    sim0 = similarity(exp_ms2, e0_ms2, method=args.library_matching_method, ms2_da=0.05)
                    pp0 = mps0 / exp_peaks
                    mps1 = len(find_match_peaks_efficient(convert_to_peaks(exp_ms2)
                                                          , convert_to_peaks(e1_ms2), shift, 0.05))
                    sim1 = similarity(exp_ms2, e1_ms2, method=args.library_matching_method, ms2_da=0.05)
                    pp1 = mps1 / exp_peaks
                    mps2 = len(find_match_peaks_efficient(convert_to_peaks(exp_ms2)
                                                          , convert_to_peaks(e2_ms2), shift, 0.05))
                    sim2 = similarity(exp_ms2, e2_ms2, method=args.library_matching_method, ms2_da=0.05)
                    pp2 = mps2 / exp_peaks

                np_ms1_match_df.loc[i, 'pair_similarity0'] = sim0
                np_ms1_match_df.loc[i, 'mps0'] = mps0
                np_ms1_match_df.loc[i, 'pp0'] = pp0
                np_ms1_match_df.loc[i, 'pair_similarity1'] = sim1
                np_ms1_match_df.loc[i, 'mps1'] = mps1
                np_ms1_match_df.loc[i, 'pp1'] = pp1
                np_ms1_match_df.loc[i, 'pair_similarity2'] = sim2
                np_ms1_match_df.loc[i, 'mps2'] = mps2
                np_ms1_match_df.loc[i, 'pp2'] = pp2
                is_ms2_path = os.path.join(parent_dir, row_id, f'{match_id}.mgf')
                with open(is_ms2_path, 'w') as f:
                    f.write('BEGIN IONS\n')
                    f.write(f'ID={match_id}\n')
                    f.write(f'PEPMASS={is_pm}\n')
                    f.write(f'SMILES={is_smile}\n')
                    f.write('ENERGY\n')
                    for item in e0_ms2:
                        f.write("%s %s\n" % (item[0], item[1]))
                    f.write('ENERGY1\n')
                    for item in e1_ms2:
                        f.write("%s %s\n" % (item[0], item[1]))
                    f.write('ENERGY2\n')
                    for item in e2_ms2:
                        f.write("%s %s\n" % (item[0], item[1]))
                    f.write('END IONS\n')
            except:
                pass
    np_ms1_match_df.to_csv(np_result_path)

def EDB_MS2_match(args):
    '''
    args.output
    args.edbms2_file:
    args.quant_file:
    args.mgf_file:
    :return:
    '''
    with open (args.edbms2_file,'r') as f:
        edbms2_info=json.load(f) # EDB_MS2 json file


    quant_name = os.path.splitext(os.path.basename(args.quant_file))[0]
    parent_dir = os.path.join(args.output, f'{quant_name}_result') # 结果文件夹路径 ：output/example_quant/
    os.makedirs(parent_dir, exist_ok=True)

    edb_result_path = os.path.join(parent_dir, f'edbMS1match_{os.path.basename(args.quant_file)}')
    edb_ms1_df = functions.df_preprocess(edb_result_path) # GNPS_shared_code MS1 match result
    exp_info = functions.mgf_process(args.mgf_file)

    edb_ms1_df['mps'] = np.nan
    edb_ms1_df['pair_similarity'] = np.nan
    edb_ms1_df['pp'] = np.nan
    for i in trange(len(edb_ms1_df)):
        row_id = str(edb_ms1_df.loc[i,'row ID'])
        match_id = str(edb_ms1_df.loc[i,'match_id'])
        if match_id != 'nan':
            try:  # quant.csv中的有些feature，没有二级,先用try:except顶着
                exp_pm = float(exp_info[exp_info['id'] == row_id].pepmass.iloc[0])
                exp_ms2 = exp_info[exp_info['id'] == row_id].ms2.iloc[0]
                exp_ms2 = spectral_entropy.clean_spectrum(exp_ms2, max_mz=exp_pm+0.01)
                exp_charge =  int(exp_info[exp_info['id'] == row_id].charge.iloc[0])
                exp_mz = np.array(exp_ms2[:, 0], dtype=np.float64)
                exp_intensty = np.array(exp_ms2[:, 1], dtype=np.float64)
                exp_spectrum = sus.MsmsSpectrum(identifier=row_id, precursor_mz=exp_pm+0.01, precursor_charge=exp_charge, mz=exp_mz,
                                             intensity=exp_intensty)

                edb_pm = float(edbms2_info[match_id]['pepmass'])
                edb_smiles = edbms2_info[match_id]['smiles']
                edb_ms2 = np.asarray(ast.literal_eval(edbms2_info[match_id]['ms2']))
                edb_ms2 = spectral_entropy.clean_spectrum(edb_ms2, max_mz=edb_pm+0.01)
                edb_charge = int(edbms2_info[match_id]['charge'])
                edb_mz = np.array(edb_ms2[:, 0], dtype=np.float64)
                edb_intensty = np.array(edb_ms2[:, 1], dtype=np.float64)
                edb_spectrum = sus.MsmsSpectrum(identifier=match_id, precursor_mz=edb_pm+0.01, precursor_charge=edb_charge, mz=edb_mz,
                                             intensity=edb_intensty)

                shift = abs(exp_pm-edb_pm)
                exp_peaks = len(exp_ms2)
                sim = 0.0
                mps = 0
                pp = 0.0
                if args.library_matching_method == 'modified_cosine_similarity':
                    try:
                        result = modified_cosine(exp_spectrum, edb_spectrum, fragment_mz_tolerance=0.05)
                        sim = result.score
                        mps = result.matches
                        pp = mps/exp_peaks
                    except:
                        pass

                elif args.library_matching_method == 'netural_loss':
                    try:
                        result = neutral_loss(exp_spectrum, edb_spectrum, fragment_mz_tolerance=0.05)
                        sim = result.score
                        mps = result.matches
                        pp = mps / exp_peaks
                    except:
                        pass
                else:
                    mps = len(find_match_peaks_efficient(convert_to_peaks(exp_ms2)
                                                          , convert_to_peaks(edb_ms2), shift, 0.05))
                    sim = similarity(exp_ms2, edb_ms2, method=args.library_matching_method, ms2_da=0.05)
                    pp = mps/exp_peaks

                edb_ms1_df.loc[i, 'pair_similarity'] = sim
                edb_ms1_df.loc[i, 'mps'] = mps
                edb_ms1_df.loc[i,'pp'] = pp

                edb_ms2_path = os.path.join(parent_dir, row_id, f'{match_id}.mgf')
                with open(edb_ms2_path, 'w') as f:
                    f.write('BEGIN IONS\n')
                    f.write(f'ID={match_id}\n')
                    f.write(f'PEPMASS={edb_pm}\n')
                    f.write(f'SMILES={edb_smiles}\n')
                    f.write('ENERGY\n')
                    for item in edb_ms2:
                        f.write("%s %s\n" % (item[0], item[1]))
                    f.write('END IONS\n')
            except:
                pass

    edb_ms1_df.to_csv(edb_result_path)
    print('MS2 matching finished!')

def mn_curating(G, topk):
    '''
    If the degree of node i exceeds K, keep only the top K most similar neighbors
    :param G: Undirected graph created by networkx
    :param topk: Max degree of a node
    :return: An curated G
    '''
    node_ids = list(G.nodes())
    for node_id in node_ids:
        if len(G[node_id]) > topk:
            edges = list(G.edges(node_id))
            result = \
                heapq.nlargest(10,
                               [(data.get('pair_similarity', 0), neighbor) for neighbor, data in G[node_id].items()])
            topk_edges = [t[1] for t in result]
            for edge in edges:
                if edge[1] not in topk_edges:
                    G.remove_edge(edge[0], edge[1])
    return G

def self_clustering(args):
    '''
    args.output
    args.quant_file
    args.mgf_file

    :param args:
    :return:
    '''
    parent_folder = f'{args.output}/{os.path.splitext(os.path.basename(args.quant_file))[0]}_result'
    exp_info = functions.mgf_process(args.mgf_file)

    G = nx.Graph()  # Creating undirected graph
    for i, (id1, pm1, charge1, spec1) in exp_info.iterrows():
        pm1 = float(pm1)
        node_attr = {'pepmass': pm1}
        G.add_node(id1, **node_attr)  # add nodes and attributes

     # parse exp ms2
    for i, (id1, pm1, charge1, spec1) in tqdm(exp_info.iterrows(), total = len(exp_info)):
        # try:
        pm1 = float(pm1)
        charge1= int(charge1)
        mz1 = np.array(spec1[:, 0], dtype=np.float64)
        spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1, precursor_charge=charge1, mz=mz1,
                                     intensity=spec1[:, 1])
        peaks1 = len(spec1)
        if args.spectrum_clean:
            spec1 = spectral_entropy.clean_spectrum(spec1
                                                    , max_mz = pm1 - 0.01
                                                    , noise_removal = 0.01
                                                    , ms2_ppm = 5
                                                    , ms2_da = 0.02
                                                    )
            spectrum1 = spectrum1.filter_intensity(min_intensity=0.01) \
                .set_mz_range(0, spectrum1.precursor_mz).remove_precursor_peak(0.1, "Da")
            peaks1 = len(spec1)

        for j, (id2, pm2, charge2, spec2) in exp_info.iloc[:i, ].iterrows():
            pm2 = float(pm2)
            charge2 = int(charge2)
            mz2 = np.array(spec2[:, 0], dtype=np.float64)
            spectrum2 = sus.MsmsSpectrum(identifier=id2, precursor_mz=pm2, precursor_charge=charge2, mz=mz2
                                         , intensity=spec2[:, 1])
            if args.spectrum_clean:
                spec2 = spectral_entropy.clean_spectrum( spec2
                                                        , max_mz = pm2 - 0.01
                                                        , noise_removal = 0.01
                                                        , ms2_ppm = 5
                                                        , ms2_da = 0.02
                                                                    )
                spectrum2 = spectrum2.filter_intensity(min_intensity=0.01) \
                                        .set_mz_range(0, spectrum2.precursor_mz).remove_precursor_peak(0.1, "Da")

            shift = abs(pm1 - pm2)
            sim = 0.0
            mps = 0
            pp = 0.0
            if args.self_clustering_method == 'modified_cosine_similarity':
                try:
                    result = modified_cosine(spectrum1,spectrum2,fragment_mz_tolerance=0.02)
                    sim = result.score
                    mps = result.matches
                    pp = mps/peaks1
                except:
                    pass

            elif args.self_clustering_method == 'neutral_loss':
                try:
                    result = neutral_loss(spectrum1, spectrum2, fragment_mz_tolerance=0.02)
                    sim = result.score
                    mps = result.matches
                    pp = mps/peaks1
                except:
                    pass

            else:
                try:
                    sim = similarity(spec1, spec2, method=args.self_clustering_method, ms2_ppm=10 ,ms2_da = 0.05)
                    mps = len(
                        spectrum_alignment.find_match_peaks_efficient(
                            spectrum_alignment.convert_to_peaks(spec1)
                            , spectrum_alignment.convert_to_peaks(spec2)
                            , shift = shift
                            , tolerance=0.02)
                            )
                    pp = mps/peaks1
                except:
                    pass
            if sim >= args.self_clustering_similarity \
                    and mps >= args.self_clustering_peaks:
                    edge_attr = {'pair_similarity': sim, 'matched_peaks': mps,'peak_percentage':pp}
                    G.add_edge(id1, id2, **edge_attr)

    G = mn_curating(G,args.top_k)
    print('Self clustering finished!')
    # return G
    MN_file = os.path.join(parent_folder,
                           f'{os.path.splitext(os.path.basename(args.mgf_file))[0]}_{args.self_clustering_method}_{args.self_clustering_similarity}_{args.self_clustering_peaks}.graphml')
    nx.write_graphml(G, MN_file)

def molecular_generation(args):

    '''
    自聚类 + 根据结果给node给结果分配level
    A : 所有MS1匹配不上的 #CCCCCC
    B1 : edb MS2 + match_unwell #B3E2CD
    B2 : edb MS2 + match_well #1B7837
    C1 : in silico MS2 + match_unwell #FDDAEC
    C2 : in silico MS2 + match_well #C51B7D

    先跟47w+58w的MS1比较，5ppm以内的，继续比较二级质谱；
    B1，C1(或C2)同时存在的情况下，会使用B1覆盖C1(或C2)；
    B2，C2(或C1)同时存在的情况下，会使用B2覆盖C2(或C1)；
    B2, C1同时存在的情况，颜色会显示成B2，只保留匹配上的edb二级

    args.quant_file:
    args.mgf_file:
    args.sim_method:
    args.pair_similarity:
    args.shared_peaks:
    args.activate_clean_spectrum:
    :return:
    '''
    parent_folder = f'{args.output}/{os.path.splitext(os.path.basename(args.quant_file))[0]}_result'
    quant_df = functions.df_preprocess(args.quant_file)
    exp_info = functions.mgf_process(args.mgf_file)
    row_ids = [int(x) for x in exp_info['id'].values.tolist()]
    G = nx.MultiGraph()  # Creating undirected graph
    for i, (id1, pm1, charge1, spec1) in exp_info.iterrows():
        pm1 = float(pm1)
        node_attr = {'pepmass': pm1}
        G.add_node(id1, **node_attr)  # add nodes and attributes

    # parse exp ms2
    for i, (id1, pm1, charge1, spec1) in tqdm(exp_info.iterrows(), total=len(exp_info)):
        # try:
        pm1 = float(pm1)
        charge1 = int(charge1)
        spec1 = spectral_entropy.clean_spectrum(spec1, max_mz=pm1 - 0.01, noise_removal=0.01)
        mz1 = np.array(spec1[:, 0], dtype=np.float64)
        intensity1 = np.array(spec1[:, 1], dtype=np.float64)
        spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 , precursor_charge= charge1
                                     , mz = mz1,intensity=intensity1).remove_precursor_peak(0.01, "Da")
        peaks1 = len(spec1)
        for j, (id2, pm2, charge2, spec2) in exp_info.iloc[:i, ].iterrows():
            pm2 = float(pm2)
            charge2 = int(charge2)
            spec2 = spectral_entropy.clean_spectrum(spec2, max_mz=pm2 - 0.01, noise_removal=0.01)
            mz2 = np.array(spec2[:, 0], dtype=np.float64)
            intensity2 = np.array(spec2[:, 1], dtype=np.float64)
            spectrum2 = sus.MsmsSpectrum(identifier=id2, precursor_mz=pm2 , precursor_charge=charge2, mz=mz2,
                                         intensity=intensity2).remove_precursor_peak(0.01, "Da")


            shift = abs(pm1 - pm2)
            sim = 0.0
            mps = 0
            pp = 0.0
            if args.self_clustering_method == 'modified_cosine':
                try:
                    result = modified_cosine(spectrum1, spectrum2, fragment_mz_tolerance=0.02)
                    sim = result.score
                    mps = result.matches
                    pp = mps / peaks1
                except:
                    pass

            elif args.self_clustering_method == 'neutral_loss':
                try:
                    result = neutral_loss(spectrum1, spectrum2, fragment_mz_tolerance=0.02)
                    sim = result.score
                    mps = result.matches
                    pp = mps / peaks1
                except:
                    pass

            else:
                try:
                    sim = similarity(spec1, spec2, method=args.self_clustering_method,ms2_da=0.02)
                    result = modified_cosine(spectrum1, spectrum2, fragment_mz_tolerance=0.02)
                    mps = result.matches
                    pp = mps / peaks1
                    # mps = len(
                    #     spectrum_alignment.find_match_peaks_efficient(
                    #         spectrum_alignment.convert_to_peaks(spec1)
                    #         , spectrum_alignment.convert_to_peaks(spec2)
                    #         , shift=shift
                    #         , tolerance=0.02)
                    # )
                    pp = mps / peaks1
                except:
                    pass
            if sim >= args.self_clustering_similarity \
                    and mps >= args.self_clustering_peaks:
                edge_attr = {'pair_similarity': sim, 'matched_peaks': mps, 'peak_percentage': pp,'edge_type': args.self_clustering_method}
                G.add_edge(id1, id2, **edge_attr)
    G = mn_curating(G, args.top_k)
    print('Self clustering finished!')

    # '''Class C1/C2 : NP ms2 match result '''
    npms1_result_path = os.path.join(parent_folder, f'npMS1match_{os.path.basename(args.quant_file)}')
    npms1_match_df = functions.df_preprocess(npms1_result_path)

    npms1_match_df['pair_similarity'] = np.nan
    npms1_match_df['mps'] = np.nan
    npms1_match_df['pp'] = np.nan
    for i in range(len(npms1_match_df)):
        max_values0 = npms1_match_df.loc[i, 'pair_similarity0']
        max_values1 = npms1_match_df.loc[i, 'pair_similarity1']
        max_values2 = npms1_match_df.loc[i, 'pair_similarity2']
        max_mps0 = npms1_match_df.loc[i, 'mps0']
        max_mps1 = npms1_match_df.loc[i, 'mps1']
        max_mps2 = npms1_match_df.loc[i, 'mps2']
        max_pp0 = npms1_match_df.loc[i, 'pp0']
        max_pp1 = npms1_match_df.loc[i, 'pp1']
        max_pp2 = npms1_match_df.loc[i, 'pp2']
        npms1_match_df.loc[i, 'pair_similarity'] = max(max_values0, max_values1, max_values2)  # 根据最大值筛选
        npms1_match_df.loc[i, 'mps'] = max(max_mps0, max_mps1, max_mps2)
        npms1_match_df.loc[i, 'pp'] = max(max_pp0, max_pp1, max_pp2)

    npms1_match_df['pair_similarity'] = pd.to_numeric(npms1_match_df['pair_similarity'], errors='coerce')  # 将缺失值转换成NA
    npms1_match_df['pp'] = pd.to_numeric(npms1_match_df['pp'], errors='coerce')
    # npms1_match_df.to_csv('np_test.csv')
    index_match, index_pp_match = [], []
    index_unmatch = []

    for j in row_ids:  # traverse npMS1match_result by id
        temp_df = npms1_match_df[npms1_match_df['row ID'] == j]
        sim_idx = temp_df['pair_similarity'].idxmax()  # get index of maximum pair_similarity
        pp_idx = temp_df['pp'].idxmax()  # get index of maximum peak_percentage
        if not pd.isna(pp_idx):
            index_pp_match.append(pp_idx)
        if not pd.isna(sim_idx):
            index_match.append(sim_idx)
        elif pd.isna(temp_df['match_id']).all():  # get index of unmatched MS1
            index_unmatch.extend(temp_df.index)
    # print(len(row_ids))
    # print(f'np : {len(index_pp_match)}\n{len(index_match)}\n{len(index_unmatch)}')

    '''C2, similarity annotation'''
    df_new_match = npms1_match_df.loc[index_match].reset_index(drop=True)
    df_new_match_well = df_new_match[(df_new_match['pair_similarity'] >= args.is_library_matching_similarity) & (
            df_new_match['mps'] >= args.is_library_matching_peaks)].reset_index(drop=True)
    for i in range(len(df_new_match_well)):
        pair_sim = df_new_match_well.loc[i, 'pair_similarity']
        matched_peaks = int(df_new_match_well.loc[i, 'mps'])
        peak_percentage = df_new_match_well.loc[i, 'pp']
        spec1_id = str(df_new_match_well.loc[i, 'row ID'])
        spec2_id = str(df_new_match_well.loc[i, 'match_id'])
        edge_attr = {'pair_similarity': pair_sim, 'matched_peaks': matched_peaks, 'peak_percentage': peak_percentage,
                     'edge_type': 'similarity'}
        G.add_edge(spec1_id, spec2_id, **edge_attr)
        G.nodes[spec1_id]['level'] = 'C2'
        G.nodes[spec2_id]['class'] = 'IS'
        G.nodes[spec2_id]['level'] = 'DB'
        G.nodes[spec2_id]['smile'] = df_new_match_well.loc[i, 'match_smiles']

    # df_new_match_well['level']='C2'
    # df_new_match_well.to_csv('C2.csv')

    '''C1'''
    df_new_match_unwell = df_new_match[(df_new_match['pair_similarity'] < args.is_library_matching_similarity) | (
            df_new_match['mps'] < args.is_library_matching_peaks)].reset_index(
        drop=True)
    for i in range(len(df_new_match_unwell)):
        spec1_id = str(df_new_match_unwell.loc[i, 'row ID'])
        G.nodes[spec1_id]['level'] = 'C1'

    '''C2, peak percentage annotation'''
    df_new_pp_match = npms1_match_df.loc[index_pp_match].reset_index(drop=True)
    df_new_pp_match_well = df_new_pp_match[(df_new_pp_match['pp'] >= args.peak_percentage_threshold) & (
            df_new_pp_match['mps'] >= args.is_library_matching_peaks)].reset_index(
        drop=True)
    for i in range(len(df_new_pp_match_well)):
        pair_sim = df_new_pp_match_well.loc[i, 'pair_similarity']
        matched_peaks = int(df_new_pp_match_well.loc[i, 'mps'])
        peak_percentage = df_new_pp_match_well.loc[i, 'pp']
        spec1_id = str(df_new_pp_match_well.loc[i, 'row ID'])
        spec2_id = str(df_new_pp_match_well.loc[i, 'match_id'])
        edge_attr = {'pair_similarity': pair_sim, 'matched_peaks': matched_peaks, 'peak_percentage': peak_percentage,
                     'edge_type': 'peak_percentage'}
        G.add_edge(spec1_id, spec2_id, **edge_attr)
        G.nodes[spec1_id]['level'] = 'C2'
        G.nodes[spec2_id]['class'] = 'IS'
        G.nodes[spec2_id]['level'] = 'DB'
        G.nodes[spec2_id]['smile'] = df_new_pp_match_well.loc[i, 'match_smiles']
    # df_new_pp_match_well['level']='C2'
    # df_new_pp_match_well.to_csv('C2.csv')

    '''Class B1/B2 : GNPS_shared_code ms2 match result '''
    edbms1_result_path = os.path.join(parent_folder, f'edbMS1match_{os.path.basename(args.quant_file)}')
    edbms1_match_df = functions.df_preprocess(edbms1_result_path)

    edb_index_match, edb_pp_index_match = [], []
    edb_index_unmatch = []
    edbms1_match_df['pair_similarity'] = pd.to_numeric(edbms1_match_df['pair_similarity'], errors='coerce')
    edbms1_match_df['pp'] = pd.to_numeric(edbms1_match_df['pp'], errors='coerce')
    for j in row_ids:
        temp_df = edbms1_match_df[edbms1_match_df['row ID'] == j]
        idx = temp_df['pair_similarity'].idxmax()  # 获取pair_similarity最大值的index
        pp_idx = temp_df['pp'].idxmax()
        if not pd.isna(pp_idx):
            edb_pp_index_match.append(pp_idx)
        if not pd.isna(idx):
            edb_index_match.append(idx)
        elif pd.isna(temp_df['match_id']).any():  # 获取MS1无匹配row ID的index，唯一
            edb_index_unmatch.extend(temp_df.index.values.tolist())

    '''B2, similarity annotation'''
    edb_df_new_match = edbms1_match_df.loc[edb_index_match].reset_index(drop=True)
    edb_quant_df_new_match_well = edb_df_new_match[
        (edb_df_new_match['pair_similarity'] >= args.library_matching_similarity) & (
                edb_df_new_match['mps'] >= args.library_matching_peaks)].reset_index(drop=True)
    for i in range(len(edb_quant_df_new_match_well)):
        pair_sim = edb_quant_df_new_match_well.loc[i, 'pair_similarity']
        matched_peaks = int(edb_quant_df_new_match_well.loc[i, 'mps'])
        peak_percentage = edb_quant_df_new_match_well.loc[i, 'pp']
        spec1_id = str(edb_quant_df_new_match_well.loc[i, 'row ID'])
        spec2_id = str(edb_quant_df_new_match_well.loc[i, 'match_id'])
        edge_attr = {'pair_similarity': pair_sim, 'matched_peaks': matched_peaks, 'peak_percentage': peak_percentage,
                     'edge_type': 'similarity'}
        G.add_edge(spec1_id, spec2_id, **edge_attr)
        G.nodes[spec1_id]['level'] = 'B2'
        G.nodes[spec2_id]['class'] = 'EDB'
        G.nodes[spec2_id]['level'] = 'DB'
        G.nodes[spec2_id]['smile'] = edb_quant_df_new_match_well.loc[i, 'match_smiles']
    # edb_quant_df_new_match_well['level']='B2'
    # edb_quant_df_new_match_well.to_csv('B2.csv')

    '''B1'''
    edb_quant_df_new_match_unwell = edb_df_new_match[
        (edb_df_new_match['pair_similarity'] < args.library_matching_similarity) | (
                edb_df_new_match['mps'] < args.library_matching_peaks)].reset_index(drop=True)
    for i in range(len(edb_quant_df_new_match_unwell)):
        spec1_id = str(edb_quant_df_new_match_unwell.loc[i, 'row ID'])
        G.nodes[spec1_id]['level'] = 'B1'

    '''B2, peak percentage annotation'''
    edb_df_new_pp_match = edbms1_match_df.loc[edb_pp_index_match].reset_index(drop=True)
    edb_quant_df_new_pp_match_well = edb_df_new_pp_match[
        (edb_df_new_pp_match['pp'] >= args.peak_percentage_threshold) & (
                edb_df_new_pp_match['mps'] >= args.library_matching_peaks)].reset_index(drop=True)
    for i in range(len(edb_quant_df_new_pp_match_well)):
        pair_sim = edb_quant_df_new_pp_match_well.loc[i, 'pair_similarity']
        matched_peaks = int(edb_quant_df_new_pp_match_well.loc[i, 'mps'])
        peak_percentage = edb_quant_df_new_pp_match_well.loc[i, 'pp']
        spec1_id = str(edb_quant_df_new_pp_match_well.loc[i, 'row ID'])
        spec2_id = str(edb_quant_df_new_pp_match_well.loc[i, 'match_id'])
        edge_attr = {'pair_similarity': pair_sim, 'matched_peaks': matched_peaks, 'peak_percentage': peak_percentage,
                     'edge_type': 'peak_percentage'}
        G.add_edge(spec1_id, spec2_id, **edge_attr)
        G.nodes[spec1_id]['level'] = 'B2'
        G.nodes[spec2_id]['class'] = 'EDB'
        G.nodes[spec2_id]['level'] = 'DB'
        G.nodes[spec2_id]['smile'] = edb_quant_df_new_pp_match_well.loc[i, 'match_smiles']

    '''Class A : MS1 no match'''
    ms1_match_file = os.path.join(parent_folder, f'MS1match_{os.path.basename(args.quant_file)}')  # MS1 result file
    ms1_match_df = functions.df_preprocess(ms1_match_file)

    temp_df = ms1_match_df.loc[:, ['row ID', 'npms1_id', 'edbms1_id']]
    empty_rows = temp_df[temp_df.loc[:, ['npms1_id', 'edbms1_id']].isnull().all(axis=1)].reset_index(drop=True)
    for i in range(len(empty_rows)):
        try:  # 为了处理quant.csv 和 mgf的feature id 不统一的情况
            spec1_id = str(empty_rows.loc[i, 'row ID'])
            G.nodes[spec1_id]['level'] = 'A'
        except:
            pass

    MN_file = os.path.join(parent_folder,
                           f'{os.path.splitext(os.path.basename(args.mgf_file))[0]}_{args.self_clustering_method}_{args.self_clustering_similarity}_{args.self_clustering_peaks}.graphml')
    nx.write_graphml(G, MN_file)
    print('Molecular networking annotation finished!')

if __name__ == '__main__':
    t = time.time()
    args = arg_parse()
    args.isdb_file = '../msdb/isdb_info.json'
    args.edbms2_file = '../msdb/edb_info.json'
    args.quant_file = '/Users/hehe/Desktop/KutzOsmac3_quant.csv'
    args.mgf_file = '/Users/hehe/Desktop/KutzOsmac3.mgf'
    args.output = '/Users/hehe/Desktop/'
    args.library_matching_method = 'modified_cosine'
    args.self_clustering_method = 'symmetric_chi_squared'
    args.self_clustering_similarity = 0.7
    args.self_clustering_peaks = 3
    args.is_library_matching_similarity = 0.7
    args.is_library_matching_peaks = 5
    args.library_matching_similarity = 0.7
    args.library_matching_peaks = 5
    args.peak_percentage_threshold = 0.7
    args.top_k = 10

    molecular_generation(args)
    # spectral_entropy_calculating(args)

    print(f'Finish in {(time.time() - t) / 60:.2f}min')
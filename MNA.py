# -*- coding: utf-8 -*-
# @Time :2022/12/11 19:35
# @Auther :Yuwenchao
# @Software : PyCharm
'''
create_result_folder
create_subfolders
match_mz
ms1_match
match_edb_mz
ms1_match_edb
'''

import sys
sys.path.append('./my_packages')

import os
import pandas as pd
from tqdm import tqdm,trange
from my_packages import functions
from my_packages.ms2tools import spectral_entropy_calculating,ms1_match,ISDB_MS2_match,EDB_MS2_match,molecular_generation
import argparse
import time


def create_result_folders(args):
    '''

    :param quant_file:
    :param output_path:
    :return:
    '''
    df = pd.read_csv(args.quant_file)
    parent_folder = f'{args.output}/{os.path.splitext(os.path.basename(args.quant_file))[0]}_result'# 结果文件名output/_quant_result/**
    os.makedirs(parent_folder, exist_ok=True)
    for _, row in df.iterrows():
        folder_name = f"{parent_folder}/{int(row['row ID'])}"
        os.makedirs(folder_name, exist_ok=True)
    print('Result folders have been created!')

def create_subresults(args):
    '''
    拆分MS1match后的的按hits展开的结果文件
    根据row ID，创建子csv,将对应的信息写入，便于后续仔细分析：in silico MS2
    :param quant_file:
    :param ms1_match_file:
    :return:
    '''
    parent_folder =  f'{args.output}/{os.path.splitext(os.path.basename(args.quant_file))[0]}_result' # 结果文件名output/_quant_result/**
    npms1_result_path =os.path.join(parent_folder, f'npMS1match_{os.path.basename(args.quant_file)}')
    edbms1_result_path = os.path.join(parent_folder, f'edbMS1match_{os.path.basename(args.quant_file)}')

    quant_df = functions.df_preprocess(args.quant_file)
    npms1_match_df = functions.df_preprocess(npms1_result_path)
    edbms1_match_df = functions.df_preprocess(edbms1_result_path)


    for i in trange(len(quant_df)):
        id = quant_df['row ID'][i]
        folder_name = os.path.join(parent_folder, str(id))

        npcsv_file = os.path.join(folder_name, f'npMS1match_{str(id)}.csv') # np result
        if not os.path.exists(npcsv_file):
            pd.DataFrame(columns=npms1_match_df.columns).to_csv(npcsv_file, index=False)
        selected_rows =npms1_match_df.loc[npms1_match_df['row ID'] == id]# 选择与子文件夹名称匹配的DataFrame行，并将它们写入CSV文件
        with open(npcsv_file, 'a', newline='') as f1:
            selected_rows.to_csv(f1, index=False, header=False)

        edbcsv_file = os.path.join(folder_name, f'edbMS1match_{str(id)}.csv') # edb result
        if not os.path.exists(edbcsv_file):
            pd.DataFrame(columns=edbms1_match_df.columns).to_csv(edbcsv_file, index=False)
        selected_rows = edbms1_match_df.loc[edbms1_match_df['row ID'] == id]
        with open(edbcsv_file, 'a', newline='') as f2:
            selected_rows.to_csv(f2, index=False, header=False)

def main(args):
    '''1.创建结果文件夹'''
    create_result_folders(args)
    '''2.1Spectral entropy calculating'''
    spectral_entropy_calculating(args)
    '''3.MS1 match(coconut,npatls,cmnpd)'''
    ms1_match(args)
    ''' 4.MS2 match'''
    ISDB_MS2_match(args)
    EDB_MS2_match(args)
    '''
    5.将npMS1match和edbMS1match的结果写入对应row ID的子文件夹中
    '''
    create_subresults(args)
    '''
    6.分子网络生成
    '''
    molecular_generation(args)



if __name__ == '__main__':
    t = time.time()
    '''数据库文件'''

    '''分析需提供的输入文件'''


    parser = argparse.ArgumentParser(prog='MNA'
                                     , description='Molecular networking annotation(MNA)'
                                     ,usage='python MNA.py main -q xxx_quant.csv -m xxx.mgf -o output_path'
                                     )
    subparsers = parser.add_subparsers(help='sub-command help')  # 添加子命令

    '''subcommand : main'''
    parser_main = subparsers.add_parser('main', help='Default analysis workflow of MNA')
    parser_main.add_argument('-q', '--quant_file'
                             ,help='Quantitative table exported by MZmine'
                             ,default='./example/example_quant.csv'
                             )
    parser_main.add_argument('-m', '--mgf_file'
                             , help='Mgf file exported by MZmine'
                             , default='./example/example.mgf'
                             )
    parser_main.add_argument('-o', '--output'
                             , help='Output path'
                             , default='./example/'
                             )
    parser_main.add_argument('-n1f', '--npms1_file'
                             , help='47w np ms1 file'
                             , default='./msdb/isdbMS1.csv'
                             )
    parser_main.add_argument('-g1f', '--edbms1_file'
                             , help='58w experimental ms1 file'
                             , default='./msdb/edbMS1.csv'
                             )
    parser_main.add_argument('-n2f'
                             , '--isdb_file'
                             , help='in silico natural product library'
                             , default='./msdb/isdb_info.json'
                             )
    parser_main.add_argument('-g2f'
                             , '--edbms2_file'
                             , help='edb ms2 library'
                             , default='./msdb/edb_info.json'
                             )

    parser_main.add_argument('-pmt'
                             , '--pepmass_match_tolerance'
                             , help = 'Allowed ppm tolerance in MS1 matching'
                             , type = int
                             , default = 5
                             )
    parser_main.add_argument('-lmm'
                             , '--library_matching_method'
                             ,help='Similarity algorithm of tandem mass matching used for library search'
                             ,default='modified_cosine_similarity'
                             )
    parser_main.add_argument('-scm'
                             , '--self_clustering_method'
                             , help='Tandem mass self clustering methods'
                             ,default='modified_cosine'
                             )
    parser_main.add_argument('-scs'
                             , '--self_clustering_similarity'
                             , help='Self clustering similarity threshold'
                             , type=float
                             ,default=0.7
                             )
    parser_main.add_argument('-scp'
                             , '--self_clustering_peaks'
                             , help='Self clustering shared peaks threshold'
                             , type=int
                             , default=6
                             )
    parser_main.add_argument('-topk'
                        ,'--top_k'
                        , help='Maximum degree of a node'
                        , type=int
                        , default=10
                        )
    parser_main.add_argument('-islms'
                             , '--is_library_matching_similarity'
                             , help='In silico library matching similarity threshold'
                             , type=float
                             , default=0.7
                             )
    parser_main.add_argument('-islmp'
                             , '--is_library_matching_peaks'
                             , help='In silico library matching shared peaks threshold'
                             , type=int
                             , default=6
                             )
    parser_main.add_argument('-lms'
                             , '--library_matching_similarity'
                             , help='Library matching similarity threshold'
                             ,type=float
                             ,default=0.7
                             )
    parser_main.add_argument('-lmp'
                             , '--library_matching_peaks'
                             , help='Library matching shared peaks threshold'
                             , type=int
                             ,default=6
                             )
    parser_main.add_argument('-ppt'
                           , '--peak_percentage_threshold'
                           , help='Library matching shared peaks perventage threshold'
                           , type=float
                           , default=0.7
                           )
    parser_main.set_defaults(func=main)

    '''subcommand : mn'''
    parser_mn = subparsers.add_parser('mn', help='Re-analysis of results of default MNA workflow')
    parser_mn.add_argument('-q', '--quant_file'
                             , help='Quantitative table exported by MZmine'
                             ,required=True
                             , default='./example/example_quant.csv'
                             )
    parser_mn.add_argument('-m', '--mgf_file'
                             , help='Mgf file exported by MZmine'
                             , required=True
                             , default='./example/example.mgf'
                             )
    parser_mn.add_argument('-o', '--output'
                             , help='Output path'
                             , required=True
                             , default='./example/'
                             )
    parser_mn.add_argument('-pmt'
                             , '--pepmass_match_tolerance'
                             , help='Allowed ppm tolerance in MS1 matching'
                             , type=int
                             ,default=5
                             )
    parser_mn.add_argument('-lmm'
                             , '--library_matching_method'
                             , help='Similarity algorithm of tandem mass matching used for library search'
                             , default='weighted_dot_product'
                             )
    parser_mn.add_argument('-scm'
                             , '--self_clustering_method'
                             , help='Tandem mass self clustering methods'
                             , default='weighted_dot_product'
                             )
    parser_mn.add_argument('-scs'
                             , '--self_clustering_similarity'
                             , help='Self clustering similarity threshold'
                             , type=float
                             , default=0.7
                             )
    parser_mn.add_argument('-scp'
                             , '--self_clustering_peaks'
                             , help='Self clustering shared peaks threshold'
                             , type=int
                             , default=6
                             )
    parser_mn.add_argument('-topk'
                        ,'--top_k'
                        , help='Maximum degree of a node'
                        , type=int
                        , default=10
                        )
    parser_mn.add_argument('-islms'
                             , '--is_library_matching_similarity'
                             , help='In silico library matching similarity threshold'
                             , type=float
                             , default=0.7
                             )
    parser_mn.add_argument('-islmp'
                             , '--is_library_matching_peaks'
                             , help='In silico library matching shared peaks threshold'
                             , type=int
                             , default=6
                             )
    parser_mn.add_argument('-lms'
                             , '--library_matching_similarity'
                             , help='Library matching similarity threshold'
                             , type=float
                             , default=0.7
                             )
    parser_mn.add_argument('-lmp'
                             , '--library_matching_peaks'
                             , help='Library matching shared peaks threshold'
                             , type=int
                             , default=6
                             )
    parser_mn.add_argument('-ppt'
                           , '--peak_percentage_threshold'
                           , help='Library matching shared peaks perventage threshold'
                           , type=float
                           , default=0.7
                           )
    parser_mn.set_defaults(func=molecular_generation)
    # 解析命令行参数并执行相应的子命令函数
    args = parser.parse_args()
    args.func(args)
    print(f'Finished in {(time.time() - t)/60:.2f}min')
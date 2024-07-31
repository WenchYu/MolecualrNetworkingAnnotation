# -*- coding: utf-8 -*-
# @Time :2023/3/12 00:00
# @Auther :Yuwenchao
# @Software : PyCharm
'''

'''
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(usage = 'python MNA.py -q example_quant.csv -m example.mgf -o output directory'
                                     ,description='Molecular Networking Annotation')

    '''input/output'''
    parser.add_argument('-q', '--quant_file'
                             , help='Quantitative table exported by MZmine'
                             , default='./example/example_quant.csv'
                             )
    parser.add_argument('-m', '--mgf_file'
                             , help='Mgf file exported by MZmine'
                             , default='./example/example.mgf'
                             )
    parser.add_argument('-o', '--output'
                             , help='Output path'
                             , default='./example/'
                             )

    parser.add_argument('-n1f', '--npms1_file'
                             , help='47w np ms1 file'
                             , default='./msdb/npMS1.csv'
                             )
    parser.add_argument('-g1f', '--gnpsms1_file'
                             , help='58w gnps ms1 file'
                             , default='./msdb/edbMS1.csv'
                             )
    parser.add_argument('-n2f'
                             , '--isdb_file'
                             , help='Output path'
                             , default='./msdb/isdb_info.json'
                             )
    parser.add_argument('-g2f'
                             , '--gnpsms2_file'
                             , help='Output path'
                             , default='./msdb/edb_info.json'
                             )
    parser.add_argument('-sc'
                             , '--spectrum_clean'
                             , help='If you pass this parameter, spectrum will be cleaned by default'
                             , action='store_true'
                             )
    parser.add_argument('-pmt'
                             , '--pepmass_match_tolerance'
                             , help='Allowed ppm tolerance in MS1 matching'
                             , type=int,
                             default=5
                             )
    parser.add_argument('-lmm'
                             , '--library_matching_method'
                             , help='Similarity algorithm of tandem mass matching used for library search'
                             , default='weighted_dot_product'
                             )
    parser.add_argument('-scm'
                             , '--self_clustering_method'
                             , help='Tandem mass self clustering methods'
                             , default='weighted_dot_product'
                             )
    parser.add_argument('-scs'
                             , '--self_clustering_similarity'
                             , help='Self clustering similarity threshold'
                             , type=float
                             , default=0.7
                             )
    parser.add_argument('-scp'
                             , '--self_clustering_peaks'
                             , help='Self clustering shared peaks threshold'
                             , type=int
                             , default=3
                             )
    parser.add_argument('-topk'
                        '--top_k'
                        , help='Maximum degree of a node'
                        , type=int
                        , default=10
                        )

    parser.add_argument('-lms'
                             , '--library_matching_similarity'
                             , help='Library matching similarity threshold'
                             , type=float
                             , default=0.7
                             )
    parser.add_argument('-lmp'
                             , '--library_matching_peaks'
                             , help='Library matching shared peaks threshold'
                             , type=int
                             , default=3
                             )
    parser.add_argument('-ppt'
                           , '--peak_percentage_threshold'
                           , help='Library matching shared peaks perventage threshold'
                           , type=float
                           , default=0.5
                           )

    return parser.parse_args()

if __name__ == '__main__':
    args= arg_parse()

    print()
# -*- coding: utf-8 -*-
# @Time :2023/2/16 14:03
# @Auther :Yuwenchao
# @Software : PyCharm
'''

'''
import os
import json
import time
from tqdm import trange
import pandas as pd
import spectral_entropy
import numpy as np
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
from my_packages.similarity import modified_cosine,cosine,neutral_loss
from my_packages.functions import mgf_process, get_mgf_info, get_gnps_info, get_isdb_info
from my_packages.spectrum_alignment import find_match_peaks, find_match_peaks_efficient, convert_to_peaks


def ms2_visualization(id, pepmass, spec1 ,activate_clean_spectrum=False):
    spec1 = np.array(spec1, dtype=float)
    if activate_clean_spectrum:
        spec1_after_clean = spectral_entropy.clean_spectrum( spec1
                                                            , max_mz = pepmass
                                                            , noise_removal = 0.01
                                                            , ms2_ppm = 5
                                                            )
    else:
        spec1_after_clean = spec1
    # np.set_printoptions(suppress=True)  # 设置打印选项，取消科学计数法
    # print(f'spec1_after_clean : {spec1_after_clean}')
    spectrum = sus.MsmsSpectrum( identifier= id
                                , precursor_mz = pepmass
                                , precursor_charge = 1
                                , mz = spec1_after_clean[:, 0]
                                , intensity = spec1_after_clean[:, 1]
                                )
    fig, ax = plt.subplots(figsize=(12, 6))
    sup.spectrum(spectrum, ax=ax, grid=False)
    # ax.set_frame_on(False)  # 是否保留外框线
    fig.suptitle(f'Top : {id}\n Precusor mass = {pepmass}\n')
    folder_name = 'Tandem mass'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(f'{folder_name}/{id}_{pepmass}.pdf')

def mirror_plotting(id1,pepmass1,spec1
                    ,id2,pepmass2,spec2
                    ,sim_method = 'symmetric_chi_squared'
                    ,activate_clean_spectrum=True):
    if activate_clean_spectrum:
        spec1 = spectral_entropy.clean_spectrum(spec1
                                                            , max_mz=pepmass1 + 0.1
                                                            , noise_removal=0.01
                                                            , ms2_ppm=5
                                                            , ms2_da=0.02
                                                            )
        spec2 = spectral_entropy.clean_spectrum(spec2
                                                            , max_mz=pepmass2 + 0.1
                                                            , noise_removal=0.01
                                                            , ms2_ppm=5
                                                            , ms2_da=0.02
                                                            )
    # else:
    #     spec1_after_clean = spec1
    #     spec2_after_clean = spec2
    # np.set_printoptions(suppress=True)  # 设置打印选项，取消科学计数法
    # print(f'spec1_after_clean : {spec1}')
    # print(f'spec2_after_clean : {spec2}')
    similarity = spectral_entropy.similarity(spec1,spec2,method=sim_method, ms2_ppm=5)
    shift = abs(pepmass2-pepmass1)
    shared_peaks = len(
                    find_match_peaks_efficient(
                        convert_to_peaks(spec1),
                        convert_to_peaks(spec2),
                        shift, 0.02))
    spectrum_top = sus.MsmsSpectrum(id1
                                    , pepmass1
                                    , 1
                                    , [item[0] for item in spec1]
                                    , [item[1] for item in spec1]
                                    )
    spectrum_bottom = sus.MsmsSpectrum(id2
                                       , pepmass2
                                       , 1
                                       , [item[0] for item in spec2]
                                       , [item[1] for item in spec2]
                                       )

    fig, ax = plt.subplots(figsize=(12, 8))
    sup.mirror(spectrum_top, spectrum_bottom, spectrum_kws={'grid': False}, ax=ax)
    # sup.spectrum(exp_spectrum, ax=ax)
    ax.set_xlim(0, 500)
    ax.set_frame_on(False)  # 是否保留外框线
    ax.xaxis.set_visible(False)  # hide x-axis
    ax.yaxis.set_visible(False)  # hide y-axis

    fig.suptitle(f'Top : {id1}\nBottom : {id2}\n {sim_method} similarity = {similarity:.2f} \nshared peaks = {shared_peaks}'
                 ,fontsize=20, fontfamily='Arial')
    plt.show()
    folder_name = 'Mirror plotting'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(f'{folder_name}/{id1}_{id2}.pdf',transparent=True)

if __name__=='__main__':
    t = time.time()

    '''import database'''
    # gnps_file = '../msdb/edb_info.json'
    # with open(gnps_file, 'r') as f:
    #     gnps_info = json.load(f)

    is_db = '../msdb/isdb_info.json'
    with open(is_db,'r') as f1:
        isdb_info = json.load(f1)

    os.chdir('/Users/hehe/Desktop')
    '''Data Importing'''
    # experiment = 'KutzOsmac1.mgf'
    # exp_info = mgf_process(experiment) # 这事一个dataframe

    # file = 'isdb.csv'
    # df = pd.read_csv(file)
    # for i in range(len(df)):
    #
    #     exp_id = str(df.loc[i,'node'])
    #     id1 = str(df.loc[i,'lib'])

    exp_id = '22'
    # id1 = '359'

    '''temp_ 帮wqh 算相似度'''
    # with open(experiment, 'r') as f:
    #     content = f.readlines()
    # titles = []
    # for line in content:
    #     if 'TITLE=' in line:
    #         tit= line.replace('TITLE=','')
    #         titles.append(tit.strip()) # 这一步 strip，除去多余的 空格，转义符
    #
    # '''experimental mgf'''
    # with open('NL_output.txt','w') as f1:
    #     for title in titles:
    #         f1.write(title+'\t')
    #     f1.write('\n')
    #
    #     for k in trange(len(titles)):
    #         exp_id = titles[k]
    #
    #         for j in range(k):
    #             id1= titles[j]
    #             temp = exp_info[exp_info['id'] == exp_id]
    #             exp_result = get_mgf_info(exp_info, exp_id)
    #             exp_pm = exp_result['pepmass'] # precusor mass
    #             exp_spec = exp_result['spec'] # tandem mass
    #             exp_charge = exp_result['charge'] # charge
    #             exp_spec = spectral_entropy.clean_spectrum(exp_spec
    #                                                        , max_mz=exp_pm + 0.01
    #                                                        )
    #             exp_mz = np.array(exp_spec[:, 0], dtype=np.float64)
    #             exp_intensty = np.array(exp_spec[:, 1], dtype=np.float64)
    #             exp_spectrum = sus.MsmsSpectrum(identifier=exp_id, precursor_mz=exp_pm, precursor_charge=exp_charge, mz=exp_mz,
    #                                             intensity=exp_intensty)  # .remove_precursor_peak(0.1, "Da")
    #             # print(exp_spectrum.mz)
    #
    #             exp_result1 = get_mgf_info(exp_info, id1)
    #             pm1 = exp_result1['pepmass']
    #             spec1 = exp_result1['spec']
    #             # exp_spectrum1 = exp_result1['spectrum']
    #             spec1 = spectral_entropy.clean_spectrum(spec1
    #                                                     , max_mz=pm1 + 0.1
    #                                                     , noise_removal=0.01
    #                                                     )
    #             exp_mz1 = np.array(spec1[:, 0], dtype=np.float64)
    #             exp_intensty1 = np.array(spec1[:, 1], dtype=np.float64)
    #             spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 + 0.01, precursor_charge=1, mz=exp_mz1,
    #                                          intensity=exp_intensty1)  # .remove_precursor_peak(0.1, "Da")
    #
    #             '''Similarity calculation'''
    #             len_exp_spec = len(exp_spectrum.mz)
    #             sim_method = 'neutral_loss'
    #             similarity = 0.0
    #             mps = 0
    #             pp = 0.0
    #             if sim_method == 'modified_cosine':
    #                 result = modified_cosine(exp_spectrum,spectrum1,fragment_mz_tolerance = 0.05)
    #                 similarity = result.score
    #                 mps = result.matches
    #                 pp = mps/len_exp_spec
    #                 matched_indices = result.matched_indices
    #                 matched_indices_other = result.matched_indices_other
    #                 # for i in range(len(matched_indices)):
    #                 #     print(exp_spectrum.mz[matched_indices[i]],spectrum1.mz[matched_indices_other[i]])
    #
    #             elif sim_method == 'neutral_loss':
    #                 result = neutral_loss(exp_spectrum, spectrum1,fragment_mz_tolerance = 0.05)
    #                 similarity = result.score
    #                 mps = result.matches
    #                 pp = mps / len_exp_spec
    #                 matched_indices = result.matched_indices
    #                 matched_indices_other = result.matched_indices_other
    #                 # for i in range(len(matched_indices)):
    #                 #     print(exp_spectrum.mz[matched_indices[i]], spectrum1.mz[matched_indices_other[i]])
    #             else:
    #                 shift = abs(exp_pm - pm1)
    #                 similarity = spectral_entropy.similarity(exp_spec,spec1,method=sim_method,ms2_da=0.05)
    #                 mps = len(find_match_peaks_efficient(
    #                         convert_to_peaks(exp_spec)
    #                         ,convert_to_peaks(spec1)
    #                         , shift = shift
    #                         , tolerance = 0.02))
    #                 pp = mps / len_exp_spec

                # '''Visualizing single spectrum'''
                # fig, ax = plt.subplots(figsize=(12, 8))
                # sup.spectrum(exp_spectrum, ax=ax, grid=False)
                # ax.set_frame_on(False)


                # '''Mirror Plotting'''
                # sup.mirror(exp_spectrum, spectrum1, spectrum_kws={'grid': False}, ax=ax)
                # ax.set_xlim(0, 500)
                # ax.set_frame_on(False)  # 是否保留外框线
                # ax.xaxis.set_visible(False)  # hide x-axis
                # ax.yaxis.set_visible(False)  # hide y-axis
                # fig.suptitle(
                #     f'Top : {exp_id}\nBottom : {id1}\n {sim_method}_similarity = {similarity:.2f} \nshared peaks = {mps}'
                #     , fontsize=20, fontfamily='Arial')
                # print(f'top: {exp_pm}, bottom: {pm1}')

                # '''Plt show and save'''
                # plt.show()
                # folder_name = 'Mirror plotting'
                # if not os.path.exists(folder_name): # 如果没有创建
                #     os.makedirs(folder_name)
                #     fig.savefig(f'{folder_name}/{os.path.splitext(os.path.basename(experiment))[0]}_{exp_id}_{id1}.pdf')
    #             # plt.savefig(f'{os.path.splitext(os.path.basename(experiment))[0]}_{exp_id}_{id1}.pdf')
    #             # print(f'top: {exp_pm}, bottom: {pm1}')
    #             # print(similarity, mps, pp)
    #
            # f1.write('\n')
    '''temp'''

    '''experimental mgf'''
    # exp_result = get_mgf_info(exp_info, exp_id)
    # exp_pm = exp_result['pepmass'] # precusor mass
    # exp_spec = exp_result['spec'] # tandem mass
    # exp_charge = exp_result['charge'] # charge
    # exp_spec = spectral_entropy.clean_spectrum(exp_spec
    #                                            , max_mz=exp_pm + 0.01
    #                                            )
    # exp_mz = np.array(exp_spec[:, 0], dtype=np.float64)
    # exp_intensty = np.array(exp_spec[:, 1], dtype=np.float64)
    # exp_spectrum = sus.MsmsSpectrum(identifier=exp_id, precursor_mz=exp_pm, precursor_charge=exp_charge, mz=exp_mz,
    #                                 intensity=exp_intensty)  # .remove_precursor_peak(0.1, "Da")
    # print(exp_spectrum.mz)

    # exp_result1 = get_mgf_info(exp_info, id1)
    # pm1 = exp_result1['pepmass']
    # spec1 = exp_result1['spec']
    # # exp_spectrum1 = exp_result1['spectrum']
    # spec1 = spectral_entropy.clean_spectrum(spec1
    #                                         , max_mz=pm1 + 0.1
    #                                         , noise_removal=0.01
    #                                         )
    # exp_mz1 = np.array(spec1[:, 0], dtype=np.float64)
    # exp_intensty1 = np.array(spec1[:, 1], dtype=np.float64)
    # spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 + 0.01, precursor_charge=1, mz=exp_mz1,
    #                              intensity=exp_intensty1)  # .remove_precursor_peak(0.1, "Da")

    id1 = 'CNP0241228'

    '''edb MS2'''
    # gnps_result = get_gnps_info(gnps_info,id1)
    # pm1 = gnps_result['pepmass'] # precusor mass
    # spec1 = gnps_result['spec'] # tandem mass
    # charge1 = gnps_result['charge'] # charge
    # spec1 = spectral_entropy.clean_spectrum(spec1
    #                                            , max_mz=pm1 - 0.01
    #                                            )
    # gnps_mz = np.array(spec1[:, 0], dtype=np.float64)
    # gnps_intensty = np.array(spec1[:, 1], dtype=np.float64)
    # spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 , precursor_charge=charge1, mz=gnps_mz,
    #                                 intensity=gnps_intensty)#.remove_precursor_peak(0.1, "Da")


    '''isdb MS2'''
    is_result = get_isdb_info(isdb_info, id1)
    pm1 = is_result['pepmass']
    spec1 = is_result['e0spec']
    spec2 = is_result['e1spec']
    spec3 = is_result['e2spec']

    with open (f'{id1}.mgf', 'w') as f:
        f.write(f'BEGIN\n')
        f.write(f'id = {id1}\n')
        f.write(f'pepmass = {pm1}\n')
        f.write(f'CE = 10 eV\n')
        for pair in spec1:
            f.write(f'{pair[0]}\t{pair[1]}\n')
        f.write(f'CE = 20 eV\n')
        for pair in spec2:
            f.write(f'{pair[0]}\t{pair[1]}\n')
        f.write(f'CE = 40 eV\n')
        for pair in spec3:
            f.write(f'{pair[0]}\t{pair[1]}\n')
        f.write(f'END\n')

    # spec1 = spectral_entropy.clean_spectrum(spec1
    #                                             , max_mz = pm1 + 0.01
    #                                             , noise_removal = 0.01
    #                                             )
    # is_mz = np.array(spec1[:, 0], dtype=np.float64)
    # is_intensty = np.array(spec1[:, 1], dtype=np.float64)
    # spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 + 0.01, precursor_charge=1, mz=is_mz,
    #                                  intensity=is_intensty)  # .remove_precursor_peak(0.1, "Da")






    '''Similarity calculation'''
    # len_exp_spec = len(exp_spectrum.mz)
    # sim_method = 'modified_cosine'
    # similarity = 0.0
    # mps = 0
    # pp = 0.0
    # if sim_method == 'modified_cosine':
    #     result = modified_cosine(exp_spectrum,spectrum1,fragment_mz_tolerance = 0.05)
    #     similarity = result.score
    #     mps = result.matches
    #     pp = mps/len_exp_spec
    #     matched_indices = result.matched_indices
    #     matched_indices_other = result.matched_indices_other
    #     for i in range(len(matched_indices)):
    #         print(exp_spectrum.mz[matched_indices[i]],spectrum1.mz[matched_indices_other[i]])
    #
    # elif sim_method == 'neutral_loss':
    #     result = neutral_loss(exp_spectrum, spectrum1,fragment_mz_tolerance = 0.05)
    #     similarity = result.score
    #     mps = result.matches
    #     pp = mps / len_exp_spec
    #     matched_indices = result.matched_indices
    #     matched_indices_other = result.matched_indices_other
    #     for i in range(len(matched_indices)):
    #         print(exp_spectrum.mz[matched_indices[i]], spectrum1.mz[matched_indices_other[i]])
    # else:
    #     shift = abs(exp_pm - pm1)
    #     similarity = spectral_entropy.similarity(exp_spec,spec1,method=sim_method,ms2_da=0.05)
    #     mps = len(find_match_peaks_efficient(
    #             convert_to_peaks(exp_spec)
    #             ,convert_to_peaks(spec1)
    #             , shift = shift
    #             , tolerance = 0.05))
    #     pp = mps / len_exp_spec
    #
    #
    # '''Visualizing single spectrum'''
    # fig, ax = plt.subplots(figsize=(12, 8))
    # # sup.spectrum(exp_spectrum, ax=ax, grid=False)
    # # ax.set_frame_on(False)
    #
    # '''Mirror Plotting'''
    # sup.mirror(exp_spectrum, spectrum1, spectrum_kws={'grid': False}, ax=ax)
    # ax.set_xlim(0, 500)
    # ax.set_frame_on(False)  # 是否保留外框线
    # ax.xaxis.set_visible(False)  # hide x-axis
    # ax.yaxis.set_visible(False)  # hide y-axis
    # fig.suptitle(
    #     f'Top : {exp_id}\nBottom : {id1}\n {sim_method}_similarity = {similarity:.2f} \nshared peaks = {mps}'
    #     , fontsize=20, fontfamily='Arial')
    # print(f'top: {exp_pm}, bottom: {pm1}')
    # print(similarity, mps, pp)
    #
    # '''Plt show and save'''
    # plt.show()
    # folder_name = 'Mirror plotting'
    # if not os.path.exists(folder_name): # 如果没有创建
    #     os.makedirs(folder_name)
    # plt.savefig(f'{folder_name}/{os.path.splitext(os.path.basename(experiment))[0]}_{exp_id}_{id1}.pdf')
    # plt.savefig(f'{os.path.splitext(os.path.basename(experiment))[0]}_{exp_id}_{id1}.pdf')
    # print(f'top: {exp_pm}, bottom: {pm1}')
    # print(similarity, mps, pp)
    print(f'Finished in {(time.time()-t)/60:.2f}min')


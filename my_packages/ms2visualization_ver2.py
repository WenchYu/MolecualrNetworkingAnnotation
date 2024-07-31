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


def db_library(edb=True, isdb=True, msdb='../msdb/'):
    '''
    读取edb和isdb的json文件
    :param edb: True则返回读取结果，False则返回None
    :param isdb: True则返回读取结果，False则返回None
    :param msdb: 数据库文件所在 folder 的路径
    :return:
    '''
    gnps_info = None
    isdb_info = None
    if edb:
        gnps_file = msdb + 'edb_info.json'
        with open(gnps_file, 'r') as f:
            gnps_info = json.load(f)
    if isdb:
        is_db = msdb + 'isdb_info.json'
        with open(is_db, 'r') as f1:
            isdb_info = json.load(f1)
    return gnps_info, isdb_info

def ex_result(result):
    '''
    提取 get_mgf_info() 提取后的csv文件
    :param result:
    :return:
    '''
    exp_pm = result['pepmass'] # precusor mass
    exp_spec = result['spec'] # tandem mass
    exp_charge = result['charge'] # charge
    exp_spec = spectral_entropy.clean_spectrum(exp_spec
                                               , max_mz=exp_pm + 0.01
                                               )
    exp_mz = np.array(exp_spec[:, 0], dtype=np.float64)
    exp_intensty = np.array(exp_spec[:, 1], dtype=np.float64)
    exp_spectrum = sus.MsmsSpectrum(identifier=exp_id, precursor_mz=exp_pm, precursor_charge=exp_charge, mz=exp_mz,
                                    intensity=exp_intensty)  # .remove_precursor_peak(0.1, "Da")
    return exp_pm, exp_spec, exp_charge, exp_mz, exp_intensty, exp_spectrum

def ex_edb_result(gnps_result):
    '''
    提取 get_gnps_info 得到的 csv的结果
    :param gnps_result:
    :return:
    '''
    pm1 = gnps_result['pepmass'] # precusor mass
    spec1 = gnps_result['spec'] # tandem mass
    charge1 = gnps_result['charge'] # charge
    spec1 = spectral_entropy.clean_spectrum(spec1
                                               , max_mz=pm1 - 0.01
                                               )
    gnps_mz = np.array(spec1[:, 0], dtype=np.float64)
    gnps_intensty = np.array(spec1[:, 1], dtype=np.float64)
    spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 , precursor_charge=charge1, mz=gnps_mz,
                                    intensity=gnps_intensty)#.remove_precursor_peak(0.1, "Da")

    return pm1, spec1, charge1, gnps_mz, gnps_intensty, spectrum1

def ex_is_result(is_result):
    charge = 1
    pm1 = is_result['pepmass']
    spec1 = is_result['e0spec']
    spec1 = spectral_entropy.clean_spectrum(spec1
                                                , max_mz = pm1 + 0.01
                                                , noise_removal = 0.01
                                                )
    is_mz = np.array(spec1[:, 0], dtype=np.float64)
    is_intensty = np.array(spec1[:, 1], dtype=np.float64)
    spectrum1 = sus.MsmsSpectrum(identifier=id1, precursor_mz=pm1 + 0.01, precursor_charge=1, mz=is_mz,
                                     intensity=is_intensty)  # .remove_precursor_peak(0.1, "Da")

    return pm1, spec1, charge, is_mz, is_intensty, spectrum1

def sim_clac(result1, result2, sim_method='modified_cosine'):
    '''
    mgf -> info -> result
    :param result1:
    :param result2:
    :param sim_method:
    :return:
    '''
    pm1, spec1, charge1, mz1, intensty1, spectrum1 = ex_result(result1)
    pm2, spec2, charge2, mz2, intensty2, spectrum2 = ex_result(result2)
    len_exp_spec = len(spectrum1.mz)
    similarity = 0.0
    mps = 0
    pp = 0.0
    if sim_method == 'modified_cosine':
        result = modified_cosine(spectrum1,spectrum2,fragment_mz_tolerance = 0.05)
        similarity = result.score
        mps = result.matches
        pp = mps/len_exp_spec
        matched_indices = result.matched_indices
        matched_indices_other = result.matched_indices_other
        for i in range(len(matched_indices)):
            print(spectrum1.mz[matched_indices[i]],spectrum2.mz[matched_indices_other[i]])

    elif sim_method == 'neutral_loss':
        result = neutral_loss(spectrum1,spectrum2,fragment_mz_tolerance = 0.05)
        similarity = result.score
        mps = result.matches
        pp = mps / len_exp_spec
        matched_indices = result.matched_indices
        matched_indices_other = result.matched_indices_other
        for i in range(len(matched_indices)):
            print(spectrum1.mz[matched_indices[i]], spectrum2.mz[matched_indices_other[i]])
    else:
        shift = abs(pm1 - pm2)
        similarity = spectral_entropy.similarity(spec1,spec2,method=sim_method,ms2_da=0.05)
        mps = len(find_match_peaks_efficient(
                convert_to_peaks(spec1)
                ,convert_to_peaks(spec2)
                , shift = shift
                , tolerance = 0.05))
        pp = mps / len_exp_spec
    return similarity, mps, pp

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

def visualize_single_spectrum(spectrum):
    """

    :param spectrum:
    :param title:
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sup.spectrum(spectrum, ax=ax, grid=False)
    ax.set_frame_on(True)
    for i in range(0, len(spectrum.mz),3): # 间隔3标注
        ax.text(
            spectrum.mz[i],
            spectrum.intensity[i] + 0.1 * max(spectrum.intensity),  # Adjust 0.05 as needed
            f'{spectrum.mz[i]:.4f}',
            fontsize=8,
            va='bottom',
            ha='center',
            rotation=90
        )

    fig.tight_layout()
    ax.set_title(f'Feature ID={exp_spectrum.identifier}, Precusor={exp_spectrum.precursor_mz}')
    plt.show()
    # plt.savefig(f'{spectrum.identifier}.pdf')

def mirror_plot(result1, result2, sim_method='modifided_cosine'):
    pm1, spec1, charge1, mz1, intensty1, spectrum1 = ex_result(result1)
    pm2, spec2, charge2, mz2, intensty2, spectrum2 = ex_result(result2)
    fig, ax = plt.subplots(figsize=(12, 8))
    sup.mirror(spectrum1, spectrum2, spectrum_kws={'grid': False}, ax=ax)
    ax.set_xlim(0, 500)
    ax.set_frame_on(False)  # 是否保留外框线
    ax.xaxis.set_visible(False)  # hide x-axis
    ax.yaxis.set_visible(False)  # hide y-axis
    sim, mps, pp = sim_clac(exp_result, exp_result1, sim_method=sim_method)
    fig.suptitle(
        f'Top : {exp_id}\nBottom : {id1}\n {sim_method}_similarity = {sim:.2f} \nshared peaks = {mps}'
        , fontsize=20, fontfamily='Arial')
    print(f'top: {exp_pm}, bottom: {pm1}')
    print(sim, mps, pp)
    plt.show()

def calculate_similarity(spec1, spec2, method='modified_cosine'):
    """Calculate similarity between two spectra."""
    len_spec1 = len(spec1.mz)
    if method == 'modified_cosine':
        result = modified_cosine(spec1, spec2, fragment_mz_tolerance=0.05)
    elif method == 'neutral_loss':
        result = neutral_loss(spec1, spec2, fragment_mz_tolerance=0.05)
    else:
        shift = abs(spec1.precursor_mz - spec2.precursor_mz)
        result = spectral_entropy.similarity(spec1, spec2, method=method, ms2_da=0.05)

    similarity = result.score
    matches = result.matches
    shared_peaks = matches / len_spec1

    return similarity, matches, shared_peaks

if __name__=='__main__':
    t = time.time()
    '''import database'''
    # gnps_info, isdb_info = db_library()
    # '''edb MS2'''
    # edb_id = 'CCMSLIB00004696188'
    # edb_result = get_gnps_info(gnps_info,edb_id)
    # edb_pm, edb_spec, edb_charge, edb_mz, edb_intensty, edb_spectrum = ex_edb_result(edb_result)
    # '''isdb MS2'''
    # isdb_id = '1'
    # is_result = get_isdb_info(isdb_info, id1)
    # is_pm, is_spec, is_mz, is_intensty, is_spectrum = ex_is_result(is_result)

    os.chdir('/Users/hehe/Desktop')
    '''Data Importing'''
    experiment = '9917.mgf'
    exp_info = mgf_process(experiment) # 生成一个dataframe

    '''experimental mgf'''
    exp_id = '2'
    id1 = '3'
    exp_result = get_mgf_info(exp_info, exp_id)
    exp_pm, exp_spec, exp_charge,  exp_mz, exp_intensty, exp_spectrum = ex_result(exp_result)
    exp_result1 = get_mgf_info(exp_info, id1)
    exp_pm1, exp_spec1, exp_charge1, exp_mz1, exp_intensty1, exp_spectrum1 = ex_result(exp_result1)

    '''Similarity calculation'''

    # print(sim, mps, pp)

    '''Visualizing single spectrum'''
    visualize_single_spectrum(exp_spectrum)

    '''Mirror Plotting'''

    '''Plt show and save'''

    # folder_name = 'Mirror plotting'
    # if not os.path.exists(folder_name): # 如果没有创建
    #     os.makedirs(folder_name)
    # plt.savefig(f'{folder_name}/{os.path.splitext(os.path.basename(experiment))[0]}_{exp_id}_{id1}.pdf')
    # plt.savefig(f'{os.path.splitext(os.path.basename(experiment))[0]}_{exp_id}_{id1}.pdf')
    # print(f'top: {exp_pm}, bottom: {pm1}')
    # print(similarity, mps, pp)
    print(f'Finished in {(time.time()-t)/60:.2f}min')


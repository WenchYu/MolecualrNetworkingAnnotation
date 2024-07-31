# -*- coding: utf-8 -*-
# @Time :2022/11/29 15:38
# @Auther :Yuwenchao
# @Software : PyCharm
'''
用法：DataFrame.drop(labels=None,axis=0, index=None, columns=None, inplace=False)

参数说明：
labels 就是要删除的行列的名字，用列表给定
axis 默认为0，指删除行，因此删除columns时要指定axis=1；
index 直接指定要删除的行
columns 直接指定要删除的列
inplace=False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；
inplace=True，则会直接在原数据上进行删除操作，删除后无法返回。

因此，删除行列有两种方式：
1）labels=None,axis=0 的组合
2）index或columns直接指定要删除的行或列

处理coconut，smiles————> formula ————> exactmass ————> m_plus_h&m_plus_na
'''
import os
import re
import time
import argparse

import numpy as np
import pandas as pd
import autode as ade

from tqdm import trange
from rdkit.Chem import rdFMCS

def MSC(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1, mol2]
                         , bondCompare=rdFMCS.BondCompare.CompareOrder
                         , atomCompare=rdFMCS.AtomCompare.CompareAny
                         , maximizeBonds = False
                         , ringMatchesRingOnly=False
                         , matchValences=False
                         , timeout=10
                         )

    mcs_num_bonds = mcs.numBonds
    mol1_num_bonds = mol1.GetNumBonds()
    mol2_num_bonds = mol2.GetNumBonds()

    similarity = mcs_num_bonds / ((mol1_num_bonds + mol2_num_bonds) - mcs_num_bonds)
    return similarity

class MyChemInfo():
    @staticmethod
    def AtomicWeight(element: str) -> float:
        """
        根据元素名称返回其原子量，区分大小写的
        # 该原子量数据从https://www2.chemistry.msu.edu/faculty/reusch/OrgPage/mass.htm 手册第95版提取
        """
        if len(element) > 2:  # 元素名称长度不应超过2个字符.
            return None
        return {
            "H": 1.007825,
            "C": 12.0000,
            "N": 14.003074,
            "O": 15.994915,
            "F": 18.000938,
            "He": 4.002602,
            "Li": 6.94,
            "Be": 9.0121831,
            "B": 10.012937,
            "Cl": 34.968853,
            "Br": 78.918337,
            "Ne": 20.1797,
            "Na": 22.98976928,
            "Mg": 24.305,
            "Al": 26.9815385,
            "Si": 27.976927,
            "P": 30.973770,
            "S": 31.972071,
            "I": 126.904473,
            "Ar": 39.948,
            "K": 39.0983,
            "Ca": 40.078,
            "Sc": 44.955908,
            "Ti": 47.867,
            "V": 50.9415,
            "Cr": 51.9961,
            "Mn": 54.938044,
            "Fe": 55.845,
            "Co": 58.933194,
            "Ni": 58.6934,
            "Cu": 63.546,
            "Zn": 65.38,
            "Ga": 69.723,
            "Ge": 72.63,
            "As": 74.921595,
            "Se": 73.922476,
            "Kr": 83.798,
            "Rb": 85.4678,
            "Sr": 87.62,
            "Y": 88.90584,
            "Zr": 91.224,
            "Nb": 92.90637,
            "Mo": 95.95,
            "Ru": 101.07,
            "Rh": 102.9055,
            "Pd": 106.42,
            "Ag": 107.8682,
            "Cd": 112.414,
            "In": 114.818,
            "Sn": 118.71,
            "Sb": 121.76,
            "Te": 127.6,
            "Xe": 131.293,
            "Cs": 132.90545196,
            "Ba": 137.327,
            "La": 138.90547,
            "Ce": 140.116,
            "Pr": 140.90766,
            "Nd": 144.242,
            "Sm": 150.36,
            "Eu": 151.964,
            "Gd": 157.25,
            "Tb": 158.92535,
            "Dy": 162.5,
            "Ho": 164.93033,
            "Er": 167.259,
            "Tm": 168.93422,
            "Yb": 173.054,
            "Lu": 174.9668,
            "Hf": 178.49,
            "Ta": 180.94788,
            "W": 183.84,
            "Re": 186.207,
            "Os": 190.23,
            "Ir": 192.217,
            "Pt": 195.084,
            "Au": 196.966569,
            "Hg": 200.592,
            "Tl": 204.38,
            "Pb": 207.2,
            "Bi": 208.9804,
            "Th": 232.0377,
            "Pa": 231.03588,
            "U": 238.02891,
            "Tc": 0,  # 有些放射性元素的原子量没有提供，以0表示。
            "Pm": 0,
            "Po": 0,
            "At": 0,
            "Rn": 0,
            "Fr": 0,
            "Ra": 0,
            "Ac": 0,
            "Np": 0,
            "Pu": 0,
            "Am": 0,
            "Cm": 0,
            "Bk": 0,
            "Cf": 0,
            "Es": 0,
            "Fm": 0,
            "Md": 0,
            "No": 0,
            "Lr": 0,
            "Rf": 0,
            "Db": 0,
            "Sg": 0,
            "Bh": 0,
            "Hs": 0,
            "Mt": 0,
            "Ds": 0,
            "Rg": 0,
            "Cn": 0,
            "Fl": 0,
            "Lv": 0}.get(element, 0.000)

    @staticmethod
    def MolWt(formula: str) -> float:
        '''

        :param formula: Molecular formular in str format
        :return:
        '''
        regStr = "([A-Z]{1}[a-z]{0,1})([0-9]{0,3})"  # 解析化学式的正则表达式
        MatchList = re.findall(regStr, formula)
        cntMatchList = len(MatchList)
        i = 0
        mW = 0.000
        while i < cntMatchList:
            eleName = MatchList[i][0]
            eleCount = int(MatchList[i][1]) if len(MatchList[i][1]) > 0 else 1
            aw = MyChemInfo.AtomicWeight(eleName)
            if (aw == 0):  # 防止错误表示不能及时识别出来。
                return 0
            mW += aw * eleCount
            i = i + 1
        return mW

    @staticmethod
    def Adduct(adduct:str)-> float:
        '''

        :param adduct:
        :return:
        '''
        regStr = "([A-Z]{1}[a-z]{0,1})([0-9]{0,3})"  # 解析化学式的正则表达式
        MatchList = re.findall(regStr, adduct)
        for i in range(len(MatchList)):
            if MatchList[i][0] != 'M':
                return MyChemInfo.AtomicWeight(MatchList[i][0])

if __name__ == '__main__':
    t=time.time()
    os.chdir('/Users/hehe/desktop/xxx/lxn')
    parser = argparse.ArgumentParser(description='外部传参！')
    parser.add_argument('-f', '--file', type=str, help='添加要处理文件')
    args = parser.parse_args()

    file='compounds.csv'
    df=pd.read_csv(file,index_col=None)
    print(df.columns)

    '''待计算'''
    df['formula']=np.nan
    df['exactmass']=np.nan

    '''charge 1+'''
    # df['m+h']=np.nan
    # df['m+nh4'] = np.nan
    # df['m+na']=np.nan
    # df['m+k']=np.nan
    # df['m+meoh+h']=np.nan
    # df['m+acn+h'] = np.nan
    # df['m+2na-h'] = np.nan
    # df['m+acn+na'] = np.nan
    # df['m+2k-h'] = np.nan
    # df['m+dmso+h'] = np.nan
    # df['m+2acn+h'] = np.nan
    # df['2m+h'] = np.nan
    # df['2m+nh4'] = np.nan
    # df['2m+na'] = np.nan
    # df['2m+k'] = np.nan
    # df['2m+acn+h'] = np.nan
    # df['2m+acn+na'] = np.nan

    '''charge 2+'''
    # df['m+2h'] = np.nan
    # df['m+h+nh4'] = np.nan
    # df['m+h+na'] = np.nan
    # df['m+h+k'] = np.nan
    # df['m+acn+2h'] = np.nan
    # df['m+2na'] = np.nan
    # df['m+2acn+2h'] = np.nan
    # df['m+3acn+2h'] = np.nan
    # df['2m+3h2o+2h'] = np.nan

    '''charge 3+'''
    # df['m+3h'] = np.nan
    # df['m+2h+na'] = np.nan
    # df['m+h+2na'] = np.nan
    # df['m+3na'] = np.nan

    '''charge - '''
    # df['m-3h'] = np.nan
    # df['m-2h'] = np.nan
    # df['m-h'] = np.nan
    # df['m+na-2h'] = np.nan
    # df['m+cl'] = np.nan
    # df['m+k-2h'] = np.nan
    # df['m+fa-h'] = np.nan
    # df['m-hac-h'] = np.nan
    # df['m+br'] = np.nan
    # df['m+tfa-h'] = np.nan
    # df['2m-h'] = np.nan
    # df['2m+fa-h'] = np.nan
    # df['2m+hac-h'] = np.nan
    # df['3m-h'] = np.nan

    for i in trange(len(df.index)):
        try:
            smile = df.smiles[i]
            df.loc[i,'formula'] = ade.Molecule(smiles=smile).formula
            df.loc[i,'exactmass'] = MyChemInfo.MolWt(df.formula[i])
            df.loc[i,'m+h'] = df.loc[i,'exactmass'] + 1.007276
        #     df.loc[i,'m+nh4'] = df.loc[i,'exactmass'] + 18.033823
            df.loc[i,'m+na'] = df.loc[i,'exactmass'] + 22.989218
            # df.loc[i, 'm-h2o+h'] = df.loc[i, 'exactmass'] + 1.007276 - 18.01056
            # df.loc[i,'m+k'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+meoh+h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+acn+h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+2na-h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+acn+na'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+2k-h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+dmso+h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'m+2acn+h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'2m+h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'2m+nh4'] = df.loc[i,'exactmass'] +
            # df.loc[i,'2m+na'] = df.loc[i,'exactmass'] +
            # df.loc[i,'2m+k'] = df.loc[i,'exactmass'] +
            # df.loc[i,'2m+acn+h'] = df.loc[i,'exactmass'] +
            # df.loc[i,'2m+acn+na'] = df.loc[i,'exactmass'] +

            # '''charge 2+'''
            # df.loc[i,'m+2h'] = np.nan
            # df.loc[i,'m+h+nh4'] = np.nan
            # df.loc[i,'m+h+na'] = np.nan
            # df.loc[i,'m+h+k'] = np.nan
            # df.loc[i,'m+acn+2h'] = np.nan
            # df.loc[i,'m+2na'] = np.nan
            # df.loc[i,'m+2acn+2h'] = np.nan
            # df.loc[i,'m+3acn+2h'] = np.nan
            # df.loc[i,'2m+3h2o+2h'] = np.nan
            #
            # '''charge 3+'''
            # df.loc[i,'m+3h'] = np.nan
            # df.loc[i,'m+2h+na'] = np.nan
            # df.loc[i,'m+h+2na'] = np.nan
            # df.loc[i,'m+3na'] = np.nan
            #
            # '''charge - '''
            # df.loc[i,'m-3h'] = np.nan
            # df.loc[i,'m-2h'] = np.nan
            # df.loc[i,'m-h'] = np.nan
            # df.loc[i,'m+na-2h'] = np.nan
            # df.loc[i,'m+cl'] = np.nan
            # df.loc[i,'m+k-2h'] = np.nan
            # df.loc[i,'m+fa-h'] = np.nan
            # df.loc[i,'m-hac-h'] = np.nan
            # df.loc[i,'m+br'] = np.nan
            # df.loc[i,'m+tfa-h'] = np.nan
            # df.loc[i,'2m-h'] = np.nan
            # df.loc[i,'2m+fa-h'] = np.nan
            # df.loc[i,'2m+hac-h'] = np.nan
            # df.loc[i,'3m-h'] = np.nan
        except:
            pass


    df.to_csv(file,index=None)










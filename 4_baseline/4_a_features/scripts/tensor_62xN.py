###################################################################################################################
###################################################################################################################

import pandas as pd
import numpy as np
import torch
import math

###################################################################################################################
###################################################################################################################

# 3SE4 complex files
un_compl = {'3SE4_C_TC44A': '3SE4_B_C', '3SE4_C_MC46A': '3SE4_B_C', '3SE4_C_MC46V': '3SE4_B_C', '3SE4_C_SC47A': '3SE4_B_C', 
         '3SE4_C_HC76A': '3SE4_B_C', '3SE4_C_EC77A': '3SE4_B_C', '3SE4_C_VC80A': '3SE4_B_C', '3SE4_C_IC103A': '3SE4_B_C', 
         '3SE4_C_HC187A': '3SE4_B_C', '3SE4_A_YA70A': '3SE4_B_A', '3SE4_A_LA131A': '3SE4_B_A', '3SE4_A_DA132A': '3SE4_B_A', 
         '3SE4_A_SA135A': '3SE4_B_A', '3SE4_A_NA155T': '3SE4_B_A', '3SE4_A_YA163A': '3SE4_B_A', '3SE4_A_TA181A': '3SE4_B_A', 
         '3SE4_A_FA238A': '3SE4_B_A', '3SE4_A_RA241A': '3SE4_B_A', '3SE4_A_NA242A': '3SE4_B_A', '3SE4_A_NA245A': '3SE4_B_A', 
         '3SE4_A_FA290A': '3SE4_B_A', '3SE4_B_PB28A': '3SE4_B_A', '3SE4_B_PB28L': '3SE4_B_A', '3SE4_B_LB32A': '3SE4_B_A', 
         '3SE4_B_MB148A': '3SE4_B_A', '3SE4_B_KB152A': '3SE4_B_C', '3SE4_B_KB152R': '3SE4_B_C'}
# PDB --> complex
pdb_compl = {'2SGP': ['2SGP_E_I'], '2BNQ': ['2BNQ_ABC_DE'], '1S0W': ['1S0W_A_C'], '1B2S': ['1B2S_A_D'], '1TM7': ['1TM7_E_I'], 
                 '2AJF': ['2AJF_A_E'], '1BP3': ['1BP3_A_B'], '1YY9': ['1YY9_CD_A'], '1MAH': ['1MAH_A_F'], '4JPK': ['4JPK_HL_A'], 
                 '1C4Z': ['1C4Z_ABC_D'], '4E6K': ['4E6K_AB_G'], '2OOB': ['2OOB_A_B'], '3UII': ['3UII_A_P'], '1GC1': ['1GC1_G_C'], 
                 '3QHY': ['3QHY_A_B'], '3B4V': ['3B4V_AB_C'], '5UFQ': ['5UFQ_A_C'], '1TM4': ['1TM4_E_I'], '1EAW': ['1EAW_A_B'], 
                 '1U7F': ['1U7F_B_AC'], '4KRO': ['4KRO_A_B'], '1IAR': ['1IAR_A_B'], '1Y3B': ['1Y3B_E_I'], '4O27': ['4O27_A_B'], 
                 '5CXB': ['5CXB_A_B'], '1YCS': ['1YCS_A_B'], '2DSQ': ['2DSQ_I_G'], '3BTH': ['3BTH_E_I'], '1TM1': ['1TM1_E_I'], 
                 '1FC2': ['1FC2_C_D'], '3SE4': ['3SE4_B_A', '3SE4_B_C'], '4KRP': ['4KRP_A_B'], '3HFM': ['3HFM_HL_Y'], 
                 '2UWE': ['2UWE_ABC_EF'], '2DVW': ['2DVW_A_B'], '2VLQ': ['2VLQ_A_B'], '1NMB': ['1NMB_N_LH'], '1CZ8': ['1CZ8_HL_VW'], 
                 '1AHW': ['1AHW_AB_C'], '3SE9': ['3SE9_HL_G'], '3NGB': ['3NGB_HL_G'], '2I26': ['2I26_N_L'], '1DQJ': ['1DQJ_AB_C'], 
                 '1EMV': ['1EMV_A_B'], '2B0Z': ['2B0Z_A_B'], '3MZG': ['3MZG_A_B'], '3SZK': ['3SZK_AB_C'], '2NU1': ['2NU1_E_I'], 
                 '3N4I': ['3N4I_A_B'], '1BD2': ['1BD2_ABC_DE'], '1LP9': ['1LP9_ABC_EF'], '2J8U': ['2J8U_ABC_EF'], '3Q3J': ['3Q3J_A_B'], 
                 '1JTG': ['1JTG_A_B'], '2C5D': ['2C5D_A_C'], '1FCC': ['1FCC_A_C'], '3H9S': ['3H9S_ABC_DE'], '4FZA': ['4FZA_A_B'], 
                 '3SE3': ['3SE3_B_A'], '1KIR': ['1KIR_AB_C'], '1TM3': ['1TM3_E_I'], '4RS1': ['4RS1_A_B'], '3QIB': ['3QIB_ABP_CD'], 
                 '1H9D': ['1H9D_A_B'], '3LZF': ['3LZF_AB_HL'], '2B0U': ['2B0U_AB_C'], '1CT4': ['1CT4_E_I'], '3SE8': ['3SE8_HL_G'], 
                 '1EFN': ['1EFN_A_B'], '4G0N': ['4G0N_A_B'], '4N8V': ['4N8V_G_ABC'], '3LB6': ['3LB6_A_C'], '2BNR': ['2BNR_ABC_DE'], 
                 '1AK4': ['1AK4_A_D'], '5F4E': ['5F4E_A_B'], '1CSE': ['1CSE_E_I'], '4RA0': ['4RA0_A_C'], '4CVW': ['4CVW_A_C'], 
                 '1Y1K': ['1Y1K_E_I'], '1E96': ['1E96_A_B'], '5E6P': ['5E6P_A_B'], '2OI9': ['2OI9_AQ_BC'], '1XGT': ['1XGT_AB_C'], 
                 '1CT2': ['1CT2_E_I'], '4K71': ['4K71_A_BC'], '2G2W': ['2G2W_A_B'], '4FTV': ['4FTV_ABC_DE'], '2JEL': ['2JEL_LH_P'], 
                 '4X4M': ['4X4M_AB_E'], '2VLR': ['2VLR_ABC_DE'], '1X1X': ['1X1X_A_D'], '5C6T': ['5C6T_HL_A'], '3D5S': ['3D5S_A_C'], 
                 '1M9E': ['1M9E_A_D'], '3C60': ['3C60_CD_AB'], '2GOX': ['2GOX_A_B'], '3BN9': ['3BN9_B_CD'], '5K39': ['5K39_A_B'], 
                 '2G2U': ['2G2U_A_B'], '1F5R': ['1F5R_A_I'], '1JCK': ['1JCK_A_B'], '3M62': ['3M62_A_B'], '3MZW': ['3MZW_A_B'], 
                 '1SGD': ['1SGD_E_I'], '2NU2': ['2NU2_E_I'], '2PCC': ['2PCC_A_B'], '1MLC': ['1MLC_AB_E'], '4PWX': ['4PWX_AB_CD'], 
                 '2KSO': ['2KSO_A_B'], '1HE8': ['1HE8_A_B'], '1GL0': ['1GL0_E_I'], '1B41': ['1B41_A_B'], '3QDJ': ['3QDJ_ABC_DE'], 
                 '3HH2': ['3HH2_AB_C'], '1XGU': ['1XGU_AB_C'], '1TM5': ['1TM5_E_I'], '1VFB': ['1VFB_AB_C'], '1KNE': ['1KNE_A_P'], 
                 '3F1S': ['3F1S_A_B'], '1C1Y': ['1C1Y_A_B'], '1KBH': ['1KBH_A_B'], '2AW2': ['2AW2_A_B'], '1KIQ': ['1KIQ_AB_C'], 
                 '3N0P': ['3N0P_A_B'], '4JFE': ['4JFE_ABC_DE'], '2J1K': ['2J1K_C_T'], '2VLP': ['2VLP_A_B'], '1DAN': ['1DAN_HL_UT'], 
                 '3TGK': ['3TGK_E_I'], '4JGH': ['4JGH_ABC_D'], '3NVQ': ['3NVQ_B_A'], '4G2V': ['4G2V_A_B'], '1CSO': ['1CSO_E_I'], 
                 '1R0R': ['1R0R_E_I'], '3KBH': ['3KBH_A_E'], '4YEB': ['4YEB_A_B'], '4NM8': ['4NM8_ABCDEF_HL'], '2VIS': ['2VIS_AB_C'], 
                 '2FTL': ['2FTL_E_I'], '1P6A': ['1P6A_A_B'], '1QSE': ['1QSE_ABC_DE'], '1P69': ['1P69_A_B'], '3NPS': ['3NPS_A_BC'], 
                 '2JCC': ['2JCC_ABC_EF'], '4NKQ': ['4NKQ_C_AB'], '3S9D': ['3S9D_A_B'], '1E50': ['1E50_A_B'], '2BTF': ['2BTF_A_P'], 
                 '4OFY': ['4OFY_A_D'], '3BT1': ['3BT1_A_U'], '3SGB': ['3SGB_E_I'], '2VLO': ['2VLO_A_B'], '2B42': ['2B42_A_B'], 
                 '2REX': ['2REX_A_B'], '1S1Q': ['1S1Q_A_B'], '1DVF': ['1DVF_AB_CD'], '2J12': ['2J12_A_B'], '3SF4': ['3SF4_A_D'], 
                 '4GXU': ['4GXU_ABCDEF_MN'], '5E9D': ['5E9D_AB_CDE'], '3BK3': ['3BK3_A_C'], '3VR6': ['3VR6_ABCDEF_GH'], 
                 '4Y61': ['4Y61_A_B'], '1GL1': ['1GL1_A_I'], '4LRX': ['4LRX_AB_CD'], '2E7L': ['2E7L_EQ_AD'], '3KUD': ['3KUD_A_B'], 
                 '1SGY': ['1SGY_E_I'], '3SEK': ['3SEK_B_C'], '1K8R': ['1K8R_A_B'], '1Y3C': ['1Y3C_E_I'], '2J0T': ['2J0T_A_D'], 
                 '1KIP': ['1KIP_AB_C'], '3EQS': ['3EQS_A_B'], '4J2L': ['4J2L_A_CD'], '2B10': ['2B10_A_B'], '3EQY': ['3EQY_A_C'], 
                 '1TO1': ['1TO1_E_I'], '1FY8': ['1FY8_E_I'], '1SGN': ['1SGN_E_I'], '3R9A': ['3R9A_AC_B'], '3PWP': ['3PWP_ABC_DE'], 
                 '1F47': ['1F47_A_B'], '3BX1': ['3BX1_A_C'], '4U6H': ['4U6H_AB_E'], '1GUA': ['1GUA_A_B'], '4MNQ': ['4MNQ_ABC_DE'], 
                 '1CBW': ['1CBW_FGH_I'], '3BTT': ['3BTT_E_I'], '4KRL': ['4KRL_A_B'], '4CPA': ['4CPA_A_I'], '4OZG': ['4OZG_ABJ_GH'], 
                 '1CHO': ['1CHO_EFG_I'], '4HSA': ['4HSA_AB_C'], '2BDN': ['2BDN_HL_A'], '3BTQ': ['3BTQ_E_I'], '1BJ1': ['1BJ1_HL_VW'], 
                 '2NYY': ['2NYY_DC_A'], '4B0M': ['4B0M_A_BM'], '1JRH': ['1JRH_LH_I'], '4JFF': ['4JFF_ABC_DE'], '4MYW': ['4MYW_A_B'], 
                 '4JEU': ['4JEU_A_B'], '1Y33': ['1Y33_E_I'], '3QFJ': ['3QFJ_ABC_DE'], '4HRN': ['4HRN_A_D'], '3NCC': ['3NCC_A_B'], 
                 '1AO7': ['1AO7_ABC_DE'], '1SBN': ['1SBN_E_I'], '3G6D': ['3G6D_LH_A'], '1JTD': ['1JTD_A_B'], '2NU4': ['2NU4_E_I'], 
                 '4L3E': ['4L3E_ABC_DE'], '3W2D': ['3W2D_A_HL'], '4BFI': ['4BFI_A_B'], '4EKD': ['4EKD_A_B'], '4YFD': ['4YFD_A_B'], 
                 '1CT0': ['1CT0_E_I'], '1A22': ['1A22_A_B'], '2AK4': ['2AK4_ABC_DE'], '4P5T': ['4P5T_CD_AB'], '2B11': ['2B11_A_B'], 
                 '2WPT': ['2WPT_A_B'], '1WQJ': ['1WQJ_I_B'], '1MI5': ['1MI5_ABC_DE'], '3U82': ['3U82_A_B'], '1B3S': ['1B3S_A_D'], 
                 '2NZ9': ['2NZ9_DC_A'], '4NZW': ['4NZW_A_B'], '3BTD': ['3BTD_E_I'], '1PPF': ['1PPF_E_I'], '2VIR': ['2VIR_AB_C'], 
                 '3N06': ['3N06_A_B'], '1XGR': ['1XGR_AB_C'], '4JFD': ['4JFD_ABC_DE'], '3QDG': ['3QDG_ABC_DE'], '2C0L': ['2C0L_A_B'], 
                 '3EG5': ['3EG5_A_B'], '3BTM': ['3BTM_E_I'], '3BTF': ['3BTF_E_I'], '5UFE': ['5UFE_A_B'], '1XGQ': ['1XGQ_AB_C'], 
                 '1SMF': ['1SMF_E_I'], '4YH7': ['4YH7_A_B'], '5XCO': ['5XCO_A_B'], '1KTZ': ['1KTZ_A_B'], '3RF3': ['3RF3_A_C'], 
                 '3WWN': ['3WWN_A_B'], '1YQV': ['1YQV_HL_Y'], '4P23': ['4P23_CD_AB'], '3LNZ': ['3LNZ_A_B'], '3AAA': ['3AAA_AB_C'], 
                 '2PCB': ['2PCB_A_B'], '3BTW': ['3BTW_E_I'], '2A9K': ['2A9K_A_B'], '1FSS': ['1FSS_A_B'], '4WND': ['4WND_A_B'], 
                 '1Y48': ['1Y48_E_I'], '1BRS': ['1BRS_A_D'], '1XGP': ['1XGP_AB_C'], '1FFW': ['1FFW_A_B'], '1X1W': ['1X1W_A_D'], 
                 '4L0P': ['4L0P_A_B'], '1XD3': ['1XD3_A_B'], '3L5X': ['3L5X_A_HL'], '4I77': ['4I77_HL_Z'], '4GNK': ['4GNK_A_B'], 
                 '4ZS6': ['4ZS6_HL_A'], '2HLE': ['2HLE_A_B'], '2VLN': ['2VLN_A_B'], '1SGP': ['1SGP_E_I'], '3D3V': ['3D3V_ABC_DE'], 
                 '3N85': ['3N85_A_LH'], '1FR2': ['1FR2_A_B'], '1NCA': ['1NCA_N_LH'], '1A4Y': ['1A4Y_A_B'], '1GRN': ['1GRN_A_B'], 
                 '1GCQ': ['1GCQ_AB_C'], '3M63': ['3M63_A_B'], '3Q8D': ['3Q8D_A_E'], '3UIG': ['3UIG_A_P'], '2O3B': ['2O3B_A_B'], 
                 '2GYK': ['2GYK_A_B'], '1REW': ['1REW_AB_C'], '1SBB': ['1SBB_A_B'], '1OGA': ['1OGA_ABC_DE'], '2P5E': ['2P5E_ABC_DE'], 
                 '3NVN': ['3NVN_B_A'], '1LFD': ['1LFD_A_B'], '2B12': ['2B12_A_B'], '3BTE': ['3BTE_E_I'], '2SGQ': ['2SGQ_E_I'], 
                 '3D5R': ['3D5R_A_C'], '3BTG': ['3BTG_E_I'], '2SIC': ['2SIC_E_I'], '5M2O': ['5M2O_A_B'], '1SGQ': ['1SGQ_E_I'], 
                 '5CYK': ['5CYK_A_B'], '1UUZ': ['1UUZ_A_D'], '3BP8': ['3BP8_A_C'], '1TMG': ['1TMG_E_I'], '2HRK': ['2HRK_A_B'], 
                 '1KAC': ['1KAC_A_B'], '1MHP': ['1MHP_HL_A'], '4HFK': ['4HFK_A_BD'], '1Y3D': ['1Y3D_E_I'], '3NCB': ['3NCB_A_B'], 
                 '2B2X': ['2B2X_HL_A'], '1B2U': ['1B2U_A_D'], '1N8O': ['1N8O_ABC_E'], '1Z7X': ['1Z7X_W_X'], '1ACB': ['1ACB_E_I'], 
                 '1Y34': ['1Y34_E_I'], '1N8Z': ['1N8Z_AB_C'], '5TAR': ['5TAR_A_B'], '1SIB': ['1SIB_E_I'], '2NU0': ['2NU0_E_I'], 
                 '1SGE': ['1SGE_E_I']}
                 
###################################################################################################################
###################################################################################################################

def make_df(file, delimiter):
    """Takes file path and returns a pandas DataFrame."""
    try:
        df = pd.read_csv(file, sep=delimiter)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def one_hot_srv(srv):
    '''Takes srv in skempi nomenclature and returns positions of m and wt 
       resiudes in the aa alphabeth'''

    aas =  'ARNDCEQGHILKMFPSTWYV'
    aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
               'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
               'SER', 'THR', 'TRP', 'TYR', 'VAL']
    
    wt = aas.find(srv[0])
    mut = aas.find(srv[-1])

    return wt, mut
    
def composition_res_surr(df):
    ''' Takes a dataframe with data and returns aa composition 
        of surroundings of residue carrying mutation'''
    
    compositions = []
    
    for i in range(df.shape[0]):
        mut = df['PDB_mut'][i]
        pos = mut[2:-1].upper()
        chain = mut[1]
        if df['PDB_ID'][i] != '3SE4':
            complex_file = '../../Binding_Affinity/data/complexes/'+pdb_compl[df['PDB_ID'][i]][0]+'.pdb'
        else:
            complex_file = '../../Binding_Affinity/data/complexes/'+un_compl[df['Unique'][i]]+'.pdb'
        
        monomer_file = f'../../Binding_Affinity/data/pdb/{df['PDB_ID'][i]}.pdb'
        
        found = False
        with open(complex_file, 'r') as reader:
            for line in reader:
                res_num = line[22:26].strip()
                icode = line[26].strip()
                pdb_res = res_num + (icode if icode else "")
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and pdb_res == pos and line[21] == chain:
                    xr = float(line[30:38])
                    yr = float(line[38:46])
                    zr = float(line[46:54])
                    found = True
                    break

        if not found:
            print(f"Residuo {pos} della catena {chain} non trovato in {complex_file.split('/')[-1]}")
            compositions.append([0]*20)
            continue

        with open(complex_file, 'r') as reader:
            counter = 0
            aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
               'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
               'SER', 'THR', 'TRP', 'TYR', 'VAL']
            res_comp = {'ALA':0, 'ARG':0, 'ASN':0, 'ASP':0, 'CYS':0, 
                          'GLU':0, 'GLN':0, 'GLY':0, 'HIS':0, 'ILE':0, 
                          'LEU':0, 'LYS':0, 'MET':0, 'PHE':0, 'PRO':0, 
                          'SER':0, 'THR':0, 'TRP':0, 'TYR':0, 'VAL':0}
            comps = []
            for line in reader:    
                res_num = line[22:26].strip()
                icode = line[26].strip()
                pdb_res = res_num + (icode if icode else "")
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and pdb_res != pos:
                    xa = float(line[30:38])
                    ya = float(line[38:46])
                    za = float(line[46:54])
                    d = ((xa - xr)**2 + (ya - yr)**2 + (za - zr)**2)**0.5
                    if d <= 12:
                        res_name = line.split()[3]
                        if res_name in res_comp:
                            res_comp[res_name] += 1
                            counter += 1
                elif line.startswith('ATOM') and line[12:16].strip() == 'CA' and pdb_res == pos and line[21]==chain:
                    res_name = line[17:20].strip()
                    if res_name in res_comp:
                        res_comp[res_name] += 1
                    else:
                        print('Trova', res_name,  'in', pos, 'della catena', chain, 'del file', complex_file.split('/')[-1])
                    counter += 1
            if counter == 0:
                print(f"Nessun vicino per {complex_file.split('/')[-1]}, posizione {pos}")
                compositions.append([0]*20)
                continue
            for aa in aa_list:
                if counter > 0:
                    comps.append(res_comp[aa]/counter)
                else:
                    print(complex_file)
        compositions.append(comps) 
            
        found = False
        with open(complex_file, 'r') as reader:
            for line in reader:
                res_num = line[22:26].strip()
                icode = line[26].strip()
                pdb_res = res_num + (icode if icode else "")
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and pdb_res == pos and line[21] == chain:
                    xr = float(line[30:38])
                    yr = float(line[38:46])
                    zr = float(line[46:54])
                    found = True
                    break

        if not found:
            print(f"Residuo {pos} della catena {chain} non trovato in {complex_file.split('/')[-1]}")
            compositions.append([0]*20)
            continue

        with open(monomer_file, 'r') as reader:
            counter = 0
            aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
               'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
               'SER', 'THR', 'TRP', 'TYR', 'VAL']
            res_comp = {'ALA':0, 'ARG':0, 'ASN':0, 'ASP':0, 'CYS':0, 
                          'GLU':0, 'GLN':0, 'GLY':0, 'HIS':0, 'ILE':0, 
                          'LEU':0, 'LYS':0, 'MET':0, 'PHE':0, 'PRO':0, 
                          'SER':0, 'THR':0, 'TRP':0, 'TYR':0, 'VAL':0}
            for line in reader:    
                res_num = line[22:26].strip()
                icode = line[26].strip()
                pdb_res = res_num + (icode if icode else "")
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and pdb_res != pos:
                    xa = float(line[30:38])
                    ya = float(line[38:46])
                    za = float(line[46:54])
                    d = ((xa - xr)**2 + (ya - yr)**2 + (za - zr)**2)**0.5
                    if d <= 12:
                        res_name = line.split()[3]
                        if res_name in res_comp:
                            res_comp[res_name] += 1
                            counter += 1
                elif line.startswith('ATOM') and line[12:16].strip() == 'CA' and pdb_res == pos and line[21]==chain:
                    res_name = line[17:20].strip()
                    if res_name in res_comp:
                        res_comp[res_name] += 1
                    else:
                        print('Trova', res_name,  'in', pos, 'della catena', chain, 'del file', complex_file.split('/')[-1])
                    counter += 1
            if counter == 0:
                print(f"Nessun vicino per {complex_file.split('/')[-1]}, posizione {pos}")
                compositions.append([0]*20)
                continue
            for aa in aa_list:
                if counter > 0:
                    comps.append(res_comp[aa]/counter)
                else:
                    print(complex_file)
        compositions.append(comps)  
    return compositions    

###################################################################################################################
###################################################################################################################

# folds 
bts_df = make_df('../data_meta/bts/BTS_data.tsv', '\t')
fold_1_df = make_df('../data_meta/training/fold_1.tsv', '\t')
fold_2_df = make_df('../data_meta/training/fold_2.tsv', '\t')
fold_3_df = make_df('../data_meta/training/fold_3.tsv', '\t')
fold_4_df = make_df('../data_meta/training/fold_4.tsv', '\t')
fold_5_df = make_df('../data_meta/training/fold_5.tsv', '\t')

# rsa
rsa_bts = make_df('../rsa/rsa_out/BTS_rsa.tsv', '\t')
rsa_1 = make_df('../rsa/rsa_out/fold_1_rsa.tsv', '\t')
rsa_2 = make_df('../rsa/rsa_out/fold_2_rsa.tsv', '\t')
rsa_3 = make_df('../rsa/rsa_out/fold_3_rsa.tsv', '\t')
rsa_4 = make_df('../rsa/rsa_out/fold_4_rsa.tsv', '\t')
rsa_5 = make_df('../rsa/rsa_out/fold_5_rsa.tsv', '\t')

folds = [bts_df, fold_1_df, fold_2_df, fold_3_df, fold_4_df, fold_5_df]
rsas = [rsa_bts, rsa_1, rsa_2, rsa_3, rsa_4, rsa_5]
tensors = ['bts', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']


###################################################################################################################
###################################################################################################################

for j in range(6):
    data = folds[j]
    rsa = rsas[j]
    compositions = composition_res_surr(data)
    x = torch.zeros(data.shape[0],62)
    for i in range(data.shape[0]):
        wt, mut = one_hot_srv(data['PDB_mut'][i])
        x[i][wt] = +1
        x[i][mut] = -1
        for k in range(40):
            x[i][20+k] = compositions[i][k]
        x[i][60] = rsa['rsa_complex'][i]
        x[i][61] = rsa['rsa_monomer'][i]
    torch.save(x, 'tensors1/'+tensors[j]+'_X_tensor.pt')
    #print(x)
    #print('Saved tensor ', j+1, '(', tensors[j], ')', '\n')'''
'''
for df in folds:
    x = torch.zeros(df.shape[0],42)
    for i in df.head().shape[0]:
        wt, mut = one_hot_srv(df['PDB_mut'][i])
        x[i][wt] = +1
        x[i][mut] = -1
        print(x[i])
'''
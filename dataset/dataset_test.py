import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels, temps, ln_As, Bs = [], [], [], [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                # smiles = row['smiles']
                smiles = row[target[12]]
                label = [row[target[0]], row[target[1]], row[target[2]], row[target[3]], row[target[4]]]
                label = list(map(float, label))
                label = list(map(lambda x: math.log(x), label))
                temp = [row[target[5]], row[target[6]], row[target[7]], row[target[8]], row[target[9]]]
                ln_A = row[target[10]]
                B = row[target[11]]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(list(map(float, label)))
                        temps.append(list(map(float, temp)))
                        ln_As.append(float(ln_A))
                        Bs.append(float(B))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels, temps, ln_As, Bs


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.temps, self.ln_As, self.Bs = read_smiles(data_path, target, task)
        self.task = task

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
            temp = torch.tensor(self.temps[index], dtype=torch.float).view(1,-1)
            ln_A = torch.tensor(self.ln_As[index], dtype=torch.float).view(1,-1)
            B = torch.tensor(self.Bs[index], dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, temp=temp, ln_A=ln_A, B=B)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, train_size, 
        data_path, target, task, splitting
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.train_size = train_size
        self.target = target
        self.task = task
        self.splitting = splitting
        print(self.valid_size, self.test_size, self.train_size)
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self, task_name):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset, task_name)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset, task_name):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            # print(indices)
            # currently using a set split. Change these based on preference
            if task_name == 'visc':
                # viscosity indices
                indices = [270, 102, 121, 16, 281, 8, 400, 179, 10, 98, 170, 221, 114, 249, 303, 388, 229, 65, 259, 456, 175, 111, 139, 357, 129, 298, 260, 215, 420, 337, 458, 379, 457, 108, 340, 241, 403, 430, 449, 243, 12, 274, 261, 4, 207, 22, 474, 31, 434, 141, 271, 192, 60, 15, 265, 122, 97, 0, 149, 284, 69, 383, 126, 324, 147, 358, 437, 127, 362, 154, 71, 292, 219, 137, 426, 231, 242, 310, 318, 256, 30, 283, 395, 196, 276, 155, 389, 142, 133, 78, 354, 72, 49, 320, 416, 210, 445, 330, 404, 460, 258, 140, 349, 187, 224, 405, 48, 132, 38, 293, 25, 470, 94, 105, 66, 398, 183, 84, 250, 40, 421, 287, 123, 419, 41, 366, 275, 29, 441, 232, 323, 370, 360, 469, 216, 350, 235, 332, 352, 81, 453, 286, 44, 312, 200, 326, 364, 115, 206, 394, 386, 191, 201, 285, 279, 80, 319, 444, 220, 74, 214, 109, 291, 296, 309, 368, 315, 347, 407, 264, 223, 353, 46, 13, 85, 331, 230, 186, 225, 401, 390, 11, 228, 106, 158, 255, 161, 20, 333, 311, 226, 425, 239, 131, 307, 443, 317, 290, 450, 348, 222, 365, 248, 177, 338, 130, 280, 378, 165, 336, 304, 156, 325, 244, 341, 466, 50, 119, 146, 189, 468, 51, 157, 34, 471, 162, 99, 47, 5, 308, 160, 410, 32, 442, 218, 475, 87, 208, 205, 116, 54, 472, 342, 53, 273, 152, 202, 321, 55, 376, 355, 237, 9, 27, 64, 253, 278, 68, 128, 19, 387, 263, 181, 90, 439, 168, 373, 135, 344, 409, 209, 43, 361, 167, 334, 297, 18, 384, 211, 174, 63, 42, 393, 294, 267, 402, 79, 172, 100, 251, 118, 58, 396, 371, 33, 335, 408, 459, 195, 14, 385, 3, 184, 277, 418, 272, 83, 234, 345, 150, 262, 289, 300, 180, 23, 417, 346, 76, 91, 125, 254, 37, 306, 21, 467, 440, 447, 327, 414, 212, 190, 163, 236, 26, 197, 88, 423, 446, 305, 73, 382, 185, 153, 104, 329, 424, 257, 406, 57, 432, 52, 107, 301, 412, 233, 103, 413, 428, 377, 148, 194, 171, 144, 363, 448, 101, 328, 45, 117, 391, 268, 7, 138, 380, 429, 198, 435, 464, 367, 247, 164, 92, 359, 143, 1, 188, 59, 203, 411, 465, 397, 35, 227, 238, 75, 56, 399, 2, 454, 436, 6, 461, 415, 112, 351, 61, 199, 269, 240, 463, 246, 28, 113, 288, 369, 299, 24, 145, 124, 176, 433, 95, 375, 473, 70, 295, 381, 245, 266, 178, 151, 314, 313, 120, 136, 431, 452, 17, 77, 62, 193, 173, 169, 166, 356, 252, 462, 93, 438, 89, 451, 422, 217, 67, 82, 204, 159, 427, 182, 343, 96, 134, 213, 392, 455, 302, 282, 36, 374, 86, 316, 322, 372, 39, 110, 339]
            elif task_name == 'cond':    
                # cond indices
                indices = [154, 996, 1114, 1031, 373, 729, 45, 521, 423, 287, 318, 236, 558, 1096, 836, 570, 412, 1046, 592, 377, 818, 1159, 789, 598, 35, 319, 834, 269, 336, 454, 726, 504, 722, 1128, 320, 1110, 806, 897, 1038, 843, 168, 90, 1092, 1100, 992, 948, 927, 355, 138, 863, 741, 1142, 297, 7, 160, 348, 766, 73, 1082, 442, 1044, 247, 658, 266, 111, 286, 572, 1086, 1177, 213, 210, 1173, 368, 1093, 257, 178, 304, 1048, 259, 1150, 899, 337, 407, 1172, 1072, 356, 452, 381, 495, 126, 898, 280, 721, 176, 367, 913, 338, 398, 877, 975, 606, 268, 613, 574, 580, 200, 997, 122, 291, 1117, 459, 1081, 315, 52, 647, 131, 151, 418, 1204, 589, 409, 30, 728, 466, 1005, 619, 14, 745, 66, 712, 175, 930, 901, 42, 196, 1074, 9, 1220, 475, 937, 795, 59, 453, 508, 892, 511, 705, 156, 220, 157, 1113, 1019, 281, 1065, 119, 468, 594, 225, 65, 187, 637, 826, 623, 498, 579, 939, 1155, 972, 400, 124, 223, 1069, 1024, 405, 366, 907, 496, 1196, 1201, 1095, 378, 876, 292, 851, 23, 627, 272, 322, 77, 879, 350, 793, 4, 462, 506, 980, 191, 911, 624, 903, 861, 617, 1146, 390, 277, 1148, 915, 46, 773, 567, 197, 414, 1144, 955, 500, 153, 533, 478, 239, 542, 147, 1203, 76, 86, 889, 54, 822, 720, 797, 169, 798, 258, 704, 294, 227, 1179, 141, 943, 963, 681, 735, 967, 703, 1045, 775, 219, 1011, 95, 343, 180, 179, 105, 951, 543, 365, 802, 649, 887, 886, 28, 796, 252, 538, 1, 524, 1213, 416, 974, 882, 988, 1047, 714, 841, 1033, 404, 487, 522, 12, 36, 1215, 58, 1125, 825, 1189, 1014, 284, 139, 1137, 285, 3, 1002, 536, 1186, 218, 1139, 110, 62, 49, 610, 708, 207, 472, 778, 785, 1170, 374, 128, 684, 1211, 746, 1124, 1200, 769, 557, 945, 510, 450, 727, 477, 936, 949, 148, 401, 1194, 688, 201, 1192, 864, 464, 837, 1154, 1210, 1003, 96, 645, 181, 1104, 1066, 1145, 691, 1140, 170, 48, 87, 57, 1206, 499, 1034, 526, 747, 217, 883, 174, 249, 670, 920, 369, 758, 1166, 1090, 555, 93, 612, 419, 484, 72, 578, 635, 989, 807, 8, 490, 307, 850, 551, 26, 842, 626, 226, 1073, 31, 1182, 548, 1099, 1135, 1085, 518, 1123, 661, 425, 830, 577, 447, 697, 631, 303, 733, 971, 134, 214, 869, 27, 305, 63, 237, 342, 644, 385, 981, 221, 479, 689, 754, 749, 917, 2, 1036, 288, 940, 1106, 1112, 264, 206, 1130, 376, 891, 756, 620, 363, 641, 819, 402, 112, 931, 654, 634, 1077, 473, 461, 267, 1032, 957, 1165, 656, 290, 817, 422, 881, 293, 799, 212, 391, 857, 198, 755, 1187, 1097, 568, 643, 70, 585, 935, 757, 424, 1167, 75, 683, 438, 942, 301, 786, 92, 858, 777, 330, 67, 1126, 491, 328, 602, 117, 970, 17, 687, 1018, 1162, 1188, 104, 1212, 840, 282, 1098, 695, 906, 202, 360, 809, 171, 482, 334, 278, 941, 968, 870, 445, 489, 717, 1208, 1207, 306, 411, 106, 662, 260, 772, 140, 208, 960, 321, 135, 29, 420, 1111, 372, 588, 1149, 136, 384, 718, 1055, 316, 118, 132, 763, 1161, 604, 1185, 685, 1195, 1151, 1078, 607, 335, 1062, 855, 1025, 597, 1017, 982, 302, 593, 1022, 709, 1217, 1000, 701, 488, 397, 1138, 759, 748, 98, 730, 590, 79, 163, 839, 724, 601, 235, 209, 541, 916, 528, 486, 675, 275, 325, 0, 231, 203, 677, 173, 194, 15, 768, 99, 91, 1119, 289, 673, 599, 710, 120, 672, 831, 455, 333, 234, 929, 781, 1012, 990, 1197, 921, 471, 816, 774, 64, 155, 1040, 364, 51, 1136, 182, 553, 995, 380, 639, 961, 513, 860, 186, 1028, 725, 788, 944, 739, 808, 1157, 609, 503, 966, 790, 83, 43, 633, 177, 299, 1219, 465, 723, 354, 792, 581, 520, 779, 615, 183, 671, 509, 41, 115, 101, 494, 1030, 872, 692, 314, 859, 457, 761, 25, 39, 636, 706, 271, 1216, 1027, 1132, 274, 909, 332, 6, 161, 847, 783, 448, 1010, 311, 853, 1171, 370, 1129, 862, 782, 1020, 794, 959, 137, 674, 1075, 679, 950, 1026, 327, 801, 868, 1061, 595, 539, 646, 737, 625, 433, 44, 663, 1202, 1051, 1049, 279, 1131, 481, 428, 565, 1035, 413, 353, 1122, 562, 791, 744, 1008, 682, 923, 470, 893, 359, 273, 238, 517, 652, 800, 736, 94, 149, 1127, 375, 569, 630, 261, 13, 956, 396, 383, 382, 582, 97, 241, 985, 846, 878, 60, 107, 638, 1103, 145, 693, 999, 324, 784, 270, 719, 1083, 1101, 738, 347, 339, 300, 399, 815, 583, 561, 37, 444, 146, 469, 821, 667, 753, 812, 1118, 653, 535, 53, 666, 100, 642, 1037, 497, 125, 516, 71, 113, 771, 1029, 16, 977, 531, 232, 651, 946, 874, 1001, 485, 904, 924, 222, 199, 395, 326, 767, 1175, 699, 702, 1156, 731, 1054, 576, 1089, 547, 523, 427, 142, 854, 803, 265, 890, 74, 165, 852, 1052, 114, 502, 190, 1091, 68, 559, 552, 811, 1102, 463, 1060, 1108, 659, 742, 650, 1079, 32, 530, 986, 740, 586, 660, 984, 317, 549, 912, 529, 244, 1076, 922, 928, 1178, 994, 611, 827, 838, 918, 483, 965, 1160, 953, 1004, 776, 805, 253, 1152, 603, 514, 587, 216, 880, 344, 1088, 298, 1116, 648, 308, 1050, 387, 1053, 55, 655, 760, 1043, 243, 629, 84, 22, 628, 38, 991, 1070, 823, 85, 431, 449, 848, 596, 392, 162, 1169, 296, 566, 591, 1191, 47, 632, 440, 865, 446, 832, 312, 164, 1180, 678, 361, 248, 443, 386, 1143, 698, 211, 229, 50, 329, 254, 900, 532, 575, 251, 1153, 564, 1007, 295, 456, 1176, 501, 780, 1174, 1009, 694, 109, 925, 993, 1080, 525, 895, 88, 751, 764, 310, 358, 690, 545, 417, 1042, 969, 246, 458, 349, 910, 1068, 926, 1134, 434, 537, 534, 1193, 1184, 716, 973, 1164, 188, 127, 5, 608, 888, 130, 1058, 1064, 474, 480, 224, 192, 341, 600, 408, 987, 1163, 896, 820, 34, 546, 676, 964, 571, 80, 1059, 665, 410, 732, 276, 1016, 954, 527, 476, 976, 228, 331, 204, 492, 255, 554, 871, 403, 560, 250, 885, 240, 1168, 998, 493, 833, 938, 143, 205, 184, 713, 309, 1071, 426, 908, 1115, 1094, 357, 1039, 323, 680, 867, 711, 515, 1105, 20, 1120, 829, 283, 1199, 934, 345, 664, 844, 432, 193, 734, 1190, 544, 640, 512, 1158, 743, 144, 1109, 668, 947, 123, 129, 849, 770, 752, 824, 81, 415, 437, 696, 371, 958, 189, 394, 884, 1023, 102, 875, 1107, 657, 18, 245, 905, 765, 810, 866, 919, 388, 1121, 621, 1021, 150, 856, 406, 1209, 19, 618, 563, 622, 185, 429, 56, 215, 762, 40, 873, 167, 10, 1141, 103, 616, 605, 835, 435, 158, 352, 932, 460, 894, 700, 256, 340, 556, 952, 1057, 814, 686, 389, 550, 195, 439, 61, 233, 540, 436, 573, 584, 441, 505, 1041, 1056, 467, 902, 230, 262, 1133, 750, 172, 430, 11, 351, 1205, 346, 1063, 787, 313, 669, 362, 166, 108, 519, 33, 1198, 1013, 82, 121, 979, 89, 116, 451, 828, 978, 393, 1006, 152, 914, 21, 983, 78, 507, 962, 1183, 1067, 69, 1221, 421, 1084, 707, 1147, 1218, 845, 159, 715, 133, 813, 1181, 614, 1214, 242, 24, 1087, 379, 804, 263, 1015, 933]
            elif task_name == 'visc_hc':
                indices = [9, 39, 15, 110, 83, 166, 61, 82, 68, 145, 128, 113, 102, 33, 8, 136, 115, 163, 96, 81, 60, 85, 170, 66, 4, 17, 37, 90, 109, 26, 131, 144, 147, 51, 104, 160, 98, 91, 99, 48, 117, 73, 59, 88, 78, 21, 151, 169, 173, 54, 178, 111, 118, 130, 179, 7, 67, 53, 126, 157, 133, 22, 153, 72, 69, 10, 80, 121, 16, 142, 161, 14, 35, 71, 38, 25, 62, 52, 152, 158, 100, 12, 75, 58, 127, 34, 114, 89, 43, 146, 177, 29, 28, 6, 46, 65, 159, 138, 57, 44, 107, 40, 134, 148, 112, 76, 129, 141, 94, 164, 103, 119, 154, 55, 106, 162, 50, 63, 143, 41, 5, 132, 137, 116, 70, 36, 122, 165, 180, 27, 24, 47, 150, 42, 13, 31, 174, 125, 167, 139, 124, 171, 120, 149, 155, 2, 79, 168, 123, 11, 87, 97, 45, 86, 93, 92, 101, 74, 1, 64, 56, 156, 30, 77, 49, 84, 105, 18, 108, 32, 23, 135, 172, 175, 20, 0, 19, 140, 3, 176, 95]
            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            split3 = int(np.floor(self.train_size * num_train))
            test_idx, valid_idx, train_idx = indices[:split2], indices[split2:split+split2], indices[split+split2:split+split2+split3]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader

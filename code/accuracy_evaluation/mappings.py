import os
import requests

# import numpy as np


def ordered_classnames():
    return ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane',
            'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle',
            'truck', 'car', 'bird', 'dog']

class HumanCategories(object):
    CLASS_INDEX_PATH = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    wnid_ordering = '/work/generalisation-humans-DNNs/code/accuracy_evaluation/categories.txt'
    
    hyp_wnids = {'knife': ['n03041632'],
                 'keyboard': ['n03085013', 'n04505470'],
                 'elephant': ['n02504013', 'n02504458'],
                 'bicycle': ['n02835271', 'n03792782'],
                 'airplane': ['n02690373', 'n03955296', 'n13861050',
                              'n13941806'],
                 'clock': ['n02708093', 'n03196217', 'n04548280'],
                 'oven': ['n03259401', 'n04111414', 'n04111531'],
                 'chair': ['n02791124', 'n03376595', 'n04099969',
                           'n00605023', 'n04429376'],
                 'bear': ['n02132136', 'n02133161', 'n02134084',
                          'n02134418'],
                 'boat': ['n02951358', 'n03344393', 'n03662601',
                          'n04273569', 'n04612373', 'n04612504'],
                 'cat': ["n02122878", "n02123045", "n02123159",
                         "n02126465", "n02123394", "n02123597",
                         "n02124075", "n02125311"],
                 'bottle': ['n02823428', 'n03937543', 'n03983396',
                            'n04557648', 'n04560804', 'n04579145',
                            'n04591713'],
                 'truck': ['n03345487', 'n03417042', 'n03770679',
                           'n03796401', 'n00319176', 'n01016201',
                           'n03930630', 'n03930777', 'n05061003',
                           'n06547832', 'n10432053', 'n03977966',
                           'n04461696', 'n04467665'],
                 'car': ['n02814533', 'n03100240', 'n03100346',
                         'n13419325', 'n04285008'],
                 'bird': ['n01321123', 'n01514859', 'n01792640',
                          'n07646067', 'n01530575', 'n01531178', 'n01532829',
                          'n01534433', 'n01537544', 'n01558993', 'n01562265',
                          'n01560419', 'n01582220', 'n10281276', 'n01592084',
                          'n01601694', 'n01614925', 'n01616318', 'n01622779',
                          'n01795545', 'n01796340', 'n01797886', 'n01798484',
                          'n01817953', 'n01818515', 'n01819313', 'n01820546',
                          'n01824575', 'n01828970', 'n01829413', 'n01833805',
                          'n01843065', 'n01843383', 'n01855032', 'n01855672',
                          'n07646821', 'n01860187', 'n02002556', 'n02002724',
                          'n02006656', 'n02007558', 'n02009229', 'n02009912',
                          'n02011460', 'n02013706', 'n02017213', 'n02018207',
                          'n02018795', 'n02025239', 'n02027492', 'n02028035',
                          'n02033041', 'n02037110', 'n02051845', 'n02056570'],
                 'dog': ['n02085782', 'n02085936', 'n02086079',
                         'n02086240', 'n02086646', 'n02086910', 'n02087046',
                         'n02087394', 'n02088094', 'n02088238', 'n02088364',
                         'n02088466', 'n02088632', 'n02089078', 'n02089867',
                         'n02089973', 'n02090379', 'n02090622', 'n02090721',
                         'n02091032', 'n02091134', 'n02091244', 'n02091467',
                         'n02091635', 'n02091831', 'n02092002', 'n02092339',
                         'n02093256', 'n02093428', 'n02093647', 'n02093754',
                         'n02093859', 'n02093991', 'n02094114', 'n02094258',
                         'n02094433', 'n02095314', 'n02095570', 'n02095889',
                         'n02096051', 'n02096294', 'n02096437', 'n02096585',
                         'n02097047', 'n02097130', 'n02097209', 'n02097298',
                         'n02097474', 'n02097658', 'n02098105', 'n02098286',
                         'n02099267', 'n02099429', 'n02099601', 'n02099712',
                         'n02099849', 'n02100236', 'n02100583', 'n02100735',
                         'n02100877', 'n02101006', 'n02101388', 'n02101556',
                         'n02102040', 'n02102177', 'n02102318', 'n02102480',
                         'n02102973', 'n02104029', 'n02104365', 'n02105056',
                         'n02105162', 'n02105251', 'n02105505', 'n02105641',
                         'n02105855', 'n02106030', 'n02106166', 'n02106382',
                         'n02106550', 'n02106662', 'n02107142', 'n02107312',
                         'n02107574', 'n02107683', 'n02107908', 'n02108000',
                         'n02108422', 'n02108551', 'n02108915', 'n02109047',
                         'n02109525', 'n02109961', 'n02110063', 'n02110185',
                         'n02110627', 'n02110806', 'n02110958', 'n02111129',
                         'n02111277', 'n08825211', 'n02111500', 'n02112018',
                         'n02112350', 'n02112706', 'n02113023', 'n02113624',
                         'n02113712', 'n02113799', 'n02113978']
                }

    def __init__(self):
        
        # Download ImageNet mappings: {str(ind): [wnid, label] for ind in range(1000)}
        self.CLASS_INDEX = requests.get(self.CLASS_INDEX_PATH).json()
        
        # Create WNID --> hypernym mapping from internal dictionary
        self.wnid_hyp_red = {wnid: hyp for hyp, wnids in self.hyp_wnids.items()
                                       for wnid in wnids}

        # Load ordered WNIDs from file
        with open(self.wnid_ordering, 'r') as wof:
            self.ind_wnid = [line.split()[0] for line in wof]

        # Create WNID --> hypernym mapping from full WNIDs in file
        self.wnid_hyp = {wnid: self.wnid_hyp_red.get(wnid) for wnid in self.ind_wnid}
        
        # Create hypernym --> index mapping
        self.hyp_ind = {hyp: ind for ind, hyp in enumerate(list(self.hyp_wnids))}

        # Add some checks
        assert len(self.wnid_hyp) == 1000
        assert len(self.wnid_hyp_red) == len(set(self.wnid_hyp_red)) # No duplicates
        # assert len(self.wnid_hyp_red) == sum([len(wnids) for _, wnids in self.hyp_wnids.items()])
        assert self.ind_wnid == sorted(list(self.wnid_hyp))  # ind_wnid is sorted

    def get_hypernym(self, wnid):
        return self.wnid_hyp[wnid]

#     def get_name_from_WNID(self, wnid):
#         for category, wnid_list in self.hyp_wnids.items():
#             if wnid in wnid_list:
#                 return category
#         return None

    def get_hypernyms(self):
        return list(self.hyp_wnids)
#         return sorted(self.hyp_wnids)

    def ind1000_wnid(self, ind1000):
        return self.CLASS_INDEX[str(ind1000)][0]

    def ind1000_label(self, ind1000):
        return self.CLASS_INDEX[str(ind1000)][1]

    def ind1000_ind16(self, ind1000):
        # ind1000 --> WNID --> hypernym --> ind16
        wnid = self.ind_wnid[ind1000]
        hypernym = self.get_hypernym(wnid)
        if hypernym:
            return self.hyp_ind[hypernym]
        else:
            return None  # np.nan

    def ind1000_hyp(self, ind1000):
        wnid = self.ind_wnid[ind1000]
        hypernym = self.get_hypernym(wnid)
        if hypernym:
            return hypernym
        else:
            return None  # np.nan

hc = HumanCategories()

# for wnid, hypernym in hc.wnid_hyp_red.items():
#     assert hypernym == hc.get_name_from_WNID(wnid)

# Check for differences between internal and file-listed WNIDs
wnids_file = [wnid for wnid in hc.ind_wnid if hc.wnid_hyp[wnid]]  # Gather WNIDs from file
print(f"WNIDs in {os.path.basename(hc.wnid_ordering)}: {len(wnids_file)}")
print(f"WNIDs in 'human_categories.py': {len(hc.wnid_hyp_red)}")  # == len(wnids)
missing = set(hc.wnid_hyp_red) - set(wnids_file)
print(f"Missing hypernyms: {len(missing)}")
for wnid in missing:
    print(wnid, hc.wnid_hyp_red[wnid])
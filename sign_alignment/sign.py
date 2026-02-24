
from dataclasses import dataclass
import os
from pymongo import MongoClient

# ============ Class Names ============
# Note: these lists are generate according to the ebl database
# class names list from model
# model.dataset_meta["classes"]
CLASSES_NAME = ['TU', '|U.GUD|', 'TUM', 'LA', 'TA', 'GAR', 'GAL', 'I', 'TI', 'LI', 'ZA', 'A', 'DI', 'MI', 'RI', 'IŠ', 'BA', 'LU', 'TE', 'DA', '|GUD×KUR|', 'MA', 'E₂', 'DIŠ', 'MU', 'DU', 'ŠU₂', 'EN', 'KUL', 'SI', '|I.A|', 'HI', 'MUŠ₃', 'AN', 'NA', 'BAD', 'AMAR', 'UD', 'UnclearSign', '|HI×BAD|', '|UD×(U.U.U)|', 'AB', 'AK', 'LUGAL', 'DIN', 'KI', 'DUN₃@g', 'KU₃', 'AŠ', 'IGI', 'U₂', 'ŠA₃', 'BI', 'GUR', 'ŠE', 'ZI', 'GA', 'SILA₃', 'ŠID', '|SAL.TUG₂|', 'SU', 'KAK', 'MAŠ', 'TUR', 'ŠEŠ', 'LU₂', 'IA₂', 'UR', 'KAL', '|ŠEŠ.KI|', 'ZU', 'ŠU', 'NE', 'IM', 'RA', '|U.U|', 'ZAG', '|DIŠ.DIŠ.DIŠ|', 'GA₂', 'IN', 'KIN', 'TAR', 'MAH', 'LAL', 'KID', 'GABA', 'KA', 'RU', 'ŠA', '|HI×NUN|', 'ME', 'BU', 'NI', 'IG', 'MES', 'PA', 'SAG', 'U', 'E', 'GUM', 'GIŠ', '|U.KA|', 'LUM', '|HI×AŠ₂|', 'HA', 'UŠ', '|U.U.U|', 'MIN', 'NAM', 'NU', 'AL', 'AB₂', 'IB', 'UM', 'KU', 'SUR', 'MEŠ', 'TUG₂', 'TAG', 'DIM', 'BAL', 'IR', 'ERIN₂', 'PAP', 'SA', '|PIRIG×ZA|', 'UB', 'URU', '|U.5(DIŠ)|', 'DAM', 'GAR₃', '|EN.ZU|', 'ZE₂', 'AD', 'APIN', 'EL', 'PI', 'AŠ@z', 'DAR', 'DUB', 'SAR', 'GUD', 'A₂', 'KUR', 'ARAD', '|IGI.DIB|', '6(DIŠ)', 'AŠ₂', 'IL', 'HU', 'NUN', 'SAL', 'GI', 'EŠ₂', 'UN', 'TIL', 'NIM', 'TAB', 'SUM', '|3×AN|', '|NINDA₂×ŠE|', 'MAŠ₂', 'GI₄', 'GAN', 'DIM₂', 'GU', 'MAR', 'MUŠ', 'BAR', '|IGI.RI|', 'TUK', '|UD.DU|', '|LAGAB.LAGAB|']

# ABZ class names from model
CLASSES_ABZ = ['ABZ58', 'ABZ441', 'ABZ207', 'ABZ55', 'ABZ139', 'ABZ597', 'ABZ343', 'ABZ142', 'ABZ73', 'ABZ59', 'ABZ586', 'ABZ579', 'ABZ457', 'ABZ427', 'ABZ86', 'ABZ212', 'ABZ5', 'ABZ537', 'ABZ376', 'ABZ335', 'ABZ170', 'ABZ342', 'ABZ324', 'ABZ480', 'ABZ61', 'ABZ206', 'ABZ545', 'ABZ99', 'ABZ72', 'ABZ112', 'ABZ142a', 'ABZ396', 'ABZ103', 'ABZ13', 'ABZ70', 'ABZ69', 'ABZ437', 'ABZ381', 'X', 'ABZ279', 'ABZ52', 'ABZ128', 'ABZ97', 'ABZ151', 'ABZ465', 'ABZ461', 'ABZ595', 'ABZ468', 'ABZ1', 'ABZ449', 'ABZ318', 'ABZ384', 'ABZ214', 'ABZ111', 'ABZ367', 'ABZ84', 'ABZ319', 'ABZ62', 'ABZ314', 'ABZ556', 'ABZ7', 'ABZ230', 'ABZ74', 'ABZ144', 'ABZ331', 'ABZ330', 'ABZ598a', 'ABZ575', 'ABZ322', 'NoABZ0', 'ABZ6', 'ABZ354', 'ABZ172', 'ABZ399', 'ABZ328', 'ABZ471', 'ABZ332', 'ABZ593', 'ABZ233', 'ABZ148', 'ABZ538', 'ABZ12', 'ABZ57', 'ABZ481', 'ABZ313', 'ABZ167', 'ABZ15', 'ABZ68', 'ABZ353', 'ABZ398', 'ABZ532', 'ABZ371', 'ABZ231', 'ABZ80', 'ABZ314', 'ABZ295', 'ABZ115', 'ABZ411', 'ABZ308', 'ABZ191', 'ABZ296', 'ABZ412', 'ABZ565', 'ABZ401', 'ABZ589', 'ABZ211', 'ABZ472', 'ABZ570', 'ABZ79', 'ABZ75', 'ABZ298', 'ABZ420', 'ABZ535', 'ABZ134', 'ABZ536', 'ABZ101', 'ABZ533', 'ABZ536', 'ABZ126', 'ABZ94', 'ABZ9', 'ABZ232', 'ABZ393', 'ABZ60', 'ABZ104', 'ABZ131', 'ABZ306', 'ABZ38', 'ABZ470', 'ABZ557', 'ABZ333', 'NoABZ0', 'ABZ147', 'ABZ145', 'ABZ56', 'ABZ564', 'ABZ383', 'ABZ360', 'ABZ114', 'ABZ138', 'ABZ331e+152i', 'ABZ297', 'ABZ334', 'ABZ366', 'ABZ50', 'ABZ455', 'ABZ598b', 'ABZ339', 'ABZ205', 'ABZ78', 'ABZ87', 'ABZ554', 'ABZ85', 'ABZ536', 'ABZ312', 'ABZ69', 'ABZ433', 'ABZ124', 'ABZ164', 'ABZ129a', 'NoABZ0', 'ABZ76', 'ABZ326', 'ABZ143', 'ABZ440', 'ABZ559', 'ABZ307', 'ABZ374', 'ABZ74', 'ABZ451', 'ABZ574', 'NoABZ0', 'ABZ529']


# ============ MongoDB Connection for ABZ -> Sign Name ============
# Note: we don't actually use the DB in this version
# get MongoDB connection string from ...
uri = os.getenv('MONGODB_URI', 'YOUR_MONGODB_URI')
client = MongoClient(uri)
db = client['ebl']
signs_collection = db['signs']

def build_mappings():

    abz_to_sign = {}
    sign_to_abz = {}
    
    # Build mappings from the parallel arrays
    for abz_name, sign_name in zip(CLASSES_ABZ, CLASSES_NAME):
        # Only add if not already present (handles duplicates by keeping first occurrence)
        if abz_name not in abz_to_sign:
            abz_to_sign[abz_name] = sign_name
        if sign_name not in sign_to_abz:
            sign_to_abz[sign_name] = abz_name
    
    abz_to_sign['X'] = 'UnclearSign'
    abz_to_sign['NoABZ0'] = 'UnclearSign'
    sign_to_abz['UnclearSign'] = 'X'
    return abz_to_sign, sign_to_abz

# Canonical sign representation
@dataclass(frozen=True)
class Sign:
    abz: str
    name: str
    idx: int

class SignResolver:
    ABZ_TO_SIGN, SIGN_TO_ABZ = build_mappings()

    @classmethod
    def from_abz(cls, abz):
        name = cls.ABZ_TO_SIGN.get(abz, 'UnclearSign')
        idx = CLASSES_ABZ.index(abz) if abz in CLASSES_ABZ else -1
        return Sign(abz=abz, name=name, idx=idx)
    
    @classmethod
    def from_name(cls, name):
        abz = cls.SIGN_TO_ABZ.get(name, 'X')
        idx = CLASSES_ABZ.index(abz) if abz in CLASSES_ABZ else -1
        return Sign(abz=abz, name=name, idx=idx)
    
    @classmethod
    def from_idx(cls, idx):
        if idx < 0 or idx >= len(CLASSES_ABZ):
            return Sign(abz='X', name='UnclearSign', idx=-1)
        abz = CLASSES_ABZ[idx]
        name = cls.ABZ_TO_SIGN.get(abz, 'UnclearSign')
        return Sign(abz=abz, name=name, idx=idx)
    
    @classmethod
    def resolve(cls, sign, expected_type=None):
        type = None
        if expected_type not in (None, 'ABZ', 'SIGN', 'INDEX'):
            raise ValueError("expected type must be one of None, 'ABZ', 'SIGN', 'INDEX'")
        if isinstance(sign, int):
            type = 'INDEX'
        elif sign.startswith('ABZ') or sign.startswith('NoABZ') or sign == 'X':
            type = 'ABZ'
        else:
            type = 'SIGN'

        if expected_type and type != expected_type:
            print(f"Warning: sign type of sign '{sign}' mismatch. Expected {expected_type}, got {type}. Setting to 'X'.")
            sign = "X"
            type = 'ABZ'

        if type == 'INDEX':
            return cls.from_idx(sign)
        elif type == 'ABZ':
            return cls.from_abz(sign)
        else:
            return cls.from_name(sign)



        
# ============ Sign Name Converter ============

# class SignNameConverter:
#     def __init__(self, sign=None, expected_type=None):
#         if expected_type not in (None, 'ABZ', 'SIGN', 'INDEX'):
#             raise ValueError("expected type must be one of None, 'ABZ', 'SIGN', 'INDEX'")
#         if sign != None:
#             if expected_type:
#                 name_type = self._check_name_type(sign)
#                 if name_type != expected_type:
#                     print(f"Warning: sign type of sign '{sign}' mismatch. Expected {expected_type}, got {name_type}. Setting to 'X'.")
#                     sign = "X"
#             self._set_signs(sign)
    
#     def get_abz(self):
#         return self.abz 
#     def get_sign_name(self):
#         return self.sign_name
#     def get_sign_idx(self):
#         return self.sign_idx
#     def set_signs(self, sign):
#         self._set_signs(sign)

#     def _set_signs(self, sign):
#         self.original_sign = sign
#         sign_type = self._check_name_type(sign=self.original_sign)
#         if sign_type == 'ABZ':
#             self.abz = sign
#             self.sign_name = self._convert_abz_to_sign_name(sign)
#             if self.abz in CLASSES_ABZ:
#                 self.sign_idx = CLASSES_ABZ.index(self.abz)
#             else:
#                 self.sign_idx = -1 # outside of known CLASSES_ABZ, which only possible for annotated signs, not detected ones
#         elif sign_type == 'SIGN':
#             self.sign_name = sign
#             self.abz = self._convert_sign_name_to_abz(sign)
#             if self.abz in CLASSES_ABZ:
#                 self.sign_idx = CLASSES_ABZ.index(self.abz)
#             else:
#                 self.sign_idx = -1 # outside of known CLASSES_ABZ, which only possible for annotated signs, not detected ones
#         elif sign_type == 'INDEX':
#             self.sign_idx = sign
#             self.abz = CLASSES_ABZ[self.sign_idx]
#             self.sign_name = self._convert_abz_to_sign_name(self.abz)
#         else:
#             raise TypeError

#     @staticmethod
#     def _build_mappings():

#         abz_to_sign = {}
#         sign_to_abz = {}
        
#         # Build mappings from the parallel arrays
#         for abz_name, sign_name in zip(CLASSES_ABZ, CLASSES_NAME):
#             # Only add if not already present (handles duplicates by keeping first occurrence)
#             if abz_name not in abz_to_sign:
#                 abz_to_sign[abz_name] = sign_name
#             if sign_name not in sign_to_abz:
#                 sign_to_abz[sign_name] = abz_name
        
#         abz_to_sign['X'] = 'UnclearSign'
#         abz_to_sign['NoABZ0'] = 'UnclearSign'
#         sign_to_abz['UnclearSign'] = 'X'

#         return abz_to_sign, sign_to_abz
    
#     ABZ_TO_SIGN, SIGN_TO_ABZ = _build_mappings.__func__()
    
#     def _check_name_type(self, sign):
#         if isinstance(sign, int):
#             return 'INDEX'
#         if sign.startswith('ABZ') or sign.startswith('NoABZ') or sign == 'X':
#             return 'ABZ'
#         return 'SIGN'
    
#     def _convert_abz_to_sign_name(self, sign):
#         return self.ABZ_TO_SIGN.get(sign, 'UnclearSign')

#     def _convert_sign_name_to_abz(self, sign_name):
#         return self.SIGN_TO_ABZ.get(sign_name, 'X')

# def abz_to_sign_name(abz):
#     converter = SignNameConverter(abz)
#     return converter.get_sign_name()     
#!/usr/bin/env python3
"""Converter for sign names and ABZ codes"""

from dataclasses import dataclass
from typing import Optional, Dict
from pymongo import MongoClient

# ABZ class names
CLASSES_ABZ = ['ABZ58', 'ABZ441', 'ABZ207', 'ABZ55', 'ABZ139', 'ABZ597', 'ABZ343', 'ABZ142', 'ABZ73', 'ABZ59', 'ABZ586', 'ABZ579', 'ABZ457', 'ABZ427', 'ABZ86', 'ABZ212', 'ABZ5', 'ABZ537', 'ABZ376', 'ABZ335', 'ABZ170', 'ABZ342', 'ABZ324', 'ABZ480', 'ABZ61', 'ABZ206', 'ABZ545', 'ABZ99', 'ABZ72', 'ABZ112', 'ABZ142a', 'ABZ396', 'ABZ103', 'ABZ13', 'ABZ70', 'ABZ69', 'ABZ437', 'ABZ381', 'X', 'ABZ279', 'ABZ52', 'ABZ128', 'ABZ97', 'ABZ151', 'ABZ465', 'ABZ461', 'ABZ595', 'ABZ468', 'ABZ1', 'ABZ449', 'ABZ318', 'ABZ384', 'ABZ214', 'ABZ111', 'ABZ367', 'ABZ84', 'ABZ319', 'ABZ62', 'ABZ314', 'ABZ556', 'ABZ7', 'ABZ230', 'ABZ74', 'ABZ144', 'ABZ331', 'ABZ330', 'ABZ598a', 'ABZ575', 'ABZ322', 'NoABZ0', 'ABZ6', 'ABZ354', 'ABZ172', 'ABZ399', 'ABZ328', 'ABZ471', 'ABZ332', 'ABZ593', 'ABZ233', 'ABZ148', 'ABZ538', 'ABZ12', 'ABZ57', 'ABZ481', 'ABZ313', 'ABZ167', 'ABZ15', 'ABZ68', 'ABZ353', 'ABZ398', 'ABZ532', 'ABZ371', 'ABZ231', 'ABZ80', 'ABZ314', 'ABZ295', 'ABZ115', 'ABZ411', 'ABZ308', 'ABZ191', 'ABZ296', 'ABZ412', 'ABZ565', 'ABZ401', 'ABZ589', 'ABZ211', 'ABZ472', 'ABZ570', 'ABZ79', 'ABZ75', 'ABZ298', 'ABZ420', 'ABZ535', 'ABZ134', 'ABZ536', 'ABZ101', 'ABZ533', 'ABZ536', 'ABZ126', 'ABZ94', 'ABZ9', 'ABZ232', 'ABZ393', 'ABZ60', 'ABZ104', 'ABZ131', 'ABZ306', 'ABZ38', 'ABZ470', 'ABZ557', 'ABZ333', 'NoABZ0', 'ABZ147', 'ABZ145', 'ABZ56', 'ABZ564', 'ABZ383', 'ABZ360', 'ABZ114', 'ABZ138', 'ABZ331e+152i', 'ABZ297', 'ABZ334', 'ABZ366', 'ABZ50', 'ABZ455', 'ABZ598b', 'ABZ339', 'ABZ205', 'ABZ78', 'ABZ87', 'ABZ554', 'ABZ85', 'ABZ536', 'ABZ312', 'ABZ69', 'ABZ433', 'ABZ124', 'ABZ164', 'ABZ129a', 'NoABZ0', 'ABZ76', 'ABZ326', 'ABZ143', 'ABZ440', 'ABZ559', 'ABZ307', 'ABZ374', 'ABZ74', 'ABZ451', 'ABZ574', 'NoABZ0', 'ABZ529']

# Sign names
CLASSES_NAME = ['TU', '|U.GUD|', 'TUM', 'LA', 'TA', 'GAR', 'GAL', 'I', 'TI', 'LI', 'ZA', 'A', 'DI', 'MI', 'RI', 'IŠ', 'BA', 'LU', 'TE', 'DA', '|GUD×KUR|', 'MA', 'E₂', 'DIŠ', 'MU', 'DU', 'ŠU₂', 'EN', 'KUL', 'SI', '|I.A|', 'HI', 'MUŠ₃', 'AN', 'NA', 'BAD', 'AMAR', 'UD', 'UnclearSign', '|HI×BAD|', '|UD×(U.U.U)|', 'AB', 'AK', 'LUGAL', 'DIN', 'KI', 'DUN₃@g', 'KU₃', 'AŠ', 'IGI', 'U₂', 'ŠA₃', 'BI', 'GUR', 'ŠE', 'ZI', 'GA', 'SILA₃', 'ŠID', '|SAL.TUG₂|', 'SU', 'KAK', 'MAŠ', 'TUR', 'ŠEŠ', 'LU₂', 'IA₂', 'UR', 'KAL', '|ŠEŠ.KI|', 'ZU', 'ŠU', 'NE', 'IM', 'RA', '|U.U|', 'ZAG', '|DIŠ.DIŠ.DIŠ|', 'GA₂', 'IN', 'KIN', 'TAR', 'MAH', 'LAL', 'KID', 'GABA', 'KA', 'RU', 'ŠA', '|HI×NUN|', 'ME', 'BU', 'NI', 'IG', 'MES', 'PA', 'SAG', 'U', 'E', 'GUM', 'GIŠ', '|U.KA|', 'LUM', '|HI×AŠ₂|', 'HA', 'UŠ', '|U.U.U|', 'MIN', 'NAM', 'NU', 'AL', 'AB₂', 'IB', 'UM', 'KU', 'SUR', 'MEŠ', 'TUG₂', 'TAG', 'DIM', 'BAL', 'IR', 'ERIN₂', 'PAP', 'SA', '|PIRIG×ZA|', 'UB', 'URU', '|U.5(DIŠ)|', 'DAM', 'GAR₃', '|EN.ZU|', 'ZE₂', 'AD', 'APIN', 'EL', 'PI', 'AŠ@z', 'DAR', 'DUB', 'SAR', 'GUD', 'A₂', 'KUR', 'ARAD', '|IGI.DIB|', '6(DIŠ)', 'AŠ₂', 'IL', 'HU', 'NUN', 'SAL', 'GI', 'EŠ₂', 'UN', 'TIL', 'NIM', 'TAB', 'SUM', '|3×AN|', '|NINDA₂×ŠE|', 'MAŠ₂', 'GI₄', 'GAN', 'DIM₂', 'GU', 'MAR', 'MUŠ', 'BAR', '|IGI.RI|', 'TUK', '|UD.DU|', '|LAGAB.LAGAB|']


def build_mappings():
    """Build mappings between ABZ and sign names"""
    abz_to_sign = {}
    sign_to_abz = {}
    
    for abz_name, sign_name in zip(CLASSES_ABZ, CLASSES_NAME):
        if abz_name not in abz_to_sign:
            abz_to_sign[abz_name] = sign_name
        if sign_name not in sign_to_abz:
            sign_to_abz[sign_name] = abz_name
    
    abz_to_sign['X'] = 'UnclearSign'
    abz_to_sign['NoABZ0'] = 'UnclearSign'
    sign_to_abz['UnclearSign'] = 'X'
    
    return abz_to_sign, sign_to_abz


@dataclass(frozen=True)
class Sign:
    abz: str
    name: str
    idx: int


class SignResolver:
    """Converter for sign names and ABZ codes"""
    
    ABZ_TO_SIGN, SIGN_TO_ABZ = build_mappings()

    @classmethod
    def from_abz(cls, abz):
        """Create Sign object from ABZ code"""
        name = cls.ABZ_TO_SIGN.get(abz, 'UnclearSign')
        idx = CLASSES_ABZ.index(abz) if abz in CLASSES_ABZ else -1
        return Sign(abz=abz, name=name, idx=idx)
    
    @classmethod
    def from_name(cls, name):
        """Create Sign object from sign name"""
        abz = cls.SIGN_TO_ABZ.get(name, 'X')
        idx = CLASSES_ABZ.index(abz) if abz in CLASSES_ABZ else -1
        return Sign(abz=abz, name=name, idx=idx)
    
    @classmethod
    def from_idx(cls, idx):
        """Create Sign object from index"""
        if idx < 0 or idx >= len(CLASSES_ABZ):
            return Sign(abz='X', name='UnclearSign', idx=-1)
        abz = CLASSES_ABZ[idx]
        name = cls.ABZ_TO_SIGN.get(abz, 'UnclearSign')
        return Sign(abz=abz, name=name, idx=idx)
    
    @classmethod
    def resolve(cls, sign, expected_type=None):
        """Auto-detect and convert signs"""
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


class SignToABZResolver:
    """Sign name to ABZ number converter (MongoDB query with cache)"""
    
    def __init__(self, mongodb_uri: str):
        self.mongodb_uri = mongodb_uri
        self.client = None
        self.collection = None
        self._cache: Dict[str, Optional[str]] = {}
        self._connected = False
        
        # Initialize cache with static mappings (only cache existing mappings)
        for sign_name, abz in SignResolver.SIGN_TO_ABZ.items():
            if abz and abz != 'X':  # Only cache valid mappings
                self._cache[sign_name] = abz
    
    def connect(self):
        """Connect to MongoDB"""
        if not self._connected:
            self.client = MongoClient(self.mongodb_uri)
            db = self.client['ebl']
            self.collection = db['signs']
            self._connected = True
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self._connected = False
    
    def get_abz_number(self, sign: str) -> Optional[str]:
        """
        Get the ABZ number for a sign
        
        Args:
            sign: Sign name (e.g., 'u', 'qa', 'UnclearSign')
            
        Returns:
            ABZ number (e.g., 'ABZ1', 'ABZ15') or 'X' (unclear sign),
            None if not found
        """
        # Special handling
        if sign == "UnclearSign":
            return "X"
        
        # Check cache
        if sign in self._cache:
            return self._cache[sign]
        
        # Ensure connected
        if not self._connected:
            self.connect()
        
        # Query database - first try original format
        try:
            result = self.collection.find_one({"_id": sign})
            if result:
                for entry in result.get('lists', []):
                    if entry.get('name') == 'ABZ':
                        abz_number = 'ABZ' + entry['number']
                        self._cache[sign] = abz_number
                        return abz_number
            
            # If original format not found, try uppercase format
            sign_upper = sign.upper()
            if sign_upper != sign:
                # If uppercase version already in cache, use it directly
                if sign_upper in self._cache:
                    abz_number = self._cache[sign_upper]
                    self._cache[sign] = abz_number  # Also cache lowercase version
                    return abz_number
                
                # Otherwise query database
                result = self.collection.find_one({"_id": sign_upper})
                if result:
                    for entry in result.get('lists', []):
                        if entry.get('name') == 'ABZ':
                            abz_number = 'ABZ' + entry['number']
                            # Cache both original and uppercase formats
                            self._cache[sign] = abz_number
                            self._cache[sign_upper] = abz_number
                            return abz_number
            
            # ABZ number not found, cache None
            self._cache[sign] = None
            return None
        except Exception as e:
            print(f"Error querying sign '{sign}': {e}")
            return None
    
    def convert_signs_to_abz(self, signs: str) -> str:
        """
        Convert space-separated sign string to ABZ format
        
        Args:
            signs: Space-separated sign string (e.g., "u qa tum")
            
        Returns:
            ABZ format string (e.g., "ABZ1 ABZ15 ABZ537")
        """
        if not signs:
            return ""
        
        sign_list = signs.split()
        abz_list = []
        
        for sign in sign_list:
            abz = self.get_abz_number(sign)
            if abz:
                abz_list.append(abz)
            # If ABZ number not found, skip the sign (not included in output)
        
        return ' '.join(abz_list)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_signs': len(self._cache),
            'found': sum(1 for v in self._cache.values() if v is not None),
            'not_found': sum(1 for v in self._cache.values() if v is None)
        }
    
    def __enter__(self):
        """Support for with statement"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for with statement"""
        self.close()

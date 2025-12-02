#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertisseur DNG vers TIFF pour images de photogrammétrie
Permet de convertir les fichiers DNG en TIFF sans correction de distorsion
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import rawpy
import numpy as np
from tqdm import tqdm
import logging
import piexif
import exifread

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DNGToTIFFConverter:
    """Convertisseur de fichiers DNG vers TIFF"""
    
    def __init__(self, input_dir, output_dir=None, quality=95, keep_16bit=False, brightness=1.5, contrast=1.0, force_orientation=None):
        """
        Initialise le convertisseur
        
        Args:
            input_dir (str): Répertoire contenant les fichiers DNG
            output_dir (str): Répertoire de sortie (optionnel)
            quality (int): Qualité de compression TIFF (1-100)
            keep_16bit (bool): Conserver les 16 bits (recommandé pour photogrammétrie)
            brightness (float): Facteur de luminosité (0.5-2.0, défaut 1.5)
            contrast (float): Facteur de contraste (0.5-2.0, défaut 1.0)
            force_orientation (str): Forcer l'orientation ("landscape" ou "portrait", None pour conserver l'original)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "TIFF_output"
        
        # Validation et conversion de la qualité
        try:
            self.quality = int(quality)
            if not (1 <= self.quality <= 100):
                raise ValueError("La qualité doit être entre 1 et 100")
        except (ValueError, TypeError) as e:
            logger.warning(f"Qualité invalide '{quality}', utilisation de la valeur par défaut 95")
            self.quality = 95
        
        self.keep_16bit = keep_16bit
        
        # Validation de la luminosité
        try:
            self.brightness = float(brightness)
            if not (0.5 <= self.brightness <= 2.0):
                raise ValueError("La luminosité doit être entre 0.5 et 2.0")
        except (ValueError, TypeError) as e:
            logger.warning(f"Luminosité invalide '{brightness}', utilisation de la valeur par défaut 1.5")
            self.brightness = 1.5
        
        # Validation du contraste
        try:
            self.contrast = float(contrast)
            if not (0.5 <= self.contrast <= 2.0):
                raise ValueError("Le contraste doit être entre 0.5 et 2.0")
        except (ValueError, TypeError) as e:
            logger.warning(f"Contraste invalide '{contrast}', utilisation de la valeur par défaut 1.0")
            self.contrast = 1.0
        
        # Validation de l'orientation forcée
        if force_orientation is not None:
            force_orientation = str(force_orientation).lower()
            if force_orientation not in ['landscape', 'portrait']:
                logger.warning(f"Orientation forcée invalide '{force_orientation}', ignorée (doit être 'landscape' ou 'portrait')")
                self.force_orientation = None
            else:
                self.force_orientation = force_orientation
        else:
            self.force_orientation = None
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Répertoire d'entrée: {self.input_dir}")
        logger.info(f"Répertoire de sortie: {self.output_dir}")
    
    def find_dng_files(self):
        """Trouve tous les fichiers DNG dans le répertoire d'entrée"""
        # Utiliser un set pour éviter les doublons (Windows n'est pas sensible à la casse)
        dng_files = set(self.input_dir.glob("*.DNG")) | set(self.input_dir.glob("*.dng"))
        dng_files = sorted(list(dng_files))  # Convertir en liste triée
        logger.info(f"Trouvé {len(dng_files)} fichier(s) DNG")
        return dng_files
    
    def get_crop_factor(self, make, model, sensor_width=None, sensor_height=None):
        """
        Détermine le facteur de conversion (crop factor) pour calculer la focale équivalente 35mm
        
        Args:
            make: Marque de l'appareil photo
            model: Modèle de l'appareil photo
            sensor_width: Largeur du capteur en mm (si disponible dans les EXIF)
            sensor_height: Hauteur du capteur en mm (si disponible dans les EXIF)
            
        Returns:
            float: Facteur de conversion (1.0 = plein format, 2.0 = Micro Four Thirds, etc.)
        """
        # Si on a les dimensions du capteur, calculer le facteur précisément
        if sensor_width and sensor_height:
            # Capteur plein format = 36mm x 24mm
            full_frame_diagonal = (36**2 + 24**2)**0.5  # ≈ 43.27mm
            sensor_diagonal = (sensor_width**2 + sensor_height**2)**0.5
            crop_factor = full_frame_diagonal / sensor_diagonal
            logger.info(f"Facteur de conversion calculé depuis les dimensions du capteur: {crop_factor:.2f}x")
            return crop_factor
        
        # Sinon, utiliser les valeurs connues par marque/modèle
        make_lower = str(make).lower() if make else ""
        model_lower = str(model).lower() if model else ""
        
        # Micro Four Thirds (Olympus, Panasonic, OM Digital) - 17.3mm x 13mm
        if any(brand in make_lower for brand in ['olympus', 'om digital', 'panasonic']):
            logger.info(f"Micro Four Thirds détecté ({make} {model}), facteur: 2.0x")
            return 2.0
        
        # APS-C Canon (22.3mm x 14.9mm) - 1.6x
        if 'canon' in make_lower:
            # Certains modèles Canon sont plein format
            if any(full_frame in model_lower for full_frame in ['1d', '5d', '6d', 'r5', 'r6', 'r3']):
                logger.info(f"Canon plein format détecté ({model}), facteur: 1.0x")
                return 1.0
            logger.info(f"Canon APS-C détecté ({model}), facteur: 1.6x")
            return 1.6
        
        # APS-C Nikon (23.5mm x 15.6mm) - 1.5x
        if 'nikon' in make_lower:
            # Les modèles plein format
            if any(full_frame in model_lower for full_frame in ['d3', 'd4', 'd5', 'd6', 'd700', 'd800', 'd810', 'd850', 'z7', 'z9', 'z8']):
                logger.info(f"Nikon plein format détecté ({model}), facteur: 1.0x")
                return 1.0
            logger.info(f"Nikon APS-C détecté ({model}), facteur: 1.5x")
            return 1.5
        
        # Sony
        if 'sony' in make_lower:
            # A7, A9, A1 sont plein format
            if any(full_frame in model_lower for full_frame in ['a7', 'a9', 'a1']):
                logger.info(f"Sony plein format détecté ({model}), facteur: 1.0x")
                return 1.0
            # A6000, A6300, A6400, A6500, A6600 sont APS-C
            if any(aps_c in model_lower for aps_c in ['a6000', 'a6300', 'a6400', 'a6500', 'a6600']):
                logger.info(f"Sony APS-C détecté ({model}), facteur: 1.5x")
                return 1.5
            # Par défaut, considérer APS-C pour Sony
            logger.warning(f"Sony modèle non reconnu ({model}), utilisation du facteur APS-C par défaut: 1.5x")
            return 1.5
        
        # Fujifilm (APS-C généralement)
        if 'fujifilm' in make_lower or 'fuji' in make_lower:
            # GFX sont moyen format (0.79x)
            if 'gfx' in model_lower:
                logger.info(f"Fujifilm moyen format détecté ({model}), facteur: 0.79x")
                return 0.79
            logger.info(f"Fujifilm APS-C détecté ({model}), facteur: 1.5x")
            return 1.5
        
        # Pentax
        if 'pentax' in make_lower:
            # K-1 est plein format
            if 'k-1' in model_lower:
                logger.info(f"Pentax plein format détecté ({model}), facteur: 1.0x")
                return 1.0
            logger.info(f"Pentax APS-C détecté ({model}), facteur: 1.5x")
            return 1.5
        
        # Leica (souvent plein format)
        if 'leica' in make_lower:
            logger.info(f"Leica détecté ({model}), facteur: 1.0x (plein format présumé)")
            return 1.0
        
        # Hasselblad (moyen format)
        if 'hasselblad' in make_lower:
            logger.info(f"Hasselblad détecté ({model}), facteur: 0.64x (moyen format)")
            return 0.64
        
        # Par défaut, si on ne connaît pas, ne pas calculer la focale équivalente
        logger.warning(f"Marque/modèle inconnu ({make}/{model}), impossible de déterminer le facteur de conversion")
        logger.warning("Suggestion: vérifiez les spécifications de votre appareil ou ajoutez-le manuellement")
        return None
    
    def extract_exif_metadata(self, dng_path):
        """
        Extrait les métadonnées EXIF importantes du fichier RAW en utilisant exifread
        
        Args:
            dng_path: Chemin vers le fichier DNG
            
        Returns:
            dict: Dictionnaire des métadonnées EXIF
        """
        exif_data = {}
        
        try:
            # Lire les métadonnées EXIF directement depuis le fichier DNG
            with open(dng_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            
            # Extraire les métadonnées importantes
            if 'EXIF FocalLength' in tags:
                focal_length = float(tags['EXIF FocalLength'].values[0])
                exif_data['FocalLength'] = focal_length
            
            if 'EXIF FNumber' in tags:
                f_number = float(tags['EXIF FNumber'].values[0])
                exif_data['FNumber'] = f_number
            
            if 'EXIF ExposureTime' in tags:
                exposure_time = float(tags['EXIF ExposureTime'].values[0])
                exif_data['ExposureTime'] = exposure_time
            
            if 'EXIF ISOSpeedRatings' in tags:
                iso = int(tags['EXIF ISOSpeedRatings'].values[0])
                exif_data['ISOSpeedRatings'] = iso
            
            if 'Image Make' in tags:
                make = str(tags['Image Make'].values)
                exif_data['Make'] = make
            
            if 'Image Model' in tags:
                model = str(tags['Image Model'].values)
                exif_data['Model'] = model
            
            if 'EXIF ExifImageWidth' in tags:
                width = int(tags['EXIF ExifImageWidth'].values[0])
                exif_data['ImageWidth'] = width
            
            if 'EXIF ExifImageLength' in tags:
                height = int(tags['EXIF ExifImageLength'].values[0])
                exif_data['ImageLength'] = height
            
            # Vérifier si la focale équivalente 35mm est déjà dans les métadonnées
            if 'EXIF FocalLengthIn35mmFilm' in tags:
                focal_35mm = int(tags['EXIF FocalLengthIn35mmFilm'].values[0])
                exif_data['FocalLengthIn35mmFilm'] = focal_35mm
            
            # Essayer d'extraire des informations sur le capteur pour calculer le facteur
            # Certains appareils stockent la taille du capteur dans les EXIF
            if 'EXIF SensorWidth' in tags and 'EXIF SensorHeight' in tags:
                sensor_width = float(tags['EXIF SensorWidth'].values[0])
                sensor_height = float(tags['EXIF SensorHeight'].values[0])
                exif_data['SensorWidth'] = sensor_width
                exif_data['SensorHeight'] = sensor_height
            
            # Extraire les métadonnées GPS en conservant la précision maximale
            gps_data = {}
            if 'GPS GPSLatitude' in tags:
                lat = tags['GPS GPSLatitude'].values
                lat_ref = tags.get('GPS GPSLatitudeRef', None)
                if lat and len(lat) >= 3:
                    # Conserver directement les valeurs rationnelles DMS pour préserver la précision
                    lat_deg_rat = (int(lat[0].num), int(lat[0].den)) if hasattr(lat[0], 'num') else (int(lat[0]), 1)
                    lat_min_rat = (int(lat[1].num), int(lat[1].den)) if hasattr(lat[1], 'num') else (int(lat[1]), 1)
                    lat_sec_rat = (int(lat[2].num), int(lat[2].den)) if hasattr(lat[2], 'num') else (int(lat[2]), 1)
                    gps_data['GPSLatitude'] = (lat_deg_rat, lat_min_rat, lat_sec_rat)
                    # Extraire la référence correctement
                    if lat_ref:
                        lat_ref_str = str(lat_ref.values).strip() if hasattr(lat_ref, 'values') else str(lat_ref).strip()
                        gps_data['GPSLatitudeRef'] = lat_ref_str[0] if lat_ref_str else 'N'
                    else:
                        gps_data['GPSLatitudeRef'] = 'N'
            
            if 'GPS GPSLongitude' in tags:
                lon = tags['GPS GPSLongitude'].values
                lon_ref = tags.get('GPS GPSLongitudeRef', None)
                if lon and len(lon) >= 3:
                    # Conserver directement les valeurs rationnelles DMS pour préserver la précision
                    lon_deg_rat = (int(lon[0].num), int(lon[0].den)) if hasattr(lon[0], 'num') else (int(lon[0]), 1)
                    lon_min_rat = (int(lon[1].num), int(lon[1].den)) if hasattr(lon[1], 'num') else (int(lon[1]), 1)
                    lon_sec_rat = (int(lon[2].num), int(lon[2].den)) if hasattr(lon[2], 'num') else (int(lon[2]), 1)
                    gps_data['GPSLongitude'] = (lon_deg_rat, lon_min_rat, lon_sec_rat)
                    # Extraire la référence correctement
                    if lon_ref:
                        lon_ref_str = str(lon_ref.values).strip() if hasattr(lon_ref, 'values') else str(lon_ref).strip()
                        gps_data['GPSLongitudeRef'] = lon_ref_str[0] if lon_ref_str else 'E'
                    else:
                        gps_data['GPSLongitudeRef'] = 'E'
            
            if 'GPS GPSAltitude' in tags:
                alt = tags['GPS GPSAltitude'].values[0]
                alt_ref = tags.get('GPS GPSAltitudeRef', None)
                # Conserver directement la valeur rationnelle pour préserver la précision maximale
                if hasattr(alt, 'num') and hasattr(alt, 'den'):
                    # La valeur est déjà en format rationnel, la conserver telle quelle
                    alt_rat = (int(alt.num), int(alt.den))
                else:
                    # Convertir en format rationnel avec précision maximale (multiplier par 10000 pour avoir des centimètres)
                    alt_float = float(alt)
                    alt_rat = (int(alt_float * 10000), 10000)  # Précision au centimètre
                gps_data['GPSAltitude'] = alt_rat
                # Extraire la référence d'altitude (0 = au-dessus du niveau de la mer, 1 = en dessous)
                if alt_ref:
                    alt_ref_val = int(alt_ref.values[0]) if hasattr(alt_ref, 'values') else int(alt_ref)
                    gps_data['GPSAltitudeRef'] = alt_ref_val
                else:
                    gps_data['GPSAltitudeRef'] = 0
            
            if gps_data:
                exif_data['GPS'] = gps_data
                logger.info(f"Métadonnées GPS extraites: {gps_data}")
            
            # Extraire les métadonnées de date/heure
            if 'EXIF DateTimeOriginal' in tags:
                # Format EXIF: "YYYY:MM:DD HH:MM:SS"
                dt_original = str(tags['EXIF DateTimeOriginal'].values).strip()
                exif_data['DateTimeOriginal'] = dt_original
            
            if 'EXIF DateTimeDigitized' in tags:
                dt_digitized = str(tags['EXIF DateTimeDigitized'].values).strip()
                exif_data['DateTimeDigitized'] = dt_digitized
            
            if 'Image DateTime' in tags:
                dt_image = str(tags['Image DateTime'].values).strip()
                exif_data['DateTime'] = dt_image
            
            # Extraire les métadonnées GPS de date/heure (si disponibles)
            if 'GPS GPSDateStamp' in tags:
                gps_date = str(tags['GPS GPSDateStamp'].values).strip()
                exif_data['GPSDateStamp'] = gps_date
            
            if 'GPS GPSTimeStamp' in tags:
                gps_time = tags['GPS GPSTimeStamp'].values
                # GPSTimeStamp est un tableau de 3 valeurs rationnelles [heures, minutes, secondes]
                if gps_time and len(gps_time) >= 3:
                    gps_time_rat = []
                    for i in range(3):
                        if hasattr(gps_time[i], 'num') and hasattr(gps_time[i], 'den'):
                            gps_time_rat.append((int(gps_time[i].num), int(gps_time[i].den)))
                        else:
                            gps_time_rat.append((int(gps_time[i]), 1))
                    exif_data['GPSTimeStamp'] = tuple(gps_time_rat)
            
            if 'EXIF Orientation' in tags:
                exif_data['Orientation'] = int(tags['EXIF Orientation'].values[0])
            
            # Si pas de métadonnées disponibles, ne pas utiliser de valeurs par défaut
            if not exif_data:
                logger.warning("Aucune métadonnée EXIF trouvée, fichier TIFF sans métadonnées")
            
            if exif_data:
                logger.info(f"Métadonnées extraites: {exif_data}")
            else:
                logger.info("Aucune métadonnée EXIF disponible")
            
        except Exception as e:
            logger.warning(f"Impossible d'extraire les métadonnées EXIF: {e}")
            exif_data = {}  # Retourner un dictionnaire vide
        
        return exif_data
    
    def convert_single_file(self, dng_path):
        """
        Convertit un seul fichier DNG en TIFF
        
        Args:
            dng_path (Path): Chemin vers le fichier DNG
            
        Returns:
            bool: True si la conversion a réussi, False sinon
        """
        try:
            # Nom du fichier de sortie
            tiff_filename = dng_path.stem + ".tiff"
            tiff_path = self.output_dir / tiff_filename
            
            logger.info(f"Conversion de: {dng_path.name}")
            
            # Extraire les métadonnées EXIF importantes directement depuis le fichier DNG
            exif_data = self.extract_exif_metadata(dng_path)
            
            # Ouvrir le fichier RAW avec rawpy
            with rawpy.imread(str(dng_path)) as raw:
                
                # Obtenir les données RAW avec ajustements pour photogrammétrie
                # Cela évite la correction de distorsion automatique mais améliore la luminosité
                rgb_array = raw.postprocess(
                    use_camera_wb=True,      # Utiliser le white balance de l'appareil (comme les JPG)
                    half_size=False,         # Pleine résolution
                    no_auto_bright=True,     # Désactiver l'ajustement automatique de luminosité (évite les variations)
                    output_bps=16,           # 16 bits par canal
                    gamma=(2.222, 4.5),     # Correction gamma pour éclaircir l'image
                    bright=self.brightness,  # Facteur de luminosité ajustable (fixe pour toutes les images)
                    highlight_mode=rawpy.HighlightMode.Clip,  # Gestion des hautes lumières
                    use_auto_wb=False       # Utiliser la WB de l'appareil, pas l'AWB automatique
                )
                
                # Appliquer le contraste manuellement (rawpy ne supporte pas le paramètre contrast)
                if self.contrast != 1.0:
                    # Déterminer si on travaille en 16 bits
                    is_16bit = rgb_array.dtype == np.uint16
                    # Conversion en float pour les calculs
                    rgb_array = rgb_array.astype(np.float32)
                    # Calculer le point médian (32768 pour 16 bits, 128 pour 8 bits)
                    midpoint = 32768.0 if is_16bit else 128.0
                    # Appliquer le contraste : (pixel - midpoint) * contrast + midpoint
                    rgb_array = (rgb_array - midpoint) * self.contrast + midpoint
                    # Clamper les valeurs entre 0 et la valeur max
                    max_val = 65535.0 if is_16bit else 255.0
                    rgb_array = np.clip(rgb_array, 0, max_val)
                    # Reconvertir au type d'origine
                    if is_16bit:
                        rgb_array = rgb_array.astype(np.uint16)
                    else:
                        rgb_array = rgb_array.astype(np.uint8)
            
            # S'assurer que les données sont dans le bon format pour PIL
            if self.keep_16bit and rgb_array.dtype == np.uint16:
                # Conserver les 16 bits pour la photogrammétrie
                pass  # Garder les données en uint16
            elif rgb_array.dtype == np.uint16:
                # Convertir de 16 bits à 8 bits en préservant la qualité
                rgb_array = (rgb_array / 256).astype(np.uint8)
            elif rgb_array.dtype != np.uint8:
                # Pour d'autres types, normaliser vers uint8
                rgb_array = np.clip(rgb_array, 0, 65535)  # Clamp les valeurs
                rgb_array = (rgb_array / 256).astype(np.uint8)
            
            # Vérifier les dimensions
            if len(rgb_array.shape) != 3 or rgb_array.shape[2] != 3:
                raise ValueError(f"Format d'image non supporté: {rgb_array.shape}")
            
            # Convertir en PIL Image avec le bon mode
            if self.keep_16bit and rgb_array.dtype == np.uint16:
                image = Image.fromarray(rgb_array, 'RGB')
            else:
                image = Image.fromarray(rgb_array, 'RGB')
            
            # Appliquer la rotation forcée si demandée
            if self.force_orientation:
                width, height = image.size
                is_landscape = width > height
                orientation = exif_data.get('Orientation', 1)
                
                logger.info(f"Image {dng_path.name}: dimensions={width}x{height}, is_landscape={is_landscape}, orientation_EXIF={orientation}")
                
                if self.force_orientation == 'landscape' and not is_landscape:
                    # Forcer paysage depuis portrait : déterminer le sens de rotation selon l'orientation EXIF
                    # Orientation 6 = image doit être tournée de 90° horaire pour être correcte
                    # Orientation 8 = image doit être tournée de 90° anti-horaire pour être correcte
                    if orientation == 6:
                        # Image portrait avec orientation 6 → tourner de +90° pour paysage (inversé)
                        image = image.rotate(90, expand=True)
                        logger.info(f"Image tournée de +90° pour forcer le paysage (orientation EXIF: {orientation})")
                    elif orientation == 8:
                        # Image portrait avec orientation 8 → tourner de -90° pour paysage (inversé)
                        image = image.rotate(-90, expand=True)
                        logger.info(f"Image tournée de -90° pour forcer le paysage (orientation EXIF: {orientation})")
                    else:
                        # Par défaut, essayer l'autre sens
                        image = image.rotate(90, expand=True)
                        logger.info(f"Image tournée de +90° pour forcer le paysage (orientation EXIF: {orientation}, défaut inversé)")
                        
                elif self.force_orientation == 'portrait' and is_landscape:
                    # Forcer portrait depuis paysage : déterminer le sens de rotation selon l'orientation EXIF
                    if orientation == 6:
                        # Pour forcer portrait depuis paysage avec orientation 6 → tourner de -90° (inversé)
                        image = image.rotate(-90, expand=True)
                        logger.info(f"Image tournée de -90° pour forcer le portrait (orientation EXIF: {orientation})")
                    elif orientation == 8:
                        # Pour forcer portrait depuis paysage avec orientation 8 → tourner de +90° (inversé)
                        image = image.rotate(90, expand=True)
                        logger.info(f"Image tournée de +90° pour forcer le portrait (orientation EXIF: {orientation})")
                    else:
                        # Par défaut, essayer l'autre sens
                        image = image.rotate(90, expand=True)
                        logger.info(f"Image tournée de +90° pour forcer le portrait (orientation EXIF: {orientation}, défaut inversé)")
            
            # Sauvegarder en TIFF sans compression (compatible MicMac)
            save_kwargs = {
                'format': 'TIFF'
                # Pas de compression pour compatibilité MicMac
            }
            
            # Ajouter les métadonnées EXIF si disponibles
            if exif_data:
                try:
                    # Créer un dictionnaire EXIF avec piexif
                    exif_dict = {
                        "0th": {},
                        "Exif": {},
                        "GPS": {},
                        "1st": {},
                        "thumbnail": None,
                        "interop": {}
                    }
                    
                    # Ajouter les métadonnées principales
                    if 'Make' in exif_data:
                        exif_dict["0th"][piexif.ImageIFD.Make] = exif_data['Make'].encode('utf-8')
                    if 'Model' in exif_data:
                        exif_dict["0th"][piexif.ImageIFD.Model] = exif_data['Model'].encode('utf-8')
                    
                    # Ajouter les métadonnées EXIF
                    if 'FocalLength' in exif_data:
                        # Convertir en format rationnel piexif
                        focal_length = float(exif_data['FocalLength'])
                        focal_rational = (int(focal_length * 1000), 1000)
                        
                        # Ajouter dans le IFD EXIF (la focale n'existe pas dans le IFD Image)
                        exif_dict["Exif"][piexif.ExifIFD.FocalLength] = focal_rational
                        
                        # Ajouter la focale équivalente 35mm pour MicMac
                        if 'FocalLengthIn35mmFilm' in exif_data:
                            # Utiliser la valeur déjà présente dans les métadonnées
                            focal_35mm = int(exif_data['FocalLengthIn35mmFilm'])
                            exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = focal_35mm
                            logger.info(f"Focale équivalente 35mm trouvée dans les métadonnées: {focal_35mm}mm")
                        else:
                            # Calculer avec le facteur de conversion détecté
                            make = exif_data.get('Make', '')
                            model = exif_data.get('Model', '')
                            sensor_width = exif_data.get('SensorWidth')
                            sensor_height = exif_data.get('SensorHeight')
                            crop_factor = self.get_crop_factor(make, model, sensor_width, sensor_height)
                            
                            if crop_factor is not None:
                                focal_35mm = int(focal_length * crop_factor)
                                exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = focal_35mm
                                logger.info(f"Focale: {focal_length}mm → Équivalent 35mm: {focal_35mm}mm (facteur: {crop_factor}x)")
                            else:
                                logger.warning(f"Impossible de calculer la focale équivalente 35mm pour {make} {model}")
                    
                    if 'FNumber' in exif_data:
                        f_number = float(exif_data['FNumber'])
                        exif_dict["Exif"][piexif.ExifIFD.FNumber] = (int(f_number * 100), 100)
                    
                    if 'ExposureTime' in exif_data:
                        exposure_time = float(exif_data['ExposureTime'])
                        exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (int(exposure_time * 1000000), 1000000)
                    
                    if 'ISOSpeedRatings' in exif_data:
                        exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = int(exif_data['ISOSpeedRatings'])
                    
                    # Ajouter les métadonnées de date/heure
                    if 'DateTimeOriginal' in exif_data:
                        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_data['DateTimeOriginal'].encode('utf-8')
                    
                    if 'DateTimeDigitized' in exif_data:
                        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_data['DateTimeDigitized'].encode('utf-8')
                    
                    if 'DateTime' in exif_data:
                        exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_data['DateTime'].encode('utf-8')
                    
                    # Ajouter les métadonnées GPS de date/heure (si disponibles)
                    if 'GPSDateStamp' in exif_data:
                        exif_dict["GPS"][piexif.GPSIFD.GPSDateStamp] = exif_data['GPSDateStamp'].encode('utf-8')
                    
                    if 'GPSTimeStamp' in exif_data:
                        exif_dict["GPS"][piexif.GPSIFD.GPSTimeStamp] = exif_data['GPSTimeStamp']
                    
                    # Ajouter l'orientation (mettre à 1 si rotation forcée appliquée)
                    if self.force_orientation:
                        # Après rotation physique, l'orientation est normale
                        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                    elif 'Orientation' in exif_data:
                        exif_dict["0th"][piexif.ImageIFD.Orientation] = exif_data['Orientation']
                    
                    # Ajouter les métadonnées GPS en utilisant directement les valeurs rationnelles conservées
                    if 'GPS' in exif_data:
                        gps_data = exif_data['GPS']
                        
                        if 'GPSLatitude' in gps_data:
                            # Utiliser directement les valeurs rationnelles DMS conservées
                            lat_dms = gps_data['GPSLatitude']
                            exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = lat_dms
                            exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = gps_data.get('GPSLatitudeRef', 'N')
                        
                        if 'GPSLongitude' in gps_data:
                            # Utiliser directement les valeurs rationnelles DMS conservées
                            lon_dms = gps_data['GPSLongitude']
                            exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = lon_dms
                            exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = gps_data.get('GPSLongitudeRef', 'E')
                        
                        if 'GPSAltitude' in gps_data:
                            # Utiliser directement la valeur rationnelle conservée
                            alt_rat = gps_data['GPSAltitude']
                            exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = alt_rat
                            exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = gps_data.get('GPSAltitudeRef', 0)
                        
                        # Log pour vérification (calculer les valeurs décimales pour l'affichage)
                        try:
                            if 'GPSLatitude' in gps_data:
                                lat_dms = gps_data['GPSLatitude']
                                lat_deg = lat_dms[0][0] / lat_dms[0][1]
                                lat_min = lat_dms[1][0] / lat_dms[1][1]
                                lat_sec = lat_dms[2][0] / lat_dms[2][1]
                                lat_decimal = lat_deg + lat_min/60.0 + lat_sec/3600.0
                                if gps_data.get('GPSLatitudeRef') == 'S':
                                    lat_decimal = -lat_decimal
                            else:
                                lat_decimal = None
                            
                            if 'GPSLongitude' in gps_data:
                                lon_dms = gps_data['GPSLongitude']
                                lon_deg = lon_dms[0][0] / lon_dms[0][1]
                                lon_min = lon_dms[1][0] / lon_dms[1][1]
                                lon_sec = lon_dms[2][0] / lon_dms[2][1]
                                lon_decimal = lon_deg + lon_min/60.0 + lon_sec/3600.0
                                if gps_data.get('GPSLongitudeRef') == 'W':
                                    lon_decimal = -lon_decimal
                            else:
                                lon_decimal = None
                            
                            if 'GPSAltitude' in gps_data:
                                alt_rat = gps_data['GPSAltitude']
                                alt_decimal = alt_rat[0] / alt_rat[1]
                                if gps_data.get('GPSAltitudeRef') == 1:
                                    alt_decimal = -alt_decimal
                            else:
                                alt_decimal = None
                            
                            logger.info(f"Métadonnées GPS ajoutées (précision maximale conservée): Lat={lat_decimal:.10f}, Lon={lon_decimal:.10f}, Alt={alt_decimal:.6f}m")
                        except Exception as e:
                            logger.info(f"Métadonnées GPS ajoutées (valeurs rationnelles conservées)")
                    
                    # Convertir en bytes EXIF
                    exif_bytes = piexif.dump(exif_dict)
                    save_kwargs['exif'] = exif_bytes
                    logger.info(f"Métadonnées EXIF ajoutées: {list(exif_data.keys())}")
                    
                except Exception as e:
                    logger.warning(f"Erreur lors de l'ajout des métadonnées EXIF: {e}")
            
            image.save(tiff_path, **save_kwargs)
            
            logger.info(f"✓ Converti avec succès: {tiff_filename}")
            return True
            
        except Exception as e:
            import traceback
            logger.error(f"✗ Erreur lors de la conversion de {dng_path.name}: {str(e)}")
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            return False
    
    def convert_all(self):
        """Convertit tous les fichiers DNG trouvés"""
        dng_files = self.find_dng_files()
        
        if not dng_files:
            logger.warning("Aucun fichier DNG trouvé dans le répertoire spécifié")
            return
        
        logger.info(f"Début de la conversion de {len(dng_files)} fichier(s)")
        
        successful_conversions = 0
        failed_conversions = 0
        
        # Conversion avec barre de progression
        for dng_file in tqdm(dng_files, desc="Conversion DNG → TIFF"):
            if self.convert_single_file(dng_file):
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        # Résumé
        logger.info(f"\n=== RÉSUMÉ DE LA CONVERSION ===")
        logger.info(f"Conversions réussies: {successful_conversions}")
        logger.info(f"Conversions échouées: {failed_conversions}")
        logger.info(f"Total traité: {len(dng_files)}")
        
        if successful_conversions > 0:
            logger.info(f"Fichiers TIFF sauvegardés dans: {self.output_dir}")
            # Générer le fichier GPS pour MicMac
            self.generate_gps_file_for_micmac()
    
    def generate_gps_file_for_micmac(self):
        """
        Génère le fichier GpsCoordinatesFromExif.txt au format MicMac
        Format: nom longitude latitude altitude (une ligne par image, séparées par des espaces)
        """
        try:
            # Trouver tous les fichiers TIFF dans le dossier de sortie (utiliser un set pour éviter les doublons sur Windows)
            tiff_files = set(self.output_dir.glob("*.tiff")) | set(self.output_dir.glob("*.TIFF"))
            tiff_files = sorted(list(tiff_files))
            
            if not tiff_files:
                logger.warning("Aucun fichier TIFF trouvé pour générer le fichier GPS")
                return
            
            logger.info(f"\n=== GÉNÉRATION DU FICHIER GPS POUR MICMAC ===")
            logger.info(f"Analyse de {len(tiff_files)} fichier(s) TIFF...")
            
            gps_data_list = []
            processed_files = set()  # Pour éviter les doublons
            
            for tiff_file in tqdm(tiff_files, desc="Extraction GPS"):
                # Vérifier si le fichier n'a pas déjà été traité (normaliser en minuscules pour Windows)
                file_key = tiff_file.name.lower()
                if file_key in processed_files:
                    continue
                processed_files.add(file_key)
                try:
                    # Lire les métadonnées GPS depuis le TIFF
                    with open(tiff_file, 'rb') as f:
                        tags = exifread.process_file(f, details=False)
                    
                    # Extraire les coordonnées GPS
                    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                        # Extraire latitude (DMS → décimales avec précision maximale)
                        lat = tags['GPS GPSLatitude'].values
                        lat_ref = tags.get('GPS GPSLatitudeRef', None)
                        if lat and len(lat) >= 3:
                            lat_deg = float(lat[0].num) / float(lat[0].den) if hasattr(lat[0], 'num') else float(lat[0])
                            lat_min = float(lat[1].num) / float(lat[1].den) if hasattr(lat[1], 'num') else float(lat[1])
                            lat_sec = float(lat[2].num) / float(lat[2].den) if hasattr(lat[2], 'num') else float(lat[2])
                            lat_decimal = lat_deg + lat_min/60.0 + lat_sec/3600.0
                            if lat_ref and 'S' in str(lat_ref.values):
                                lat_decimal = -lat_decimal
                        else:
                            continue
                        
                        # Extraire longitude (DMS → décimales avec précision maximale)
                        lon = tags['GPS GPSLongitude'].values
                        lon_ref = tags.get('GPS GPSLongitudeRef', None)
                        if lon and len(lon) >= 3:
                            lon_deg = float(lon[0].num) / float(lon[0].den) if hasattr(lon[0], 'num') else float(lon[0])
                            lon_min = float(lon[1].num) / float(lon[1].den) if hasattr(lon[1], 'num') else float(lon[1])
                            lon_sec = float(lon[2].num) / float(lon[2].den) if hasattr(lon[2], 'num') else float(lon[2])
                            lon_decimal = lon_deg + lon_min/60.0 + lon_sec/3600.0
                            if lon_ref and 'W' in str(lon_ref.values):
                                lon_decimal = -lon_decimal
                        else:
                            continue
                        
                        # Extraire altitude avec précision maximale
                        altitude = None
                        if 'GPS GPSAltitude' in tags:
                            alt = tags['GPS GPSAltitude'].values[0]
                            alt_ref = tags.get('GPS GPSAltitudeRef', None)
                            if hasattr(alt, 'num') and hasattr(alt, 'den'):
                                altitude = float(alt.num) / float(alt.den)
                            else:
                                altitude = float(alt)
                            if alt_ref and int(alt_ref.values[0]) == 1:  # 1 = below sea level
                                altitude = -altitude
                        
                        if altitude is not None:
                            # Stocker les données : (nom fichier, longitude, latitude, altitude)
                            gps_data_list.append((tiff_file.name, lon_decimal, lat_decimal, altitude))
                        else:
                            logger.warning(f"Altitude manquante pour {tiff_file.name}, ignoré")
                    
                except Exception as e:
                    logger.warning(f"Erreur lors de l'extraction GPS de {tiff_file.name}: {e}")
                    continue
            
            if not gps_data_list:
                logger.warning("Aucune donnée GPS trouvée dans les fichiers TIFF")
                return
            
            # Générer le fichier au format MicMac
            gps_file_path = self.output_dir / "GpsCoordinatesFromExif.txt"
            
            with open(gps_file_path, 'w', encoding='utf-8') as f:
                # Écrire les données avec précision maximale (format: nom longitude latitude altitude)
                for filename, lon, lat, alt in gps_data_list:
                    # Utiliser un format avec suffisamment de décimales pour la précision RTK
                    f.write(f"{filename} {lon:.15f} {lat:.15f} {alt:.6f}\n")
            
            logger.info(f"✓ Fichier GPS généré: {gps_file_path}")
            logger.info(f"  {len(gps_data_list)} image(s) avec coordonnées GPS")
            logger.info(f"  Format: nom longitude latitude altitude")
            logger.info(f"  Précision: Longitude/Latitude (15 décimales), Altitude (6 décimales)")
            
        except Exception as e:
            import traceback
            logger.error(f"Erreur lors de la génération du fichier GPS: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

def interactive_mode():
    """Mode interactif pour faciliter l'utilisation"""
    print("=" * 60)
    print("    CONVERTISSEUR DNG VERS TIFF")
    print("    Pour images de photogrammétrie")
    print("=" * 60)
    print()
    
    # Demander le répertoire d'entrée
    while True:
        input_dir = input("📁 Chemin vers le dossier contenant les images DNG: ").strip()
        if not input_dir:
            print("❌ Veuillez entrer un chemin valide")
            continue
        
        # Supprimer les guillemets si présents
        input_dir = input_dir.strip('"\'')
        
        if not os.path.exists(input_dir):
            print(f"❌ Le répertoire '{input_dir}' n'existe pas")
            continue
        
        # Vérifier qu'il y a des fichiers DNG
        dng_files = list(Path(input_dir).glob("*.DNG")) + list(Path(input_dir).glob("*.dng"))
        if not dng_files:
            print(f"❌ Aucun fichier DNG trouvé dans '{input_dir}'")
            continue
        
        print(f"✅ Trouvé {len(dng_files)} fichier(s) DNG")
        break
    
    # Demander le répertoire de sortie
    print()
    output_dir = input("📁 Répertoire de sortie (Entrée pour utiliser 'TIFF_output'): ").strip()
    if not output_dir:
        output_dir = None
    else:
        output_dir = output_dir.strip('"\'')
    
    # Qualité fixée à 100 pour compatibilité MicMac (pas de compression)
    quality = 100
    
    # Demander si conserver les 16 bits
    print()
    keep_16bit_input = input("🔬 Conserver les 16 bits (recommandé pour photogrammétrie) ? (O/n): ").strip().lower()
    keep_16bit = keep_16bit_input not in ['n', 'non', 'no']
    
    # Demander la luminosité
    print()
    print("💡 Luminosité:")
    print("   • 0.5-0.8: Plus sombre")
    print("   • 1.0: Normal")
    print("   • 1.2-2.0: Plus lumineux")
    while True:
        brightness_input = input("   Facteur (0.5-2.0, défaut 1.5): ").strip()
        if not brightness_input:
            brightness = 1.5  # Valeur par défaut pour éclaircir les images
            break
        
        try:
            brightness = float(brightness_input)
            if 0.5 <= brightness <= 2.0:
                break
            else:
                print("❌ La luminosité doit être entre 0.5 et 2.0")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    # Demander le contraste
    print()
    print("🎨 Contraste:")
    print("   • 0.5-0.8: Plus doux (moins de contraste)")
    print("   • 1.0: Normal")
    print("   • 1.2-2.0: Plus contrasté")
    while True:
        contrast_input = input("   Facteur (0.5-2.0, défaut 1.0): ").strip()
        if not contrast_input:
            contrast = 1.0  # Valeur par défaut normale
            break
        
        try:
            contrast = float(contrast_input)
            if 0.5 <= contrast <= 2.0:
                break
            else:
                print("❌ Le contraste doit être entre 0.5 et 2.0")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    # Demander l'orientation forcée
    print()
    print("🔄 Orientation forcée:")
    print("   • landscape: Forcer toutes les images en paysage")
    print("   • portrait: Forcer toutes les images en portrait")
    print("   • (vide): Conserver l'orientation originale")
    while True:
        orientation_input = input("   Orientation (landscape/portrait/vide): ").strip().lower()
        if not orientation_input:
            force_orientation = None
            break
        elif orientation_input in ['landscape', 'portrait']:
            force_orientation = orientation_input
            break
        else:
            print("❌ Veuillez entrer 'landscape', 'portrait' ou laisser vide")
    
    # Confirmation
    print()
    print("📋 RÉCAPITULATIF:")
    print(f"   • Répertoire source: {input_dir}")
    print(f"   • Répertoire sortie: {output_dir or 'TIFF_output (dans le dossier source)'}")
    print(f"   • Compression: Aucune (compatible MicMac)")
    print(f"   • 16 bits conservés: {'Oui' if keep_16bit else 'Non'}")
    print(f"   • Luminosité: {brightness}")
    print(f"   • Contraste: {contrast}")
    print(f"   • Orientation forcée: {force_orientation or 'Aucune (conservation originale)'}")
    print(f"   • Nombre de fichiers: {len(dng_files)}")
    print()
    
    confirm = input("🚀 Démarrer la conversion ? (o/N): ").strip().lower()
    if confirm not in ['o', 'oui', 'y', 'yes']:
        print("❌ Conversion annulée")
        return
    
    # Lancer la conversion
    print()
    converter = DNGToTIFFConverter(
        input_dir=input_dir,
        output_dir=output_dir,
        quality=int(quality),  # S'assurer que c'est un entier
        keep_16bit=keep_16bit,
        brightness=brightness,
        contrast=contrast,
        force_orientation=force_orientation
    )
    
    converter.convert_all()

def main():
    """Fonction principale avec mode interactif et ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Convertisseur DNG vers TIFF pour images de photogrammétrie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python dng_to_tiff_converter.py                    # Mode interactif
  python dng_to_tiff_converter.py /chemin/vers/images
  python dng_to_tiff_converter.py /chemin/vers/images -o /chemin/sortie
  python dng_to_tiff_converter.py /chemin/vers/images -b 1.5 -c 1.2 --16bit
        """
    )
    
    parser.add_argument(
        'input_dir',
        nargs='?',
        help='Répertoire contenant les fichiers DNG à convertir (optionnel pour mode interactif)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Répertoire de sortie pour les fichiers TIFF (défaut: TIFF_output dans le répertoire d\'entrée)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mode verbeux'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Forcer le mode interactif'
    )
    
    parser.add_argument(
        '--16bit',
        action='store_true',
        dest='keep_16bit',
        help='Conserver les 16 bits (recommandé pour photogrammétrie)'
    )
    
    parser.add_argument(
        '-b', '--brightness',
        type=float,
        default=1.5,
        help='Facteur de luminosité (0.5-2.0, défaut 1.5)'
    )
    
    parser.add_argument(
        '-c', '--contrast',
        type=float,
        default=1.0,
        help='Facteur de contraste (0.5-2.0, défaut 1.0)'
    )
    
    parser.add_argument(
        '--force-orientation',
        type=str,
        choices=['landscape', 'portrait'],
        default=None,
        help='Forcer toutes les images en paysage ou portrait (ignore l\'orientation EXIF)'
    )
    
    args = parser.parse_args()
    
    # Ajuster le niveau de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Si aucun argument n'est fourni ou si mode interactif demandé
    if not args.input_dir or args.interactive:
        interactive_mode()
        return
    
    # Mode ligne de commande
    # Vérifier que le répertoire d'entrée existe
    if not os.path.exists(args.input_dir):
        logger.error(f"Le répertoire '{args.input_dir}' n'existe pas")
        sys.exit(1)
    
    # Créer et lancer le convertisseur
    converter = DNGToTIFFConverter(
        input_dir=args.input_dir,
        output_dir=args.output,
        quality=100,  # Qualité fixée pour compatibilité MicMac
        keep_16bit=args.keep_16bit,
        brightness=args.brightness,
        contrast=args.contrast,
        force_orientation=args.force_orientation
    )
    
    converter.convert_all()

if __name__ == "__main__":
    main()


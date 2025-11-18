# Raw Exporter

Convertisseur de fichiers RAW (ORF, DNG) vers TIFF pour images de photogrammétrie.

## Fonctionnalités

- Conversion ORF → TIFF sans correction de distorsion
- Conversion DNG → TIFF sans correction de distorsion
- Préservation des métadonnées EXIF (GPS, focale, ouverture, ISO, etc.)
- Ajustement de la luminosité et du contraste
- Support 16 bits pour la photogrammétrie
- Compatible MicMac (TIFF non compressé)
- Mode interactif et ligne de commande

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Convertisseur ORF

**Mode interactif :**
```bash
python orf_to_tiff_converter.py
```

**Ligne de commande :**
```bash
python orf_to_tiff_converter.py /chemin/vers/images -b 1.5 -c 1.0 --16bit
```

### Convertisseur DNG

**Mode interactif :**
```bash
python dng_to_tiff_converter.py
```

**Ligne de commande :**
```bash
python dng_to_tiff_converter.py /chemin/vers/images -b 1.5 -c 1.0 --16bit
```

## Options

- `-o, --output` : Répertoire de sortie (défaut: TIFF_output)
- `-b, --brightness` : Facteur de luminosité (0.5-2.0, défaut: 1.5)
- `-c, --contrast` : Facteur de contraste (0.5-2.0, défaut: 1.0)
- `--16bit` : Conserver les 16 bits (recommandé pour photogrammétrie)
- `-v, --verbose` : Mode verbeux
- `-i, --interactive` : Forcer le mode interactif

## Métadonnées préservées

- **GPS** : Latitude, Longitude, Altitude
- **Caméra** : Focale, Ouverture, Vitesse, ISO
- **Date/Heure** : DateTimeOriginal, DateTimeDigitized
- **Appareil** : Marque, Modèle

## Fichiers générés

Les fichiers TIFF sont sauvegardés dans le répertoire `TIFF_output` (ou le répertoire spécifié avec `-o`).

## Compatibilité

- Compatible avec MicMac (photogrammétrie)
- Supporte les appareils Olympus, OM Digital, Panasonic, Canon, Nikon, Sony, Fujifilm, Leica, etc.


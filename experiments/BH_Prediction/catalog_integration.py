"""
Catalog Integration for P3 Black Hole Predictions

Handles GAIA DR3 data and cross-matching with known black hole catalogs
including SIMBAD, X-ray binaries (XRB), and Active Galactic Nuclei (AGN).
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class CatalogSource:
    """Represents a known astronomical source from catalogs."""
    name: str
    ra: float          # Right Ascension (degrees)
    dec: float         # Declination (degrees)
    source_type: str   # 'BH', 'XRB', 'AGN', etc.
    catalog: str       # Source catalog name
    confidence: float  # Confidence level (0-1)
    
    def __post_init__(self):
        """Validate coordinates."""
        if not (0 <= self.ra < 360):
            raise ValueError(f"RA must be in [0, 360), got {self.ra}")
        if not (-90 <= self.dec <= 90):
            raise ValueError(f"Dec must be in [-90, 90], got {self.dec}")

class CatalogIntegrator:
    """
    Integrates multiple astronomical catalogs for black hole cross-matching.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize catalog integrator.
        
        Args:
            data_dir: Directory containing catalog data files
        """
        self.data_dir = data_dir or Path("data")
        self.sources: List[CatalogSource] = []
        self.loaded_catalogs: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"Catalog Integrator initialized with data_dir: {self.data_dir}")
    
    def load_known_black_holes(self) -> List[CatalogSource]:
        """
        Load known black hole candidates from multiple sources.
        
        For demo purposes, we'll start with a curated list of well-known
        black holes and X-ray binaries.
        
        Returns:
            List of known black hole sources
        """
        logger.info("Loading known black hole catalog...")
        
        # Known black holes with accurate coordinates
        known_bh = [
            # Stellar-mass black holes
            CatalogSource("Cygnus X-1", 299.59033, 35.20167, "BH", "SIMBAD", 0.95),
            CatalogSource("LMC X-1", 84.912, -69.742, "BH", "SIMBAD", 0.90),
            CatalogSource("LMC X-3", 84.734, -64.083, "BH", "SIMBAD", 0.85),
            CatalogSource("GRS 1915+105", 288.798, 10.946, "BH", "SIMBAD", 0.95),
            CatalogSource("V404 Cygni", 306.016, 33.867, "BH", "SIMBAD", 0.90),
            
            # X-ray binaries (potential BHs)
            CatalogSource("GX 339-4", 255.706, -48.790, "XRB", "SIMBAD", 0.80),
            CatalogSource("GRO J1655-40", 253.500, -39.844, "XRB", "SIMBAD", 0.85),
            CatalogSource("XTE J1550-564", 237.742, -56.477, "XRB", "SIMBAD", 0.80),
            CatalogSource("H 1743-322", 266.567, -32.235, "XRB", "SIMBAD", 0.75),
            
            # Supermassive black holes (AGN)
            CatalogSource("Sagittarius A*", 266.417, -29.008, "SMBH", "SIMBAD", 0.99),
            CatalogSource("M87*", 187.706, 12.391, "SMBH", "SIMBAD", 0.99),
            CatalogSource("3C 273", 187.278, 2.052, "AGN", "SIMBAD", 0.95),
            CatalogSource("3C 279", 194.047, -5.789, "AGN", "SIMBAD", 0.90),
        ]
        
        self.sources.extend(known_bh)
        logger.info(f"Loaded {len(known_bh)} known black hole sources")
        
        return known_bh
    
    def angular_separation(self, ra1: float, dec1: float, 
                          ra2: float, dec2: float) -> float:
        """
        Compute angular separation between two sky positions.
        
        Args:
            ra1, dec1: First position (degrees)
            ra2, dec2: Second position (degrees)
            
        Returns:
            Angular separation in degrees
        """
        # Convert to radians
        ra1_rad, dec1_rad = np.radians(ra1), np.radians(dec1)
        ra2_rad, dec2_rad = np.radians(ra2), np.radians(dec2)
        
        # Haversine formula for angular separation
        dra = ra2_rad - ra1_rad
        ddec = dec2_rad - dec1_rad
        
        a = (np.sin(ddec/2)**2 + 
             np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(dra/2)**2)
        
        separation = 2 * np.arcsin(np.sqrt(a))
        return np.degrees(separation)
    
    def cross_match_sky_patch(self, ra: float, dec: float, 
                             radius: float = 1.5) -> List[CatalogSource]:
        """
        Find catalog sources within a sky patch.
        
        Args:
            ra: Patch center RA (degrees)
            dec: Patch center Dec (degrees)
            radius: Patch radius (degrees)
            
        Returns:
            List of sources within the patch
        """
        matches = []
        
        for source in self.sources:
            separation = self.angular_separation(ra, dec, source.ra, source.dec)
            
            if separation <= radius:
                matches.append(source)
                logger.debug(f"Match: {source.name} at {separation:.2f}° from patch center")
        
        return matches
    
    def compute_sky_density(self, ra_range: Tuple[float, float],
                           dec_range: Tuple[float, float]) -> float:
        """
        Compute source density in a sky region.
        
        Args:
            ra_range: (min_ra, max_ra) in degrees
            dec_range: (min_dec, max_dec) in degrees
            
        Returns:
            Source density (sources per square degree)
        """
        ra_min, ra_max = ra_range
        dec_min, dec_max = dec_range
        
        # Count sources in region
        count = 0
        for source in self.sources:
            if (ra_min <= source.ra <= ra_max and 
                dec_min <= source.dec <= dec_max):
                count += 1
        
        # Compute area (approximate for small regions)
        area = (ra_max - ra_min) * (dec_max - dec_min)
        area *= np.cos(np.radians((dec_min + dec_max) / 2))  # Cos(dec) correction
        
        density = count / area if area > 0 else 0
        logger.debug(f"Sky density: {count} sources in {area:.2f} deg² = {density:.4f} sources/deg²")
        
        return density
    
    def generate_random_sky_patches(self, n_patches: int, patch_radius: float,
                                   ra_range: Tuple[float, float] = (0, 360),
                                   dec_range: Tuple[float, float] = (-90, 90)) -> List[Tuple[float, float]]:
        """
        Generate random sky patch centers for baseline comparison.
        
        Args:
            n_patches: Number of random patches
            patch_radius: Patch radius (degrees)
            ra_range: RA range for random sampling
            dec_range: Dec range for random sampling
            
        Returns:
            List of (RA, Dec) tuples for random patch centers
        """
        ra_min, ra_max = ra_range
        dec_min, dec_max = dec_range
        
        # Generate uniform random RA
        ras = np.random.uniform(ra_min, ra_max, n_patches)
        
        # Generate random Dec with proper sky distribution (sin(dec) weighting)
        dec_min_rad, dec_max_rad = np.radians(dec_min), np.radians(dec_max)
        sin_dec_min, sin_dec_max = np.sin(dec_min_rad), np.sin(dec_max_rad)
        
        sin_decs = np.random.uniform(sin_dec_min, sin_dec_max, n_patches)
        decs = np.degrees(np.arcsin(sin_decs))
        
        random_patches = list(zip(ras, decs))
        logger.info(f"Generated {n_patches} random sky patches")
        
        return random_patches
    
    def save_catalog_cache(self, filepath: Path):
        """Save loaded catalog data to cache file."""
        cache_data = {
            'sources': [
                {
                    'name': s.name, 'ra': s.ra, 'dec': s.dec,
                    'source_type': s.source_type, 'catalog': s.catalog,
                    'confidence': s.confidence
                }
                for s in self.sources
            ],
            'timestamp': pd.Timestamp.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Saved catalog cache to {filepath}")
    
    def load_catalog_cache(self, filepath: Path):
        """Load catalog data from cache file."""
        with open(filepath, 'r') as f:
            cache_data = json.load(f)
        
        self.sources = [
            CatalogSource(**source_data)
            for source_data in cache_data['sources']
        ]
        
        logger.info(f"Loaded {len(self.sources)} sources from cache: {filepath}")


def demo_catalog_integration():
    """Demonstrate catalog integration functionality."""
    print("📡 Catalog Integration Demo")
    print("=" * 40)
    
    # Initialize integrator
    integrator = CatalogIntegrator()
    
    # Load known black holes
    bh_sources = integrator.load_known_black_holes()
    
    print(f"\n📊 Loaded {len(bh_sources)} known sources:")
    print("Type | Count")
    print("-" * 15)
    types = {}
    for source in bh_sources:
        types[source.source_type] = types.get(source.source_type, 0) + 1
    
    for source_type, count in types.items():
        print(f"{source_type:4s} | {count:5d}")
    
    # Test cross-matching with Cygnus region
    print(f"\n🎯 Cross-matching Cygnus region (RA=300°, Dec=40°, r=5°):")
    matches = integrator.cross_match_sky_patch(300, 40, 5.0)
    
    print("Name           | RA     | Dec    | Type | Separation")
    print("-" * 55)
    for match in matches:
        sep = integrator.angular_separation(300, 40, match.ra, match.dec)
        print(f"{match.name:14s} | {match.ra:6.1f} | {match.dec:6.1f} | {match.source_type:4s} | {sep:6.2f}°")
    
    # Test sky density computation
    density = integrator.compute_sky_density((280, 320), (30, 50))
    print(f"\n📈 Sky density in Cygnus region: {density:.4f} sources/deg²")
    
    # Test random patch generation
    random_patches = integrator.generate_random_sky_patches(5, 1.5)
    print(f"\n🎲 Sample random patches:")
    print("RA     | Dec")
    print("-" * 15)
    for ra, dec in random_patches:
        print(f"{ra:6.1f} | {dec:6.1f}")
    
    print(f"\n✅ Demo complete!")
    return integrator


if __name__ == "__main__":
    demo_catalog_integration()
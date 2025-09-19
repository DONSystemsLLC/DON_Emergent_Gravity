"""
Pre-Registration System for P3 Black Hole Predictions

Implements SHA256 hashing system for pre-registering DON field predictions
before blind testing against known catalogs.
"""

import hashlib
import json
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .sky_scanner import SkyPatch, DONFieldSkyScanner

logger = logging.getLogger(__name__)

@dataclass
class PreRegistration:
    """
    Represents a pre-registered prediction set.
    """
    timestamp: str
    protocol_version: str
    engine_version: str
    random_seed: int
    top_n: int
    patch_radius: float
    targets: List[Dict[str, float]]  # List of {ra, dec, score, rank}
    scan_parameters: Dict[str, Any]
    hash_sha256: str
    
    def __post_init__(self):
        """Validate pre-registration data."""
        if len(self.targets) != self.top_n:
            raise ValueError(f"Expected {self.top_n} targets, got {len(self.targets)}")
        
        # Verify hash matches content
        computed_hash = PreRegistrationSystem.compute_target_hash(self.targets, self.random_seed)
        if self.hash_sha256 != computed_hash:
            logger.warning(f"Hash mismatch: stored={self.hash_sha256}, computed={computed_hash}")

class PreRegistrationSystem:
    """
    Handles pre-registration of DON field predictions with cryptographic hashing
    to prevent cherry-picking and ensure blind testing integrity.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize pre-registration system.
        
        Args:
            output_dir: Directory for saving pre-registration files
        """
        self.output_dir = output_dir or Path("proofs/P3_preregistration")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pre-registration system initialized: {self.output_dir}")
    
    @staticmethod
    def compute_target_hash(targets: List[Dict[str, float]], seed: int) -> str:
        """
        Compute SHA256 hash of target list.
        
        Args:
            targets: List of target dictionaries
            seed: Random seed used in generation
            
        Returns:
            SHA256 hash as hex string
        """
        # Create deterministic string representation
        hash_data = {
            'seed': seed,
            'targets': sorted(targets, key=lambda t: t['rank'])  # Sort by rank for determinism
        }
        
        # Convert to JSON string with sorted keys
        json_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA256
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def freeze_engine_version(self, scanner: DONFieldSkyScanner) -> str:
        """
        Create version hash of the DON field engine for reproducibility.
        
        Args:
            scanner: DONFieldSkyScanner instance
            
        Returns:
            Engine version hash
        """
        # For now, use a simple version string
        # In production, this would hash the actual code
        version_data = {
            'class': scanner.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'python_version': f"{np.__version__}",  # Use numpy version as proxy
        }
        
        version_str = json.dumps(version_data, sort_keys=True)
        engine_hash = hashlib.sha256(version_str.encode('utf-8')).hexdigest()[:16]
        
        logger.info(f"Engine version frozen: {engine_hash}")
        return f"DON-v1.0-{engine_hash}"
    
    def create_preregistration(self, 
                              scanner: DONFieldSkyScanner,
                              top_n: int = 50,
                              random_seed: Optional[int] = None) -> PreRegistration:
        """
        Create pre-registration of DON field predictions.
        
        Args:
            scanner: Configured DONFieldSkyScanner
            top_n: Number of top targets to pre-register
            random_seed: Random seed for reproducibility
            
        Returns:
            PreRegistration object with hashed predictions
        """
        if random_seed is None:
            random_seed = np.random.randint(0, 2**31)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        logger.info(f"Creating pre-registration: top_n={top_n}, seed={random_seed}")
        
        # Get engine version
        engine_version = self.freeze_engine_version(scanner)
        
        # Run sky scan to get targets
        if not scanner.scan_complete:
            logger.info("Running full sky scan for pre-registration...")
            scanner.run_full_scan()
        
        # Get top N targets
        top_patches = scanner.get_top_n_targets(top_n)
        
        # Convert to serializable format
        targets = []
        for patch in top_patches:
            targets.append({
                'ra': float(patch.ra),
                'dec': float(patch.dec),
                'field_score': float(patch.field_score),
                'rank': int(patch.rank)
            })
        
        # Compute hash
        target_hash = self.compute_target_hash(targets, random_seed)
        
        # Create pre-registration
        prereg = PreRegistration(
            timestamp=datetime.now().isoformat(),
            protocol_version="P3-v1.0",
            engine_version=engine_version,
            random_seed=random_seed,
            top_n=top_n,
            patch_radius=1.5,  # Default patch radius
            targets=targets,
            scan_parameters={
                'total_patches': len(scanner.sky_patches),
                'scan_complete': scanner.scan_complete
            },
            hash_sha256=target_hash
        )
        
        logger.info(f"Pre-registration created: {target_hash[:16]}...")
        return prereg
    
    def save_preregistration(self, prereg: PreRegistration, 
                           filename: Optional[str] = None) -> Path:
        """
        Save pre-registration to file.
        
        Args:
            prereg: PreRegistration object
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prereg_{timestamp}_{prereg.hash_sha256[:8]}.json"
        
        filepath = self.output_dir / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(asdict(prereg), f, indent=2)
        
        logger.info(f"Pre-registration saved: {filepath}")
        
        # Also save a summary file
        summary_file = self.output_dir / f"PREREGISTRATION_SUMMARY.md"
        self._save_summary(prereg, summary_file)
        
        return filepath
    
    def _save_summary(self, prereg: PreRegistration, filepath: Path):
        """Save human-readable summary of pre-registration."""
        summary = f"""# P3 Black Hole Prediction Pre-Registration

## Registration Details
- **Timestamp**: {prereg.timestamp}
- **Protocol Version**: {prereg.protocol_version}
- **Engine Version**: {prereg.engine_version}
- **Random Seed**: {prereg.random_seed}
- **SHA256 Hash**: `{prereg.hash_sha256}`

## Prediction Summary
- **Top N Targets**: {prereg.top_n}
- **Patch Radius**: {prereg.patch_radius}°
- **Total Patches Scanned**: {prereg.scan_parameters.get('total_patches', 'Unknown')}

## Top 10 Predicted Black Hole Locations
| Rank | RA (°) | Dec (°) | DON Score |
|------|--------|---------|-----------|
"""
        
        # Add top 10 targets to summary
        for target in prereg.targets[:10]:
            summary += f"| {target['rank']:4d} | {target['ra']:6.1f} | {target['dec']:6.1f} | {target['field_score']:9.4f} |\n"
        
        summary += f"""
## Verification
To verify this pre-registration:
1. Use the same engine version: `{prereg.engine_version}`
2. Set random seed: `{prereg.random_seed}`
3. Compute SHA256 of target list
4. Verify hash matches: `{prereg.hash_sha256}`

## Next Steps
1. **DO NOT** look up these coordinates in any catalog
2. Run blind test using `P3_predictions.blind_test`
3. Compare results against this pre-registered list

---
*This pre-registration prevents cherry-picking and ensures scientific integrity*
"""
        
        with open(filepath, 'w') as f:
            f.write(summary)
        
        logger.info(f"Pre-registration summary saved: {filepath}")
    
    def load_preregistration(self, filepath: Path) -> PreRegistration:
        """
        Load pre-registration from file.
        
        Args:
            filepath: Path to pre-registration file
            
        Returns:
            PreRegistration object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        prereg = PreRegistration(**data)
        logger.info(f"Loaded pre-registration: {prereg.hash_sha256[:16]}...")
        
        return prereg
    
    def verify_preregistration(self, prereg: PreRegistration) -> bool:
        """
        Verify integrity of pre-registration.
        
        Args:
            prereg: PreRegistration to verify
            
        Returns:
            True if verification passes
        """
        # Recompute hash
        computed_hash = self.compute_target_hash(prereg.targets, prereg.random_seed)
        
        if computed_hash == prereg.hash_sha256:
            logger.info("✅ Pre-registration verification PASSED")
            return True
        else:
            logger.error(f"❌ Pre-registration verification FAILED")
            logger.error(f"   Stored hash: {prereg.hash_sha256}")
            logger.error(f"   Computed hash: {computed_hash}")
            return False


def demo_preregistration():
    """Demonstrate pre-registration system."""
    print("🔒 Pre-Registration System Demo")
    print("=" * 40)
    
    # Initialize systems
    from .sky_scanner import DONFieldSkyScanner
    
    prereg_system = PreRegistrationSystem()
    scanner = DONFieldSkyScanner()
    
    # Run quick scan for demo
    print("🔭 Running DON field sky scan...")
    scanner.run_full_scan(
        ra_min=280, ra_max=320,
        dec_min=30, dec_max=50,
        ra_step=2.0, dec_step=2.0
    )
    
    # Create pre-registration
    print("🔒 Creating pre-registration...")
    prereg = prereg_system.create_preregistration(
        scanner, top_n=10, random_seed=42
    )
    
    print(f"✅ Pre-registration created:")
    print(f"   Engine Version: {prereg.engine_version}")
    print(f"   Random Seed: {prereg.random_seed}")
    print(f"   SHA256 Hash: {prereg.hash_sha256}")
    print(f"   Top {len(prereg.targets)} targets locked")
    
    # Verify integrity
    print("\n🔍 Verifying pre-registration integrity...")
    is_valid = prereg_system.verify_preregistration(prereg)
    
    # Save pre-registration
    print("\n💾 Saving pre-registration...")
    saved_path = prereg_system.save_preregistration(prereg)
    print(f"   Saved to: {saved_path}")
    
    # Show top 5 targets (without revealing what they are!)
    print(f"\n🎯 Top 5 Pre-Registered Targets:")
    print("Rank | RA (°) | Dec (°) | DON Score")
    print("-" * 35)
    for target in prereg.targets[:5]:
        print(f"{target['rank']:4d} | {target['ra']:6.1f} | {target['dec']:6.1f} | {target['field_score']:9.4f}")
    
    print(f"\n🔒 PREDICTIONS ARE NOW LOCKED!")
    print(f"Hash: {prereg.hash_sha256[:16]}...")
    print(f"Ready for blind testing phase.")
    
    return prereg_system, prereg


if __name__ == "__main__":
    demo_preregistration()
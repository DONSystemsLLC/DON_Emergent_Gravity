"""
P3 Black Hole Predictions Test Runner

Tests all components of the prediction system.
"""

import os
import pytest

from .sky_scanner import DONFieldSkyScanner
from .catalog_integration import CatalogIntegrator
from .preregistration import PreRegistrationSystem

pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_P3_INTEGRATION') != '1',
    reason='P3 integration pipeline is computationally heavy; set RUN_P3_INTEGRATION=1 to run',
)

def test_full_prediction_pipeline():
    """Test the complete P3 prediction pipeline."""
    print("🚀 P3 BLACK HOLE PREDICTION PIPELINE TEST")
    print("=" * 50)
    
    # 1. Initialize all systems
    print("\n1️⃣ Initializing systems...")
    scanner = DONFieldSkyScanner()
    integrator = CatalogIntegrator()
    prereg_system = PreRegistrationSystem()
    
    # 2. Load known catalog sources
    print("\n2️⃣ Loading black hole catalog...")
    bh_sources = integrator.load_known_black_holes()
    print(f"   Loaded {len(bh_sources)} known sources")
    
    # 3. Run DON field sky scan
    print("\n3️⃣ Running DON field sky scan...")
    patches = scanner.run_full_scan(
        ra_min=280, ra_max=320,  # Cygnus region
        dec_min=30, dec_max=50,
        ra_step=3.0, dec_step=3.0,
        patch_radius=1.5
    )
    print(f"   Scanned {len(patches)} sky patches")
    
    # 4. Create pre-registration
    print("\n4️⃣ Creating pre-registration...")
    prereg = prereg_system.create_preregistration(
        scanner, top_n=20, random_seed=12345
    )
    print(f"   Hash: {prereg.hash_sha256[:16]}...")
    print(f"   Top {len(prereg.targets)} targets locked")
    
    # 5. Save pre-registration
    print("\n5️⃣ Saving pre-registration...")
    saved_path = prereg_system.save_preregistration(prereg)
    
    # 6. NOW we can test against catalog (in real scenario, this would be blind)
    print("\n6️⃣ Testing against catalog (SIMULATION OF BLIND TEST)...")
    hits = 0
    hit_details = []
    
    for i, target in enumerate(prereg.targets[:10]):  # Test top 10
        matches = integrator.cross_match_sky_patch(
            target['ra'], target['dec'], prereg.patch_radius
        )
        
        if matches:
            hits += 1
            for match in matches:
                separation = integrator.angular_separation(
                    target['ra'], target['dec'], match.ra, match.dec
                )
                hit_details.append({
                    'rank': target['rank'],
                    'target_ra': target['ra'],
                    'target_dec': target['dec'],
                    'match_name': match.name,
                    'match_type': match.source_type,
                    'separation': separation
                })
    
    # 7. Show results
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Top 10 patches tested")
    print(f"   Hits: {hits}")
    print(f"   Hit rate: {hits/10:.1%}")
    
    if hit_details:
        print(f"\n📊 HIT DETAILS:")
        print("Rank | Target RA/Dec | Match Name    | Type | Sep(°)")
        print("-" * 55)
        for hit in hit_details:
            print(f"{hit['rank']:4d} | {hit['target_ra']:6.1f},{hit['target_dec']:5.1f} | "
                  f"{hit['match_name']:12s} | {hit['match_type']:4s} | {hit['separation']:5.2f}")
    
    # 8. Compute enrichment factor (simplified)
    baseline_density = integrator.compute_sky_density((0, 360), (-90, 90))
    test_area = 10 * (prereg.patch_radius * 2) ** 2  # Approximate area of 10 patches
    expected_hits = baseline_density * test_area
    
    if expected_hits > 0:
        enrichment = hits / expected_hits
        print(f"\n📈 ENRICHMENT ANALYSIS:")
        print(f"   Baseline density: {baseline_density:.6f} sources/deg²")
        print(f"   Expected hits: {expected_hits:.2f}")
        print(f"   Actual hits: {hits}")
        print(f"   Enrichment factor: {enrichment:.2f}")
        
        if enrichment >= 2.0:
            print(f"   🎉 SUCCESS: E = {enrichment:.2f} ≥ 2.0")
        else:
            print(f"   📉 Below threshold: E = {enrichment:.2f} < 2.0")
    
    print(f"\n✅ Pipeline test complete!")
    print(f"Pre-registration saved: {saved_path}")
    
    return {
        'hits': hits,
        'total_tested': 10,
        'hit_rate': hits/10,
        'enrichment': enrichment if expected_hits > 0 else 0,
        'prereg_hash': prereg.hash_sha256,
        'hit_details': hit_details
    }

if __name__ == "__main__":
    results = test_full_prediction_pipeline()
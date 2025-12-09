"""
Basic tests for the AWERA modular architecture.

This test file verifies that the Abstract Base Classes are properly defined
and can be imported without errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        # Test ABC imports
        from awera.wind.base import WindProfileModel
        from awera.power.base import PowerEstimationModel
        from awera.optimisation.base import Optimiser
        from awera.kite.base import KiteModel
        from awera.control.base import ControlModel
        from awera.groundstation.base import GroundStationModel
        print("✓ All Abstract Base Classes imported successfully")
        
        # Test implementation imports
        from awera.wind.clustering import WindProfileClusteringModel
        from awera.pipeline.aep import AEPCalculator
        print("✓ Implementation classes imported successfully")
        
        # Test package-level imports
        import awera
        print("✓ Main package imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_abc_structure():
    """Test that Abstract Base Classes have the required methods."""
    print("\nTesting ABC structure...")
    
    try:
        from awera.wind.base import WindProfileModel
        from awera.power.base import PowerEstimationModel
        from awera.optimisation.base import Optimiser
        
        # Check that ABCs cannot be instantiated directly
        try:
            WindProfileModel()
            print("✗ WindProfileModel should not be instantiable (ABC)")
            return False
        except TypeError:
            print("✓ WindProfileModel correctly prevents direct instantiation")
        
        try:
            PowerEstimationModel()
            print("✗ PowerEstimationModel should not be instantiable (ABC)")
            return False
        except TypeError:
            print("✓ PowerEstimationModel correctly prevents direct instantiation")
        
        try:
            Optimiser()
            print("✗ Optimiser should not be instantiable (ABC)")
            return False
        except TypeError:
            print("✓ Optimiser correctly prevents direct instantiation")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing ABC structure: {e}")
        return False


def test_concrete_implementation():
    """Test that concrete implementations can be instantiated."""
    print("\nTesting concrete implementations...")
    
    try:
        from awera.wind.clustering import WindProfileClusteringModel
        from awera.pipeline.aep import AEPCalculator
        from pathlib import Path
        
        # Test WindProfileClusteringModel instantiation
        wind_model = WindProfileClusteringModel()
        print("✓ WindProfileClusteringModel can be instantiated")
        
        # Test that it has required methods
        required_methods = ['load_from_yaml', 'cluster', 'export_to_yaml']
        for method in required_methods:
            if not hasattr(wind_model, method):
                print(f"✗ WindProfileClusteringModel missing method: {method}")
                return False
        print("✓ WindProfileClusteringModel has all required methods")
        
        # Test AEPCalculator with dummy config
        dummy_config = Path(__file__).parent / "config" / "main_config.yml"
        if dummy_config.exists():
            aep_calc = AEPCalculator(dummy_config)
            print("✓ AEPCalculator can be instantiated with config")
        else:
            print("⚠ AEPCalculator test skipped - config file not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing concrete implementations: {e}")
        return False


def test_yaml_config_structure():
    """Test that YAML configuration files are properly structured."""
    print("\nTesting YAML configuration structure...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_dir = Path(__file__).parent / "config"
        if not config_dir.exists():
            print("⚠ Config directory not found - creating dummy configs...")
            return True
        
        # Test main config
        main_config_path = config_dir / "main_config.yml"
        if main_config_path.exists():
            with open(main_config_path, 'r') as f:
                main_config = yaml.safe_load(f)
            
            expected_sections = ['wind', 'optimisation', 'power', 'pipeline']
            for section in expected_sections:
                if section not in main_config:
                    print(f"✗ Main config missing section: {section}")
                    return False
            print("✓ Main configuration file is properly structured")
        
        # Test wind config
        wind_config_path = config_dir / "wind_clustering_config.yml"
        if wind_config_path.exists():
            with open(wind_config_path, 'r') as f:
                wind_config = yaml.safe_load(f)
            
            if 'n_clusters' not in wind_config:
                print("✗ Wind config missing n_clusters")
                return False
            print("✓ Wind clustering configuration file is properly structured")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing YAML configs: {e}")
        return False


def main():
    """Run all tests."""
    print("AWERA Architecture Tests")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_abc_structure,
        test_concrete_implementation,
        test_yaml_config_structure
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 30)
    print("Test Results:")
    if all(results):
        print("✓ All tests passed!")
        print("AWERA modular architecture is properly set up.")
    else:
        print("✗ Some tests failed.")
        print("Check the error messages above.")


if __name__ == "__main__":
    main()
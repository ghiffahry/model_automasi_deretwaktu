"""
Test script untuk memverifikasi backend setup
"""
import sys

def test_imports():
    """Test semua import module"""
    print("Testing imports...")
    errors = []
    
    modules = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'statsmodels',
        'torch',
        'matplotlib',
        'sklearn'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            errors.append(module)
    
    return len(errors) == 0, errors

def test_config():
    """Test config file"""
    print("\nTesting config...")
    try:
        from config import Config
        print(f"  ✓ Config imported")
        
        # Check VALIDATION
        if hasattr(Config, 'VALIDATION'):
            print(f"  ✓ Config.VALIDATION exists")
            print(f"    MIN_DATA_POINTS: {Config.VALIDATION['MIN_DATA_POINTS']}")
        else:
            print(f"  ✗ Config.VALIDATION missing")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False

def test_api():
    """Test API import"""
    print("\nTesting API...")
    try:
        from api import app
        print(f"  ✓ API imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ API import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline():
    """Test pipeline modules"""
    print("\nTesting pipeline modules...")
    try:
        from pipeline import ForecastingPipeline
        print(f"  ✓ Pipeline imported")
        
        from exploratory import TimeSeriesExplorer
        print(f"  ✓ Explorer imported")
        
        from data_loader import TimeSeriesDataLoader
        print(f"  ✓ DataLoader imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("SURADATA BACKEND TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    success, errors = test_imports()
    results.append(('Dependencies', success))
    if not success:
        print(f"\nMissing modules: {', '.join(errors)}")
        print("Run: pip install -r requirements.txt")
    
    # Test 2: Config
    success = test_config()
    results.append(('Config', success))
    
    # Test 3: API
    success = test_api()
    results.append(('API', success))
    
    # Test 4: Pipeline
    success = test_pipeline()
    results.append(('Pipeline', success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Backend is ready.")
        print("\nRun the server with:")
        print("  uvicorn api:app --reload --port 8000")
        return 0
    else:
        print("\n✗ Some tests failed. Fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
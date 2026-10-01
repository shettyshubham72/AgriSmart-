"""
app_standalone.py  -  AgriSmart entry point (PyCharm runs this)
"""
import sys, os, importlib.util

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, 'models'))

# Step 1: alias idbn_core as models.idbn so pickle can find the class
_idbn_spec = importlib.util.spec_from_file_location(
    "models.idbn",
    os.path.join(_base, "models", "idbn_core.py")
)
_idbn_mod = importlib.util.module_from_spec(_idbn_spec)
sys.modules["models.idbn"] = _idbn_mod
_idbn_spec.loader.exec_module(_idbn_mod)

# Step 2: load app.py by ABSOLUTE PATH — bypasses all caching & sys.path confusion
_app_spec = importlib.util.spec_from_file_location(
    "app",
    os.path.join(_base, "app.py")
)
_app_mod = importlib.util.module_from_spec(_app_spec)
sys.modules["app"] = _app_mod
_app_spec.loader.exec_module(_app_mod)
app = _app_mod.app

if __name__ == '__main__':
    print("=" * 52)
    print("  AgriSmart - AI-Based Crop Advisory for Maharashtra")
    print("  Open:  http://127.0.0.1:5000")
    print("=" * 52)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

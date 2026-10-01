"""
download_map.py
Run ONCE to download Maharashtra district map:
    python download_map.py

After this, the map works permanently (even offline).
"""
import urllib.request, ssl, json, os

_base = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(_base, 'static', 'maharashtra.geojson')

# Skip SSL verification (fixes Windows certificate errors)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode    = ssl.CERT_NONE

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120'}

# Sources to try — some are full India (needs filtering), some are direct
SOURCES = [
    # Direct Maharashtra files
    {
        'url':    'https://raw.githubusercontent.com/datameet/maps/master/districts/maharashtra.geojson',
        'filter': False,
    },
    {
        'url':    'https://raw.githubusercontent.com/datameet/maps/master/Districts/maharashtra.geojson',
        'filter': False,
    },
    # All-India files — filter for Maharashtra
    {
        'url':    'https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson',
        'filter': True,
        'state_keys': ['NAME_1', 'ST_NM', 'state', 'State', 'STATE'],
        'state_val':  'maharashtra',
    },
    {
        'url':    'https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States',
        'filter': True,
        'state_keys': ['NAME_1', 'ST_NM'],
        'state_val':  'maharashtra',
    },
    {
        'url':    'https://raw.githubusercontent.com/HindustanTimesLabs/shrug-geodata/master/district/census2011/districts-census2011.geojson',
        'filter': True,
        'state_keys': ['state_name', 'ST_NM', 'NAME_1'],
        'state_val':  'maharashtra',
    },
]

def try_download(src):
    url = src['url']
    print(f'  Trying: {url}')
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
        raw  = resp.read()
        data = json.loads(raw.decode('utf-8'))

    if not data.get('features'):
        print('  -> No features found')
        return None

    if src.get('filter'):
        keys    = src['state_keys']
        val     = src['state_val']
        subset  = []
        for f in data['features']:
            props = f.get('properties', {})
            for k in keys:
                if str(props.get(k, '')).lower() == val:
                    subset.append(f)
                    break
        if not subset:
            # Show available keys to help debug
            sample = list(data['features'][0]['properties'].keys()) if data['features'] else []
            print(f'  -> Maharashtra not found. Property keys: {sample}')
            return None
        data = {'type': 'FeatureCollection', 'features': subset}

    n = len(data['features'])
    print(f'  -> Got {n} features')
    return data

print('=' * 55)
print('  Maharashtra District Map Downloader')
print('=' * 55)

success = False
for src in SOURCES:
    try:
        data = try_download(src)
        if data:
            with open(OUT, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            print(f'\nSaved to: {OUT}')
            print(f'Districts: {len(data["features"])}')
            print('\nDONE. Restart the Flask server and the map will load.')
            success = True
            break
    except Exception as e:
        print(f'  -> Error: {e}')

if not success:
    print('\nAll sources failed.')
    print('Please download manually from:')
    print('  https://github.com/datameet/maps')
    print(f'And save as: {OUT}')

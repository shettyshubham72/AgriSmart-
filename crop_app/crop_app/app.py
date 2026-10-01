"""
app.py  -  AgriSmart: AI-Based Crop Advisory for Maharashtra
Run directly:  python app.py
"""
import sys, os, importlib.util

_base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base)
sys.path.insert(0, os.path.join(_base, 'models'))

# Alias models/idbn.py (crop_prediction IDBN — matches idbn_full.pkl)
_spec = importlib.util.spec_from_file_location(
    "models.idbn", os.path.join(_base, "models/idbn.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["models.idbn"] = _mod
_spec.loader.exec_module(_mod)

import pickle, warnings, urllib.request
import numpy as np
import joblib
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# ── Load new 20-feature model + RF ensemble ───────────────────
preprocessor = joblib.load(os.path.join(_base, 'models/preprocessor_new.pkl'))
FEATURES = preprocessor['features']   # 20 features
le       = preprocessor['le']         # 12 crops
scaler   = preprocessor['scaler']

with open(os.path.join(_base, 'models/idbn_full.pkl'), 'rb') as f:
    model = pickle.load(f)

rf_model   = joblib.load(os.path.join(_base, 'models/rf_ensemble.pkl'))
ens_cfg    = joblib.load(os.path.join(_base, 'models/ensemble_config.pkl'))
W_IDBN     = ens_cfg['idbn_weight']   # 0.6
W_RF       = ens_cfg['rf_weight']     # 0.4

# ── Crop info (12 crops) ──────────────────────────────────────
CROP_INFO = {
    'Rice':      {'season':'Kharif (Jun-Nov)',  'water':'High',    'duration':'90-150 days',  'maharashtra':'Konkan, Vidarbha'},
    'Wheat':     {'season':'Rabi (Nov-Mar)',    'water':'Medium',  'duration':'100-120 days', 'maharashtra':'Nashik, Jalgaon, Aurangabad'},
    'Maize':     {'season':'Kharif / Rabi',     'water':'Medium',  'duration':'80-95 days',   'maharashtra':'Nashik, Pune, Ahmednagar'},
    'Cotton':    {'season':'Kharif (Jun-Nov)',  'water':'Medium',  'duration':'150-180 days', 'maharashtra':'Vidarbha, Marathwada (Black soil)'},
    'Sugarcane': {'season':'Year-round',        'water':'High',    'duration':'12-18 months', 'maharashtra':'Pune, Nashik, Kolhapur'},
    'Soybean':   {'season':'Kharif',            'water':'Low',     'duration':'90-100 days',  'maharashtra':'Marathwada, Vidarbha'},
    'Groundnut': {'season':'Kharif',            'water':'Low',     'duration':'90-110 days',  'maharashtra':'Pune, Satara, Solapur'},
    'Barley':    {'season':'Rabi (Oct-Mar)',     'water':'Low',     'duration':'90-110 days',  'maharashtra':'Nashik, Aurangabad'},
    'Chickpea':  {'season':'Rabi (Oct-Mar)',     'water':'Low',     'duration':'90-100 days',  'maharashtra':'Vidarbha, Marathwada'},
    'Lentil':    {'season':'Rabi',              'water':'Low',     'duration':'80-110 days',  'maharashtra':'Nashik, Aurangabad'},
    'Mustard':   {'season':'Rabi (Oct-Mar)',     'water':'Low',     'duration':'80-110 days',  'maharashtra':'Nashik, Aurangabad, Pune'},
    'Sunflower': {'season':'Kharif / Rabi',     'water':'Medium',  'duration':'90-100 days',  'maharashtra':'All regions'},
    'Jowar':     {'season':'Kharif / Rabi',     'water':'Low',     'duration':'100-120 days', 'maharashtra':'Marathwada, Vidarbha, Solapur'},
    'Bajra':     {'season':'Kharif',            'water':'Low',     'duration':'75-90 days',   'maharashtra':'Nashik, Ahmednagar, Pune (rainfed)'},
    'Tur':       {'season':'Kharif (Jun-Nov)',  'water':'Low',     'duration':'150-180 days', 'maharashtra':'Marathwada, Vidarbha (major pulse)'},
    'Onion':     {'season':'Rabi (Oct-Feb)',    'water':'Medium',  'duration':'100-120 days', 'maharashtra':'Nashik, Ahmednagar, Pune (Maharashtra is #1 producer)'},
}

# ── Ideal ranges (from CROP_CONFIG in generate_dataset.py) ───
IDEAL_RANGES = {
    'Rice':      {'N':(80,120),  'P':(40,60),  'K':(40,60),  'temperature':(20,35),'humidity':(70,90), 'ph':(5.5,7.0),'rainfall':(150,300)},
    'Wheat':     {'N':(100,150), 'P':(50,80),  'K':(40,60),  'temperature':(12,25),'humidity':(50,70), 'ph':(6.0,7.5),'rainfall':(50,100)},
    'Maize':     {'N':(120,200), 'P':(50,80),  'K':(80,120), 'temperature':(18,32),'humidity':(55,75), 'ph':(5.8,7.0),'rainfall':(60,120)},
    'Cotton':    {'N':(80,120),  'P':(40,60),  'K':(40,80),  'temperature':(25,38),'humidity':(50,70), 'ph':(6.0,8.0),'rainfall':(60,120)},
    'Sugarcane': {'N':(100,150), 'P':(40,60),  'K':(120,160),'temperature':(20,35),'humidity':(70,85), 'ph':(6.0,7.5),'rainfall':(100,200)},
    'Soybean':   {'N':(20,40),   'P':(60,80),  'K':(40,60),  'temperature':(20,30),'humidity':(60,80), 'ph':(6.0,7.0),'rainfall':(60,100)},
    'Groundnut': {'N':(20,40),   'P':(60,80),  'K':(40,60),  'temperature':(25,35),'humidity':(55,75), 'ph':(6.0,7.0),'rainfall':(50,100)},
    'Barley':    {'N':(80,120),  'P':(40,60),  'K':(40,60),  'temperature':(10,22),'humidity':(45,65), 'ph':(6.0,7.5),'rainfall':(40,80)},
    'Chickpea':  {'N':(20,40),   'P':(60,80),  'K':(40,60),  'temperature':(15,30),'humidity':(40,60), 'ph':(6.0,8.0),'rainfall':(40,80)},
    'Lentil':    {'N':(20,40),   'P':(40,60),  'K':(20,40),  'temperature':(15,25),'humidity':(40,60), 'ph':(6.0,8.0),'rainfall':(30,60)},
    'Mustard':   {'N':(80,120),  'P':(40,60),  'K':(40,60),  'temperature':(10,25),'humidity':(40,60), 'ph':(6.0,7.5),'rainfall':(30,60)},
    'Sunflower': {'N':(60,100),  'P':(60,80),  'K':(80,120), 'temperature':(20,32),'humidity':(50,70), 'ph':(6.0,7.5),'rainfall':(50,100)},
    'Jowar':     {'N':(60,100),  'P':(40,60),  'K':(40,60),  'temperature':(25,35),'humidity':(40,65), 'ph':(6.0,7.5),'rainfall':(40,80)},
    'Bajra':     {'N':(60,100),  'P':(30,50),  'K':(20,40),  'temperature':(25,38),'humidity':(35,60), 'ph':(6.0,7.5),'rainfall':(30,60)},
    'Tur':       {'N':(20,40),   'P':(50,80),  'K':(30,60),  'temperature':(20,35),'humidity':(50,75), 'ph':(6.0,8.0),'rainfall':(50,100)},
    'Onion':     {'N':(80,120),  'P':(40,60),  'K':(60,100), 'temperature':(15,25),'humidity':(50,70), 'ph':(6.0,7.0),'rainfall':(50,100)},
}

# ── Yield ranges (quintal / hectare) ─────────────────────────
YIELD_QTL = {
    'Rice':18,'Wheat':25,'Maize':30,'Cotton':15,'Sugarcane':700,
    'Soybean':10,'Groundnut':14,'Barley':20,'Chickpea':10,
    'Lentil':9,'Mustard':12,'Sunflower':13,
}
YIELD_QTL_RANGE = {
    'Rice':(18,40),'Wheat':(25,40),'Maize':(25,45),'Cotton':(12,22),
    'Sugarcane':(600,900),'Soybean':(8,16),'Groundnut':(10,20),
    'Barley':(18,30),'Chickpea':(8,16),'Lentil':(7,14),
    'Mustard':(8,18),'Sunflower':(10,20),
    'Jowar':(10,22),'Bajra':(8,18),'Tur':(8,16),'Onion':(120,220),
}

# ── Dataset means for XAI perturbation (from training data) ──
# ── Full feature ranges (dataset min–max) for XAI perturbation ──
FEATURE_RANGES = {
    'nitrogen':(0,250),'phosphorus':(0,150),'potassium':(0,200),
    'soil_pH':(4.0,9.0),'calcium':(200,2000),'magnesium':(50,500),
    'zinc':(0.2,3.0),'manganese':(1,20),'boron':(0.1,2.0),
    'cation_exchange_capacity':(5,40),'soil_moisture':(20,80),
    'temperature':(5,48),'rainfall':(10,500),'humidity':(15,100),
    'wind_speed':(0.5,5.0),'solar_radiation':(10,30),
    'evapotranspiration':(0.5,15),'length_growing_period':(60,270),
    'npk_ratio':(0,220),'altitude':(0,2000),
}

DATASET_MEANS = {
    'nitrogen':84.154,'phosphorus':59.26,'potassium':64.966,'soil_pH':6.697,
    'calcium':1100.212,'magnesium':275.294,'zinc':1.615,'manganese':10.627,
    'boron':1.051,'cation_exchange_capacity':22.474,'soil_moisture':49.744,
    'temperature':23.902,'rainfall':89.037,'humidity':61.828,
    'wind_speed':2.773,'solar_radiation':19.948,'evapotranspiration':1.106,
    'length_growing_period':164.192,'npk_ratio':69.46,'altitude':999.816,
}

FEATURE_LABELS = {
    'nitrogen':'Nitrogen (N)','phosphorus':'Phosphorus (P)','potassium':'Potassium (K)',
    'soil_pH':'Soil pH','calcium':'Calcium','magnesium':'Magnesium',
    'zinc':'Zinc','manganese':'Manganese','boron':'Boron',
    'cation_exchange_capacity':'CEC','soil_moisture':'Soil Moisture',
    'temperature':'Temperature','rainfall':'Rainfall','humidity':'Humidity',
    'wind_speed':'Wind Speed','solar_radiation':'Solar Radiation',
    'evapotranspiration':'Evapotranspiration','length_growing_period':'Growing Period',
    'npk_ratio':'NPK Ratio','altitude':'Altitude',
}

# ── Soil type → micronutrient defaults ───────────────────────
SOIL_MICRONUTRIENTS = {
    'deep_black':   {'calcium':1400,'magnesium':300,'zinc':0.8,'manganese':5.0,'boron':0.6,'cation_exchange_capacity':35},
    'medium_black': {'calcium':900, 'magnesium':200,'zinc':0.9,'manganese':4.5,'boron':0.55,'cation_exchange_capacity':25},
    'red_laterite': {'calcium':420, 'magnesium':85, 'zinc':0.5,'manganese':3.0,'boron':0.4,'cation_exchange_capacity':12},
    'alluvial':     {'calcium':700, 'magnesium':150,'zinc':1.0,'manganese':6.0,'boron':0.7,'cation_exchange_capacity':20},
    'default':      {'calcium':700, 'magnesium':150,'zinc':0.8,'manganese':4.0,'boron':0.5,'cation_exchange_capacity':20},
}

DIVISION_WEATHER = {
    'Konkan':    {'wind_speed':3.5,'solar_radiation':17.0},
    'Nashik':    {'wind_speed':3.0,'solar_radiation':20.0},
    'Pune':      {'wind_speed':2.5,'solar_radiation':20.0},
    'Aurangabad':{'wind_speed':3.5,'solar_radiation':21.0},
    'Amravati':  {'wind_speed':3.0,'solar_radiation':20.5},
    'Nagpur':    {'wind_speed':3.0,'solar_radiation':20.0},
}

DISTRICT_ALTITUDE = {
    'Mumbai City':10,'Mumbai Suburban':10,'Thane':15,'Raigad':30,
    'Ratnagiri':50,'Sindhudurg':30,'Nashik':565,'Dhule':220,
    'Nandurbar':100,'Jalgaon':210,'Ahmednagar':700,'Pune':560,
    'Satara':670,'Sangli':540,'Solapur':480,'Kolhapur':570,
    'Aurangabad':580,'Jalna':500,'Beed':620,'Osmanabad':640,
    'Latur':540,'Nanded':360,'Hingoli':440,'Parbhani':420,
    'Buldhana':440,'Akola':280,'Washim':380,'Amravati':350,
    'Yavatmal':460,'Nagpur':310,'Wardha':280,'Chandrapur':190,
    'Gadchiroli':170,'Gondia':310,'Bhandara':260,'Wardha':280,
}

SEASON_LGP = {'Kharif':120,'Rabi':100,'Summer':75}

def get_soil_key(soil_type_str):
    s = soil_type_str.lower()
    if 'deep black' in s or 'black cotton' in s: return 'deep_black'
    if 'medium' in s and 'black' in s:           return 'medium_black'
    if 'laterite' in s or 'red' in s:            return 'red_laterite'
    if 'alluvial' in s:                          return 'alluvial'
    return 'default'

def derive_full_features(base_inputs, district_data, season, district_name=''):
    """Derive all 20 model features from 7 user inputs + district + season."""
    N, P, K = base_inputs['N'], base_inputs['P'], base_inputs['K']
    micro = SOIL_MICRONUTRIENTS[get_soil_key(district_data.get('soil_type',''))]
    dw    = DIVISION_WEATHER.get(district_data.get('division','Pune'),
                                  {'wind_speed':3.0,'solar_radiation':20.0})
    irr   = district_data.get('irrigation_pct', 30)
    smf   = {'Kharif':1.2,'Rabi':0.9,'Summer':0.7}.get(season, 1.0)
    soil_moisture = min(80, max(20, (irr * 0.6 + 30) * smf))
    altitude = DISTRICT_ALTITUDE.get(district_name, district_data.get('altitude', 400))
    return {
        'nitrogen':N, 'phosphorus':P, 'potassium':K,
        'soil_pH': base_inputs['ph'],
        'calcium': micro['calcium'], 'magnesium': micro['magnesium'],
        'zinc': micro['zinc'], 'manganese': micro['manganese'],
        'boron': micro['boron'],
        'cation_exchange_capacity': micro['cation_exchange_capacity'],
        'soil_moisture': round(soil_moisture, 1),
        'temperature': base_inputs['temperature'],
        'rainfall': base_inputs['rainfall'],
        'humidity': base_inputs['humidity'],
        'wind_speed': dw['wind_speed'],
        'solar_radiation': dw['solar_radiation'],
        'evapotranspiration': district_data.get('et_rate', 4.2),
        'length_growing_period': SEASON_LGP.get(season, 100),
        'npk_ratio': round((N + P + K) / 3, 2),
        'altitude': altitude,
    }

# Fallback: if no district selected, use Maharashtra averages
MAHARASHTRA_DEFAULTS = {
    'soil_type':'Medium black','division':'Pune','irrigation_pct':35,'et_rate':4.2
}

# ── Market prices & input costs ───────────────────────────────
MARKET_PRICES = {
    'Rice':2200,'Wheat':2275,'Maize':1850,'Cotton':6620,'Sugarcane':315,
    'Soybean':4300,'Groundnut':5850,'Barley':1635,'Chickpea':5440,
    'Lentil':5500,'Mustard':5650,'Sunflower':6400,
    'Jowar':2970,'Bajra':2500,'Tur':7000,'Onion':1500,
}
INPUT_COSTS = {
    'Rice':25000,'Wheat':16000,'Maize':18000,'Cotton':28000,'Sugarcane':70000,
    'Soybean':15000,'Groundnut':18000,'Barley':12000,'Chickpea':12000,
    'Lentil':12000,'Mustard':12000,'Sunflower':15000,
    'Jowar':10000,'Bajra':9000,'Tur':15000,'Onion':38000,
}

# ── ICAR fertilizer schedules (kg/ha) ────────────────────────
FERTILIZER_SCHEDULE = {
    'Rice':      [{'stage':'Basal (transplanting)',       'N':50,'P':50,'K':50,'note':'Apply DAP + MOP at planting'},
                  {'stage':'Tillering (21-25 DAT)',       'N':25,'P':0, 'K':0, 'note':'Top-dress urea'},
                  {'stage':'Panicle initiation (45 DAT)','N':25,'P':0, 'K':0, 'note':'Last urea split'}],
    'Wheat':     [{'stage':'Basal (sowing)',              'N':60,'P':60,'K':40,'note':'Full P & K at sowing'},
                  {'stage':'Crown root (21 DAS)',         'N':30,'P':0, 'K':0, 'note':'First urea split — irrigated'},
                  {'stage':'Jointing (45 DAS)',           'N':30,'P':0, 'K':0, 'note':'Second urea split'}],
    'Maize':     [{'stage':'Basal (sowing)',              'N':40,'P':60,'K':40,'note':'Furrow application'},
                  {'stage':'Knee-high (V6 stage)',        'N':60,'P':0, 'K':0, 'note':'Side-dress urea'},
                  {'stage':'Tasseling',                   'N':20,'P':0, 'K':0, 'note':'Foliar urea 2%'}],
    'Cotton':    [{'stage':'Basal',                      'N':30,'P':60,'K':30,'note':'At sowing, full P & K'},
                  {'stage':'Square formation (30 DAS)',  'N':45,'P':0, 'K':0, 'note':'Urea top-dress'},
                  {'stage':'Boll development (60 DAS)',  'N':45,'P':0, 'K':30,'note':'Final split'}],
    'Sugarcane': [{'stage':'Planting (basal)',            'N':50,'P':85,'K':60,'note':'Full P & K at planting'},
                  {'stage':'3rd month',                  'N':75,'P':0, 'K':60,'note':'Urea split — after earthing'},
                  {'stage':'5th month',                  'N':75,'P':0, 'K':60,'note':'Final dose before grand growth'}],
    'Soybean':   [{'stage':'Basal (sowing)',              'N':20,'P':60,'K':40,'note':'Rhizobium inoculation recommended'},
                  {'stage':'Flowering (30 DAS)',          'N':0, 'P':0, 'K':20,'note':'K for pod fill'}],
    'Groundnut': [{'stage':'Basal (sowing)',              'N':20,'P':60,'K':60,'note':'Full P & K — gypsum 400 kg/ha'},
                  {'stage':'Pegging (30 DAS)',            'N':0, 'P':0, 'K':20,'note':'K + gypsum for pod development'}],
    'Barley':    [{'stage':'Basal (sowing)',              'N':40,'P':40,'K':30,'note':'Apply at sowing'},
                  {'stage':'Tillering (25 DAS)',          'N':30,'P':0, 'K':0, 'note':'Top-dress urea'},
                  {'stage':'Jointing (45 DAS)',           'N':20,'P':0, 'K':0, 'note':'Final urea dose'}],
    'Chickpea':  [{'stage':'Basal (sowing)',              'N':20,'P':50,'K':20,'note':'Low N — rhizobium fixes rest'},
                  {'stage':'Flowering (40 DAS)',          'N':0, 'P':0, 'K':20,'note':'K boosts pod set'}],
    'Lentil':    [{'stage':'Basal (sowing)',              'N':20,'P':40,'K':20,'note':'Rhizobium inoculation essential'},
                  {'stage':'Pre-flowering (45 DAS)',      'N':0, 'P':0, 'K':20,'note':'K improves seed quality'}],
    'Mustard':   [{'stage':'Basal (sowing)',              'N':40,'P':40,'K':30,'note':'Sulphur 30 kg/ha recommended'},
                  {'stage':'Rosette stage (25 DAS)',      'N':40,'P':0, 'K':0, 'note':'Top-dress urea'},
                  {'stage':'Flower bud (45 DAS)',         'N':20,'P':0, 'K':0, 'note':'Foliar boron 0.1% spray'}],
    'Sunflower': [{'stage':'Basal (sowing)',              'N':30,'P':60,'K':30,'note':'Full P & K at planting'},
                  {'stage':'V4 stage (25 DAS)',           'N':30,'P':0, 'K':0, 'note':'Side-dress urea'},
                  {'stage':'Bud formation (45 DAS)',      'N':30,'P':0, 'K':30,'note':'K critical for head fill'}],
    'Jowar':     [{'stage':'Basal (sowing)',              'N':40,'P':50,'K':30,'note':'Full P & K at sowing'},
                  {'stage':'Knee-high (20 DAS)',          'N':40,'P':0, 'K':0, 'note':'Top-dress urea'},
                  {'stage':'Pre-flowering (45 DAS)',      'N':20,'P':0, 'K':0, 'note':'Final urea dose'}],
    'Bajra':     [{'stage':'Basal (sowing)',              'N':40,'P':40,'K':20,'note':'Full P & K at sowing'},
                  {'stage':'Tillering (20 DAS)',          'N':40,'P':0, 'K':0, 'note':'Top-dress urea after first rain'},
                  {'stage':'Pre-panicle (35 DAS)',        'N':20,'P':0, 'K':0, 'note':'Final urea dose'}],
    'Tur':       [{'stage':'Basal (sowing)',              'N':20,'P':50,'K':30,'note':'Rhizobium + PSB inoculation — full P & K'},
                  {'stage':'Branching (30 DAS)',          'N':0, 'P':0, 'K':20,'note':'K for pod development'},
                  {'stage':'Flowering (60 DAS)',          'N':0, 'P':0, 'K':20,'note':'Final K dose — foliar boron 0.2%'}],
    'Onion':     [{'stage':'Basal (transplanting)',       'N':50,'P':50,'K':50,'note':'Full P & K at transplanting; apply FYM 25 t/ha before planting'},
                  {'stage':'Establishment (30 DAT)',      'N':50,'P':0, 'K':0, 'note':'Top-dress urea after good canopy cover'},
                  {'stage':'Bulb initiation (60 DAT)',    'N':0, 'P':0, 'K':50,'note':'High K for bulb fill; stop N to avoid soft bulbs'}],
}

# ── Crop calendar — real sowing months ───────────────────────
CROP_SOWING_MONTH = {
    'Rice':      {'Kharif':6,'Rabi':11,'Summer':2},
    'Wheat':     {'Kharif':None,'Rabi':11,'Summer':None},
    'Maize':     {'Kharif':6,'Rabi':10,'Summer':2},
    'Cotton':    {'Kharif':5,'Rabi':None,'Summer':None},
    'Sugarcane': {'Kharif':6,'Rabi':10,'Summer':2},
    'Soybean':   {'Kharif':6,'Rabi':None,'Summer':None},
    'Groundnut': {'Kharif':6,'Rabi':None,'Summer':2},
    'Barley':    {'Kharif':None,'Rabi':11,'Summer':None},
    'Chickpea':  {'Kharif':None,'Rabi':10,'Summer':None},
    'Lentil':    {'Kharif':None,'Rabi':11,'Summer':None},
    'Mustard':   {'Kharif':None,'Rabi':10,'Summer':None},
    'Sunflower': {'Kharif':6,'Rabi':10,'Summer':2},
    'Jowar':     {'Kharif':6,'Rabi':10,'Summer':None},
    'Bajra':     {'Kharif':6,'Rabi':None,'Summer':None},
    'Tur':       {'Kharif':6,'Rabi':None,'Summer':None},
    'Onion':     {'Kharif':None,'Rabi':10,'Summer':1},
}

# ── FAO-56 crop coefficients (Kc, mid-season values) ─────────
CROP_KT = {
    'Rice':1.20,'Wheat':0.90,'Maize':1.10,'Cotton':0.95,'Sugarcane':1.25,
    'Soybean':0.85,'Groundnut':0.75,'Barley':0.90,'Chickpea':0.75,
    'Lentil':0.75,'Mustard':0.85,'Sunflower':0.95,
    'Jowar':0.80,'Bajra':0.80,'Tur':0.85,'Onion':0.95,
}

def compute_irrigation_advisory(crop, full_features):
    """ET-based irrigation estimate using FAO-56 Penman-Monteith method."""
    kc       = CROP_KT.get(crop, 0.9)
    et0      = float(full_features.get('evapotranspiration', 4.2))   # mm/day
    lgp      = float(full_features.get('length_growing_period', 100)) # days
    ann_rain = float(full_features.get('rainfall', 80))               # mm/year
    seasonal_rain   = ann_rain * (lgp / 365.0)                        # scale to season
    etc             = kc * et0 * lgp                                  # crop water req (mm)
    effective_rain  = seasonal_rain * 0.75                            # FAO 75% efficiency
    net_irr         = max(0.0, etc - effective_rain)
    freq            = max(5, int(40 / max(0.5, et0 * kc)))           # days between irrigations
    return {
        'etc_mm':           round(etc),
        'effective_rain_mm': round(effective_rain),
        'net_irrigation_mm': round(net_irr),
        'frequency_days':    freq,
        'kc':                kc,
        'lgp_days':          round(lgp),
        'et0_rate':          round(et0, 2),
    }

# ── Parameter bounds for counterfactual ──────────────────────
PARAM_BOUNDS = {'N':(0,200),'P':(5,145),'K':(5,205),'ph':(3.5,9.5),
                'temperature':(8,44),'humidity':(14,100),'rainfall':(20,300)}
PARAM_STEPS  = {'N':10,'P':8,'K':8,'ph':0.3,'temperature':1,'humidity':5,'rainfall':15}

# ── Logic functions ───────────────────────────────────────────
def run_model(full_inputs):
    """Run IDBN+RF ensemble on full 20-feature input dict."""
    row       = np.array([[full_inputs[f] for f in FEATURES]])
    X         = scaler.transform(row)
    idbn_prob = model.predict_proba(X)[0]
    rf_prob   = rf_model.predict_proba(X)[0]
    probs     = W_IDBN * idbn_prob + W_RF * rf_prob
    return probs, X

def compute_feature_importance(full_features, base_probs, best_idx):
    """
    Extreme-value perturbation XAI: each feature pushed to its min and max extreme.
    Importance = max confidence drop across both extremes.
    Always produces non-zero values for truly influential features.
    """
    base_conf = float(base_probs[best_idx])
    importances = []
    for feat in FEATURES:
        lo, hi = FEATURE_RANGES.get(feat, (0, 1))
        best_drop = 0.0
        for pv in [lo, hi]:                              # test both extremes
            perturbed = {**full_features, feat: pv}
            row = np.array([[perturbed[f] for f in FEATURES]])
            X   = scaler.transform(row)
            p   = W_IDBN * model.predict_proba(X)[0] + W_RF * rf_model.predict_proba(X)[0]
            best_drop = max(best_drop, base_conf - float(p[best_idx]))
        importances.append({'feature': feat, 'label': FEATURE_LABELS.get(feat, feat),
                            'drop': max(0.0, best_drop)})
    total = sum(d['drop'] for d in importances) or 1.0
    for d in importances:
        d['importance'] = round(d['drop'] / total * 100, 1)
    importances.sort(key=lambda x: x['importance'], reverse=True)
    return importances

def compute_counterfactual(base_inputs, district_data, season, district_name, current_crop):
    results = []
    for feat in ['N','P','K','ph','temperature','humidity','rainfall']:
        lo, hi = PARAM_BOUNDS[feat]; step = PARAM_STEPS[feat]
        for direction in [1,-1]:
            for mult in [1,2,3,5,8]:
                delta   = direction * step * mult
                new_val = round(base_inputs[feat] + delta, 2)
                if new_val < lo or new_val > hi: continue
                perturbed = {**base_inputs, feat: new_val}
                full = derive_full_features(perturbed, district_data, season, district_name)
                p, _ = run_model(full)
                new_crop = le.classes_[p.argmax()]
                if new_crop != current_crop:
                    lbl = {'N':'Nitrogen','P':'Phosphorus','K':'Potassium',
                           'ph':'Soil pH','temperature':'Temperature',
                           'humidity':'Humidity','rainfall':'Rainfall'}[feat]
                    results.append({
                        'feature': feat,'label': lbl,
                        'direction':'Increase' if delta>0 else 'Decrease',
                        'change': round(abs(delta),2),
                        'from_val': round(base_inputs[feat],2),
                        'to_val': new_val,'new_crop': new_crop,
                        'new_prob': round(float(p.max())*100,1),
                    })
                    break
    results.sort(key=lambda x: x['change'])
    return results[:2]

def estimate_yield(crop, confidence, inputs=None, ideal=None):
    lo, hi = YIELD_QTL_RANGE.get(crop, (10,30))
    if inputs and ideal:
        scores = []
        for param, (plo,phi) in ideal.items():
            key = 'ph' if param=='ph' else param
            val = inputs.get(key, (plo+phi)/2)
            if plo <= val <= phi: scores.append(1.0)
            else:
                gap = (plo-val)/plo if val<plo else (val-phi)/phi
                scores.append(max(0.0, 1.0 - min(gap,1.0)))
        match_score = sum(scores)/len(scores) if scores else 0.5
    else:
        match_score = 0.5
    blend     = 0.65 * match_score + 0.35 * (confidence/100.0)
    estimated = round(lo + (hi-lo) * blend, 1)
    return {'low':lo,'high':hi,'estimated':estimated,
            'unit':'quintal/hectare','match_score':round(match_score*100,1)}

def estimate_roi(crop, yield_est, land_size):
    price   = MARKET_PRICES.get(crop, 2000)
    cost_ha = INPUT_COSTS.get(crop, 20000)
    acres   = float(land_size); ha = acres * 0.4047
    revenue = round(yield_est * price * ha, 0)
    cost    = round(cost_ha * ha, 0)
    profit  = round(revenue - cost, 0)
    roi_pct = round((profit/cost*100) if cost>0 else 0, 1)
    return {'price_per_qtl':price,'yield_qtl':yield_est,'area_ha':round(ha,2),
            'area_acres':acres,'gross_revenue':int(revenue),'input_cost':int(cost),
            'net_profit':int(profit),'roi_pct':roi_pct}

def get_recommendation(base_inputs, district_data, season, district_name):
    full  = derive_full_features(base_inputs, district_data, season, district_name)
    probs, _ = run_model(full)
    top5_idx = probs.argsort()[::-1][:5]
    top5 = [{'crop':le.classes_[i],'probability':round(float(probs[i])*100,1),
              'info':CROP_INFO.get(le.classes_[i],{})} for i in top5_idx]
    best       = top5[0]['crop']
    best_idx   = top5_idx[0]
    confidence = top5[0]['probability']
    best_ideal = IDEAL_RANGES.get(best,{})
    yield_est  = estimate_yield(best, confidence, base_inputs, best_ideal)
    return {
        'top_crop':       best,
        'top_info':       CROP_INFO.get(best,{}),
        'top5':           top5,
        'confidence':     confidence,
        'ideal':          best_ideal,
        'importance':     compute_feature_importance(full, probs, best_idx),
        'yield_est':      yield_est,
        'fertilizer':     FERTILIZER_SCHEDULE.get(best,[]),
        'counterfactual': compute_counterfactual(base_inputs, district_data,
                                                 season, district_name, best),
        'irrigation':     compute_irrigation_advisory(best, full),
        'derived_features': {k:v for k,v in full.items()
            if k not in ['nitrogen','phosphorus','potassium','soil_pH',
                         'temperature','rainfall','humidity']},
    }

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/debug')
def debug():
    tmpl = os.path.join(app.root_path,'templates','index.html')
    return (f'<b>app.py loaded — 20-feature IDBN+RF ensemble</b><br>'
            f'Features: {FEATURES}<br>Crops: {list(le.classes_)}<br>'
            f'Template size: {os.path.getsize(tmpl)} bytes')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        base_inputs = {
            'N':           float(data['nitrogen']),
            'P':           float(data['phosphorus']),
            'K':           float(data['potassium']),
            'ph':          float(data['ph']),
            'temperature': float(data['temperature']),
            'humidity':    float(data['humidity']),
            'rainfall':    float(data['rainfall']),
        }
        season       = data.get('season','Kharif')
        land_size    = float(data.get('land_size',1.0))
        district_name= data.get('district','')
        district_data= data.get('district_data', MAHARASHTRA_DEFAULTS)

        result = get_recommendation(base_inputs, district_data, season, district_name)
        result['season']    = season
        result['land_size'] = land_size
        result['roi']       = estimate_roi(result['top_crop'],
                                           result['yield_est']['estimated'],
                                           land_size)
        return jsonify({'success':True,'result':result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success':False,'error':str(e)})

@app.route('/sample/<crop>')
def sample(crop):
    samples = {
        'Rice':      {'nitrogen':90, 'phosphorus':45,'potassium':43,'temperature':26,'humidity':82,'ph':6.3,'rainfall':220},
        'Cotton':    {'nitrogen':95, 'phosphorus':47,'potassium':55,'temperature':32,'humidity':62,'ph':7.0,'rainfall':85},
        'Wheat':     {'nitrogen':120,'phosphorus':62,'potassium':50,'temperature':18,'humidity':58,'ph':7.0,'rainfall':68},
        'Maize':     {'nitrogen':150,'phosphorus':60,'potassium':95,'temperature':24,'humidity':65,'ph':6.3,'rainfall':90},
        'Sugarcane': {'nitrogen':120,'phosphorus':48,'potassium':135,'temperature':28,'humidity':78,'ph':6.5,'rainfall':150},
        'Soybean':   {'nitrogen':28, 'phosphorus':68,'potassium':48,'temperature':25,'humidity':70,'ph':6.5,'rainfall':80},
        'Groundnut': {'nitrogen':25, 'phosphorus':65,'potassium':48,'temperature':30,'humidity':65,'ph':6.5,'rainfall':75},
        'Chickpea':  {'nitrogen':28, 'phosphorus':68,'potassium':48,'temperature':22,'humidity':48,'ph':7.2,'rainfall':58},
        'Jowar':     {'nitrogen':75, 'phosphorus':48,'potassium':47,'temperature':30,'humidity':52,'ph':6.8,'rainfall':62},
        'Bajra':     {'nitrogen':75, 'phosphorus':38,'potassium':30,'temperature':32,'humidity':46,'ph':6.8,'rainfall':45},
        'Tur':       {'nitrogen':28, 'phosphorus':62,'potassium':43,'temperature':28,'humidity':62,'ph':7.0,'rainfall':72},
        'Onion':     {'nitrogen':95, 'phosphorus':48,'potassium':75,'temperature':20,'humidity':62,'ph':6.5,'rainfall':70},
    }
    return jsonify(samples.get(crop,{}))

@app.route('/whatif', methods=['POST'])
def whatif():
    try:
        data = request.get_json()
        base_inputs = {
            'N':float(data['nitrogen']),'P':float(data['phosphorus']),
            'K':float(data['potassium']),'ph':float(data['ph']),
            'temperature':float(data['temperature']),
            'humidity':float(data['humidity']),'rainfall':float(data['rainfall']),
        }
        season       = data.get('season','Kharif')
        district_name= data.get('district','')
        district_data= data.get('district_data', MAHARASHTRA_DEFAULTS)
        full  = derive_full_features(base_inputs, district_data, season, district_name)
        probs, _ = run_model(full)
        top3_idx = probs.argsort()[::-1][:3]
        top3 = [{'crop':le.classes_[i],'probability':round(float(probs[i])*100,1)}
                for i in top3_idx]
        best       = top3[0]['crop']
        confidence = top3[0]['probability']
        ideal      = IDEAL_RANGES.get(best,{})
        yield_est  = estimate_yield(best, confidence, base_inputs, ideal)
        return jsonify({'success':True,'top3':top3,'yield_est':yield_est,
                        'top_crop':best,'confidence':confidence})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/geojson/maharashtra')
def maharashtra_geojson():
    import ssl
    from flask import Response
    local = os.path.join(_base,'static','maharashtra.geojson')
    if os.path.exists(local):
        return send_file(local, mimetype='application/json')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    urls = [
        'https://cdn.jsdelivr.net/gh/datameet/maps@master/Districts/maharashtra.geojson',
        'https://raw.githubusercontent.com/datameet/maps/master/Districts/maharashtra.geojson',
    ]
    for url in urls:
        try:
            req_obj = urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req_obj,timeout=15,context=ctx) as resp:
                content = resp.read().decode('utf-8')
            if 'FeatureCollection' not in content: continue
            with open(local,'w',encoding='utf-8') as f: f.write(content)
            return Response(content, mimetype='application/json')
        except Exception as e:
            print(f'[GeoJSON] {url}: {e}')
    return jsonify({'error':'GeoJSON unavailable'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data     = request.get_json()
        question = data.get('question', '').strip()
        ctx      = data.get('context', {})
        if not question:
            return jsonify({'answer': 'Please ask a question.'})

        crop    = ctx.get('top_crop', 'Unknown')
        conf    = ctx.get('confidence', 0)
        n       = ctx.get('nitrogen', '—')
        p       = ctx.get('phosphorus', '—')
        k       = ctx.get('potassium', '—')
        ph      = ctx.get('ph', '—')
        temp    = ctx.get('temperature', '—')
        rain    = ctx.get('rainfall', '—')
        land    = ctx.get('land_size', 1)
        season  = ctx.get('season', 'Kharif')
        district= ctx.get('district', 'Maharashtra')
        yield_e = ctx.get('yield_est', {})
        irr     = ctx.get('irrigation', {})
        roi_d   = ctx.get('roi', {})
        fert    = ctx.get('fertilizer', [])

        system_prompt = f"""You are AgriSmart, an expert agricultural advisory assistant for Maharashtra farmers.
You have just completed an AI analysis. Here are the results:

RECOMMENDATION:
- Recommended crop: {crop} ({conf}% confidence)
- Season: {season} | District: {district} | Land: {land} acres

SOIL TEST VALUES:
- Nitrogen (N): {n} kg/ha | Phosphorus (P): {p} kg/ha | Potassium (K): {k} kg/ha | pH: {ph}

WEATHER:
- Temperature: {temp}°C | Rainfall: {rain} mm/yr

YIELD ESTIMATE:
- {yield_e.get('estimated','—')} qtl/ha (range: {yield_e.get('low','—')}–{yield_e.get('high','—')} qtl/ha)

IRRIGATION:
- Net irrigation needed: {irr.get('net_irrigation_mm','—')} mm/season
- Irrigation interval: every {irr.get('interval_days','—')} days

ROI / ECONOMICS:
- Market price: ₹{roi_d.get('price_per_qtl','—')}/qtl
- Gross revenue: ₹{roi_d.get('gross_revenue','—')} | Net profit: ₹{roi_d.get('net_profit','—')}
- ROI: {roi_d.get('roi_pct','—')}%

FERTILIZER SCHEDULE (kg/ha per stage):
{chr(10).join([f"  - {s.get('stage','')}: N={s.get('N',0)}, P={s.get('P',0)}, K={s.get('K',0)}" for s in fert]) if fert else '  Not available'}

Answer the user's question concisely (2–4 sentences). Be practical and specific.
When asked about bags: 1 bag Urea=50kg, contains 46% N. 1 bag DAP=50kg, 18% N + 46% P. 1 bag MOP=50kg, 60% K. 1 bag SSP=50kg, 16% P.
Convert kg/ha to bags per acre when asked (1 acre = 0.4 ha).
If asked about something outside this data, say you don't have that information."""

        # Try Ollama (local free LLM — install from ollama.com, run: ollama pull llama3.2)
        try:
            import urllib.request as _ur, json as _json
            payload = _json.dumps({
                'model': 'llama3.2',
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': question}
                ],
                'stream': False
            }).encode()
            req_obj = _ur.Request('http://localhost:11434/api/chat',
                data=payload, headers={'Content-Type':'application/json'})
            with _ur.urlopen(req_obj, timeout=20) as resp:
                result = _json.loads(resp.read())
            answer = result.get('message', {}).get('content', '')
            if answer:
                return jsonify({'answer': answer, 'source': 'ollama'})
        except Exception:
            pass  # Ollama not running — fall through to rule-based

        # Rule-based fallback (always works, no external dependency)
        answer = _rule_based_chat(question.lower(), crop, conf, n, p, k, ph,
                                  land, yield_e, irr, roi_d, fert, season, district)
        return jsonify({'answer': answer, 'source': 'rule'})

    except Exception as e:
        return jsonify({'answer': f'Error: {str(e)}'})


def _rule_based_chat(q, crop, conf, n, p, k, ph, land, yield_e, irr, roi_d, fert, season, district):
    """Smart keyword-based fallback when Ollama is not running."""
    land = float(land) if land else 1.0
    area_ha = land * 0.4047

    # Bag calculations
    def bags(kg_per_ha, fertilizer='urea'):
        total_kg = kg_per_ha * area_ha
        bag_size = 50
        pct = {'urea':0.46,'dap':0.18,'mop':0.60,'ssp':0.16}.get(fertilizer, 0.46)
        nutrient_per_bag = bag_size * pct
        return round(total_kg / nutrient_per_bag, 1) if nutrient_per_bag else 0

    if any(w in q for w in ['bag','urea','dap','mop','ssp','fertilizer','fertiliser','khad']):
        total_n = sum(s.get('N',0) for s in fert) if fert else float(n or 0)
        total_p = sum(s.get('P',0) for s in fert) if fert else float(p or 0)
        total_k = sum(s.get('K',0) for s in fert) if fert else float(k or 0)
        u_bags = bags(total_n/0.46 if total_n else 0, 'urea')
        d_bags = bags(total_p/0.46*0.5 if total_p else 0, 'dap')
        m_bags = bags(total_k/0.60 if total_k else 0, 'mop')
        return (f"For {land} acres of {crop}: approximately {u_bags} bags of Urea (50kg), "
                f"{d_bags} bags of DAP, and {m_bags} bags of MOP. "
                f"Always split application per the ICAR schedule — basal at sowing, top-dress at 30 and 60 days.")

    if any(w in q for w in ['profit','roi','revenue','earn','income','return','money','rupee','₹']):
        return (f"For {land} acres of {crop}: estimated gross revenue ₹{roi_d.get('gross_revenue','—')}, "
                f"input costs ₹{roi_d.get('input_cost','—')}, net profit ₹{roi_d.get('net_profit','—')}. "
                f"ROI is {roi_d.get('roi_pct','—')}% at ₹{roi_d.get('price_per_qtl','—')}/qtl market price.")

    if any(w in q for w in ['yield','production','qtl','quintal','ton','output']):
        return (f"{crop} yield estimate for your soil: {yield_e.get('estimated','—')} qtl/ha "
                f"(range {yield_e.get('low','—')}–{yield_e.get('high','—')} qtl/ha). "
                f"For {land} acres ({round(land*0.4047,2)} ha) total estimated output is "
                f"{round(float(yield_e.get('estimated',0) or 0)*land*0.4047,1)} qtl.")

    if any(w in q for w in ['irrigat','water','sinch','mm','drip']):
        return (f"{crop} needs {irr.get('net_irrigation_mm','—')} mm net irrigation this {season} season. "
                f"Irrigate every {irr.get('interval_days','—')} days. "
                f"Total ETc is {irr.get('etc_mm','—')} mm/season (FAO-56 Penman-Monteith method).")

    if any(w in q for w in ['when','sow','plant','harvest','calendar','month']):
        sow_map = {'Rice':'June–July','Wheat':'November','Maize':'June or November',
                   'Cotton':'June','Sugarcane':'January–March','Soybean':'June–July',
                   'Groundnut':'June','Chickpea':'October','Jowar':'June or October',
                   'Bajra':'June–July','Tur':'June–July','Onion':'October–November'}
        return (f"{crop} sowing window for {season}: {sow_map.get(crop,'check ICAR calendar')}. "
                f"Harvest is expected ~{yield_e.get('duration','90–120 days')} after sowing. "
                f"For {district}, adjust sowing by 1–2 weeks based on monsoon onset.")

    if any(w in q for w in ['why','recommend','reason','confidence','model','ai']):
        return (f"The AI model recommended {crop} with {conf}% confidence based on your soil values "
                f"(N={n}, P={p}, K={k} kg/ha, pH={ph}) and {season} season weather in {district}. "
                f"Key factors: nitrogen level, temperature fit, and rainfall match for this crop's ideal range.")

    if any(w in q for w in ['ph','acid','alkalin','lime','gypsum']):
        ph_v = float(ph) if ph != '—' else 7.0
        remedy = ('Apply 2–3 t/ha lime (calcium carbonate) to raise pH.' if ph_v < 6.0
                  else 'Apply 1 t/ha gypsum to reduce alkalinity.' if ph_v > 7.5
                  else 'pH is in the ideal range (6.0–7.5) — no amendment needed.')
        return f"Your soil pH is {ph}. {remedy} Retest soil after 3 months."

    if any(w in q for w in ['stage','dose','schedule','apply','basal','top']):
        if fert:
            lines = ' | '.join([f"{s['stage']}: N={s.get('N',0)}, P={s.get('P',0)}, K={s.get('K',0)} kg/ha"
                                 for s in fert])
            return f"ICAR fertilizer schedule for {crop}: {lines}. Always apply basal dose at sowing."
        return f"Follow the ICAR-PKV Akola fertilizer schedule for {crop} shown in the Fertilizer Schedule card."

    return (f"Based on the analysis, {crop} is recommended for your {land}-acre farm in {district} "
            f"({season} season) with {conf}% confidence. "
            f"Yield estimate: {yield_e.get('estimated','—')} qtl/ha, "
            f"net profit: ₹{roi_d.get('net_profit','—')}. "
            f"Ask me about fertilizer bags, irrigation, ROI, or sowing dates for more details.")


if __name__ == '__main__':
    print("="*52)
    print("  AgriSmart - AI-Based Crop Advisory for Maharashtra")
    print("  Model: IDBN+RF Ensemble | 20 features | 12 crops")
    print("  Open:  http://127.0.0.1:5000")
    print("="*52)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

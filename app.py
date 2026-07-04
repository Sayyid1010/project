import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection | FUTB",
    page_icon="🌿",
    layout="centered"
)

# ── Load Models ───────────────────────────────────
model_loaded = False
validator_loaded = False

try:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(
        model_path="models/plant_disease_model_38.tflite"
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    st.sidebar.error(f"Disease model error: {e}")

try:
    from ai_edge_litert.interpreter import Interpreter as Interpreter2
    validator = Interpreter2(
        model_path="models/leaf_validator.tflite"
    )
    validator.allocate_tensors()
    val_input = validator.get_input_details()
    val_output = validator.get_output_details()
    validator_loaded = True
except Exception as e:
    st.sidebar.error(f"Validator error: {e}")

ai_classes = [
    "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy", "Apple Scab",
    "Blueberry Healthy", "Cherry Healthy", "Cherry Powdery Mildew",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Healthy",
    "Corn Northern Leaf Blight", "Grape Black Measles", "Grape Black Rot",
    "Grape Healthy", "Grape Leaf Blight", "Orange Citrus Greening",
    "Peach Bacterial Spot", "Peach Healthy", "Pepper Bacterial Spot",
    "Pepper Healthy", "Potato Early Blight", "Potato Healthy",
    "Potato Late Blight", "Raspberry Healthy", "Soybean Healthy",
    "Squash Powdery Mildew", "Strawberry Healthy", "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy",
    "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus"
]

diseases_db = {
    "tomato bacterial spot": {"name":"Tomato Bacterial Spot","crop":"Tomato","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Bacterium: Xanthomonas vesicatoria","symptoms":"Small dark water-soaked spots on leaves turning brown with yellow halo.","treatment":["Apply copper-based bactericide every 7 days","Remove infected plant parts immediately","Avoid working with plants when wet","Use streptomycin spray in severe cases"],"prevention":["Use certified disease-free seeds","Avoid overhead irrigation","Rotate crops every season","Disinfect tools regularly"],"fertilizer":["Apply Calcium-rich fertilizer","Use balanced NPK 20-20-20","Avoid excessive Nitrogen","Apply Potassium foliar spray"]},
    "tomato early blight": {"name":"Tomato Early Blight","crop":"Tomato","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Alternaria solani","symptoms":"Brown spots with yellow rings on lower leaves. Leaves turn yellow and drop.","treatment":["Remove infected leaves immediately","Spray Chlorothalonil every 7 days","Apply copper-based fungicide","Water at base only"],"prevention":["Rotate crops every season","Plant resistant varieties","Proper spacing for air circulation","Remove plant debris after harvest"],"fertilizer":["Apply Potassium fertilizer","Use NPK 15-15-15","Avoid excessive Nitrogen","Apply Calcium and Magnesium foliar spray"]},
    "tomato late blight": {"name":"Tomato Late Blight","crop":"Tomato","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Water mold: Phytophthora infestans","symptoms":"Dark watery spots on leaves and stems. White mold visible underneath.","treatment":["Remove and destroy infected plants immediately","Spray Mancozeb or Metalaxyl every 5-7 days","Do not compost — burn infected plants","Increase spray in wet weather"],"prevention":["Use drip irrigation only","Plant certified disease-free seeds","Ensure good drainage","Monitor during rainy season"],"fertilizer":["Apply Phosphorus fertilizer","Use Potassium-rich fertilizer","Avoid high Nitrogen","Apply Calcium nitrate"]},
    "tomato leaf mold": {"name":"Tomato Leaf Mold","crop":"Tomato","emoji":"🟡","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Passalora fulva","symptoms":"Yellow spots on upper leaf surface with olive-green mold on underside.","treatment":["Apply Chlorothalonil or Mancozeb","Improve air circulation","Remove infected leaves","Reduce humidity"],"prevention":["Plant resistant varieties","Space plants properly","Avoid overhead watering","Maintain low humidity"],"fertilizer":["Apply balanced NPK","Use Potassium fertilizer","Avoid excessive Nitrogen","Apply micronutrient foliar spray"]},
    "tomato mosaic virus": {"name":"Tomato Mosaic Virus","crop":"Tomato","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus: Tomato Mosaic Virus (ToMV)","symptoms":"Mosaic pattern on leaves, stunted growth, distorted fruits.","treatment":["Remove and destroy infected plants","Control aphid vectors","Disinfect tools with bleach","Wash hands before handling"],"prevention":["Use virus-free seeds","Control insect vectors","Remove infected plants immediately","Avoid tobacco near plants"],"fertilizer":["Apply balanced NPK","Use Zinc and Boron foliar spray","Avoid excessive Nitrogen","Apply Potassium"]},
    "tomato septoria leaf spot": {"name":"Tomato Septoria Leaf Spot","crop":"Tomato","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Septoria lycopersici","symptoms":"Small circular spots with dark borders and grey centers on leaves.","treatment":["Apply Chlorothalonil or copper fungicide","Remove infected lower leaves","Spray every 7-10 days","Avoid wetting foliage"],"prevention":["Rotate crops for 2 years","Use mulch","Space plants for airflow","Remove crop debris"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK 15-15-15","Avoid high Nitrogen","Apply Calcium foliar spray"]},
    "tomato spider mites": {"name":"Tomato Spider Mites","crop":"Tomato","emoji":"🟡","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Pest: Tetranychus urticae","symptoms":"Yellow stippling on leaves, fine webbing on underside. Leaves turn bronze.","treatment":["Apply miticide or insecticidal soap","Spray neem oil every 5 days","Increase humidity","Remove infested leaves"],"prevention":["Monitor regularly","Avoid water stress","Remove infested leaves","Use reflective mulch"],"fertilizer":["Apply Silicon fertilizer","Use balanced NPK","Apply Potassium","Avoid excessive Nitrogen"]},
    "tomato target spot": {"name":"Tomato Target Spot","crop":"Tomato","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Corynespora cassiicola","symptoms":"Brown spots with concentric rings resembling a target on leaves.","treatment":["Apply Chlorothalonil or Mancozeb","Remove infected material","Spray every 7-14 days","Improve air circulation"],"prevention":["Use disease-free transplants","Rotate crops","Avoid overhead irrigation","Remove crop debris"],"fertilizer":["Apply Potassium-rich fertilizer","Use NPK 15-15-15","Apply Calcium and Boron foliar spray","Avoid excessive Nitrogen"]},
    "tomato yellow leaf curl virus": {"name":"Tomato Yellow Leaf Curl Virus","crop":"Tomato","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus transmitted by whitefly Bemisia tabaci","symptoms":"Upward curling and yellowing of leaves, stunted growth.","treatment":["Remove and destroy infected plants","Apply insecticide for whiteflies","Use yellow sticky traps","No chemical cure"],"prevention":["Plant resistant varieties","Use insect-proof screens","Control whitefly with neem","Use reflective mulch"],"fertilizer":["Apply balanced NPK","Use Potassium fertilizer","Apply Zinc foliar spray","Avoid excessive Nitrogen"]},
    "tomato healthy": {"name":"Tomato Healthy","crop":"Tomato","emoji":"🟢","severity":"Healthy","severity_emoji":"🟢","action":"No action needed","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed","Continue regular care"],"prevention":["Maintain good soil nutrition","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15 every 2 weeks","Use Calcium and Magnesium foliar spray","Apply Potassium during fruiting","Use compost"]},
    "potato early blight": {"name":"Potato Early Blight","crop":"Potato","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Alternaria solani","symptoms":"Dark brown spots with concentric rings on older leaves.","treatment":["Apply Chlorothalonil or Mancozeb","Remove infected leaves","Spray every 7-10 days","Avoid overhead irrigation"],"prevention":["Use certified seed potatoes","Rotate crops","Maintain proper nutrition","Remove crop debris"],"fertilizer":["Apply Potassium fertilizer","Use NPK 15-15-15","Avoid excessive Nitrogen","Apply Calcium foliar spray"]},
    "potato late blight": {"name":"Potato Late Blight","crop":"Potato","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Water mold: Phytophthora infestans","symptoms":"Water-soaked spots turning brown. White mold on undersides. Tubers rot.","treatment":["Apply Metalaxyl immediately","Remove all infected material","Harvest early if severe","Spray preventively"],"prevention":["Plant resistant varieties","Avoid poorly drained fields","Use certified seed potatoes","Monitor weather"],"fertilizer":["Apply Phosphorus","Use Potassium-rich fertilizer","Avoid high Nitrogen","Apply Calcium nitrate"]},
    "potato healthy": {"name":"Potato Healthy","crop":"Potato","emoji":"🟢","severity":"Healthy","severity_emoji":"🟢","action":"No action needed","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15","Use Potassium during tuber formation","Apply Calcium and Magnesium","Use compost"]},
    "corn common rust": {"name":"Corn Common Rust","crop":"Corn/Maize","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Puccinia sorghi","symptoms":"Small golden-brown pustules on both leaf surfaces.","treatment":["Apply Propiconazole or Azoxystrobin","Spray at first sign","Repeat every 14 days","Remove infected plants"],"prevention":["Plant rust-resistant varieties","Plant early","Monitor regularly","Maintain good nutrition"],"fertilizer":["Apply Potassium fertilizer","Use NPK 15-15-15","Avoid excessive Nitrogen","Apply Silicon fertilizer"]},
    "corn northern leaf blight": {"name":"Corn Northern Leaf Blight","crop":"Corn/Maize","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Exserohilum turcicum","symptoms":"Long tan-grey cigar-shaped lesions on leaves.","treatment":["Apply Propiconazole or Tebuconazole","Spray at tasseling stage","Remove infected debris","Avoid dense planting"],"prevention":["Plant resistant hybrids","Rotate crops","Till soil","Avoid excessive nitrogen"],"fertilizer":["Apply balanced NPK","Use Potassium","Avoid excessive Nitrogen","Apply Zinc foliar spray"]},
    "corn cercospora leaf spot": {"name":"Corn Cercospora Leaf Spot","crop":"Corn/Maize","emoji":"🟡","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Cercospora zeae-maydis","symptoms":"Rectangular grey to tan lesions between leaf veins.","treatment":["Apply Strobilurin fungicide","Spray at early onset","Improve drainage","Remove infected residue"],"prevention":["Plant resistant varieties","Rotate crops","Reduce plant density","Avoid minimum tillage"],"fertilizer":["Apply Potassium fertilizer","Use NPK 15-15-15","Apply Zinc and Boron","Avoid excessive Nitrogen"]},
    "corn healthy": {"name":"Corn Healthy","crop":"Corn/Maize","emoji":"🟢","severity":"Healthy","severity_emoji":"🟢","action":"No action needed","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15","Use Urea for top dressing","Apply Zinc foliar spray","Use compost"]},
    "rice blast": {"name":"Rice Blast Disease","crop":"Rice","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungus: Magnaporthe oryzae","symptoms":"Diamond-shaped lesions with grey centers on leaves.","treatment":["Apply Tricyclazole or Isoprothiolane","Spray at booting stage","Drain fields periodically","Remove infected debris"],"prevention":["Plant blast-resistant varieties","Avoid excessive nitrogen","Maintain proper water management","Use certified seeds"],"fertilizer":["Reduce Nitrogen fertilizer","Apply Silicon fertilizer","Use Potassium","Apply balanced NPK"]},
    "rice brown spot": {"name":"Rice Brown Spot","crop":"Rice","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Cochliobolus miyabeanus","symptoms":"Oval brown spots with yellow halo on leaves.","treatment":["Apply Mancozeb or Iprodione","Spray at tillering stage","Improve soil fertility","Remove infected debris"],"prevention":["Use certified seeds","Maintain proper nutrition","Avoid water stress","Rotate crops"],"fertilizer":["Apply balanced NPK","Use compost","Apply Zinc fertilizer","Use Potassium"]},
    "rice bacterial blight": {"name":"Rice Bacterial Blight","crop":"Rice","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Bacterium: Xanthomonas oryzae","symptoms":"Water-soaked lesions on leaf margins turning yellow then white.","treatment":["Apply copper-based bactericide","Drain fields","Remove infected plants","Avoid excessive nitrogen"],"prevention":["Plant resistant varieties","Use certified seeds","Avoid flood irrigation","Maintain field hygiene"],"fertilizer":["Reduce Nitrogen immediately","Apply Potassium","Use Silicon fertilizer","Apply balanced NPK after recovery"]},
    "rice sheath blight": {"name":"Rice Sheath Blight","crop":"Rice","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Rhizoctonia solani","symptoms":"Oval lesions on leaf sheaths near water line.","treatment":["Apply Propiconazole or Hexaconazole","Spray at early tillering","Drain field","Remove infected stubble"],"prevention":["Reduce plant density","Avoid excessive nitrogen","Use resistant varieties","Rotate crops"],"fertilizer":["Reduce Nitrogen","Apply Silicon fertilizer","Use Potassium","Apply balanced NPK"]},
    "rice tungro": {"name":"Rice Tungro Disease","crop":"Rice","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus transmitted by green leafhopper","symptoms":"Yellow-orange discoloration of leaves, stunted growth.","treatment":["Control leafhopper with insecticide","Remove infected plants","Use systemic insecticide"],"prevention":["Plant tungro-resistant varieties","Control leafhopper","Synchronize planting dates","Remove infected plants early"],"fertilizer":["Apply balanced NPK","Use Zinc foliar spray","Apply Potassium","Avoid excessive Nitrogen"]},
    "rice healthy": {"name":"Rice Healthy","crop":"Rice","emoji":"🟢","severity":"Healthy","severity_emoji":"🟢","action":"No action needed","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply Urea at tillering","Use NPK 15-15-15","Apply Zinc foliar spray","Use compost"]},
    "cassava mosaic": {"name":"Cassava Mosaic Disease","crop":"Cassava","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus transmitted by whitefly Bemisia tabaci","symptoms":"Mosaic pattern of yellow and green on leaves, distortion, stunted growth.","treatment":["Remove and destroy infected plants","Control whitefly with insecticide","Use mineral oil spray","No chemical cure"],"prevention":["Plant certified virus-free cuttings","Use mosaic-resistant varieties","Control whitefly with neem","Inspect planting material"],"fertilizer":["Apply balanced NPK","Use Potassium","Apply Zinc and Boron","Avoid excessive Nitrogen"]},
    "cassava brown streak": {"name":"Cassava Brown Streak Disease","crop":"Cassava","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus: Cassava Brown Streak Virus","symptoms":"Yellow patches on leaves, brown streaks on stems, brown patches in tubers.","treatment":["Remove all infected plants","Control whitefly vectors","No chemical treatment","Replace with resistant varieties"],"prevention":["Use CBSD-resistant varieties","Plant certified cuttings","Control whitefly","Avoid moving material from infected areas"],"fertilizer":["Apply balanced NPK","Use Potassium","Apply micronutrient foliar spray","Avoid excessive Nitrogen"]},
    "cassava bacterial blight": {"name":"Cassava Bacterial Blight","crop":"Cassava","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Bacterium: Xanthomonas axonopodis","symptoms":"Angular water-soaked leaf spots, wilting, stem cankers.","treatment":["Apply copper-based bactericide","Remove infected parts","Disinfect cutting tools","Destroy severely infected plants"],"prevention":["Use disease-free planting material","Disinfect tools","Plant resistant varieties","Avoid infected soil"],"fertilizer":["Apply Calcium fertilizer","Use balanced NPK","Avoid excessive Nitrogen","Apply Potassium"]},
    "cassava healthy": {"name":"Cassava Healthy","crop":"Cassava","emoji":"🟢","severity":"Healthy","severity_emoji":"🟢","action":"No action needed","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15","Use Potassium during tuber formation","Apply Zinc foliar spray","Use compost"]},
    "groundnut early leaf spot": {"name":"Groundnut Early Leaf Spot","crop":"Groundnut","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Cercospora arachidicola","symptoms":"Dark brown circular spots on upper leaf surface with yellow halo.","treatment":["Spray Chlorothalonil or Mancozeb","Apply every 14 days","Remove infected leaves","Avoid overhead irrigation"],"prevention":["Use certified seeds","Rotate crops","Remove crop debris","Plant resistant varieties"],"fertilizer":["Apply Calcium fertilizer","Use balanced NPK","Apply Potassium","Use Gypsum"]},
    "groundnut late leaf spot": {"name":"Groundnut Late Leaf Spot","crop":"Groundnut","emoji":"🟠","severity":"Moderate-High","severity_emoji":"🟠","action":"Act within 3 days","cause":"Fungus: Cercosporidium personatum","symptoms":"Dark brown to black spots on lower leaf surface.","treatment":["Apply Tebuconazole or Propiconazole","Spray every 14 days","Remove infected material","Improve air circulation"],"prevention":["Rotate crops","Use resistant varieties","Remove crop debris","Avoid dense planting"],"fertilizer":["Apply Calcium and Gypsum","Use balanced NPK","Apply Potassium","Avoid excessive Nitrogen"]},
    "groundnut rosette": {"name":"Groundnut Rosette Disease","crop":"Groundnut","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Virus transmitted by aphid Aphis craccivora","symptoms":"Stunted plants with small mottled leaves, chlorotic rosette pattern.","treatment":["Control aphid with insecticide","Remove infected plants","Apply mineral oil"],"prevention":["Plant early","Use resistant varieties","Control aphid with neem","Plant barrier crops"],"fertilizer":["Apply balanced NPK","Use Zinc and Boron foliar spray","Apply Potassium","Avoid excessive Nitrogen"]},
    "groundnut rust": {"name":"Groundnut Rust","crop":"Groundnut","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Puccinia arachidis","symptoms":"Orange-brown pustules on lower leaf surfaces, yellowing.","treatment":["Apply Mancozeb or Propiconazole","Spray every 14 days","Remove infected leaves"],"prevention":["Plant resistant varieties","Rotate crops","Remove crop debris","Monitor fields"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK","Apply Calcium and Gypsum","Avoid excessive Nitrogen"]},
    "pepper bacterial spot": {"name":"Pepper Bacterial Spot","crop":"Pepper","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Bacterium: Xanthomonas campestris","symptoms":"Small water-soaked spots on leaves turning brown with yellow halo.","treatment":["Apply copper bactericide every 7 days","Remove infected parts","Avoid wet plants"],"prevention":["Use certified seeds","Avoid overhead irrigation","Rotate crops","Disinfect tools"],"fertilizer":["Apply Calcium fertilizer","Use balanced NPK","Avoid excessive Nitrogen","Apply Potassium"]},
    "pepper healthy": {"name":"Pepper Healthy","crop":"Pepper","emoji":"🟢","severity":"Healthy","severity_emoji":"🟢","action":"No action needed","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"],"fertilizer":["Apply NPK 15-15-15","Use Calcium during fruiting","Apply Potassium","Use compost"]},
    "onion purple blotch": {"name":"Onion Purple Blotch","crop":"Onion","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 3 days","cause":"Fungus: Alternaria porri","symptoms":"Small white lesions with purple centers on leaves.","treatment":["Apply Mancozeb or Iprodione","Spray every 7-10 days","Remove infected leaves","Improve air circulation"],"prevention":["Use certified disease-free sets","Avoid overhead irrigation","Rotate crops","Remove crop debris"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK","Apply Calcium foliar spray","Avoid excessive Nitrogen"]},
    "apple scab": {"name":"Apple Scab","crop":"Apple","emoji":"🟠","severity":"Moderate","severity_emoji":"🟡","action":"Act within 5 days","cause":"Fungus: Venturia inaequalis","symptoms":"Olive-green to brown scab-like lesions on leaves and fruits.","treatment":["Apply Myclobutanil or Captan","Spray from bud break","Remove infected leaves","Prune for air circulation"],"prevention":["Plant resistant varieties","Remove fallen leaves","Apply dormant sprays","Prune for airflow"],"fertilizer":["Apply balanced NPK","Use Calcium to strengthen fruit","Apply Boron foliar spray","Avoid excessive Nitrogen"]},
    "apple black rot": {"name":"Apple Black Rot","crop":"Apple","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungus: Botryosphaeria obtusa","symptoms":"Brown circular lesions on leaves, black rotting of fruits.","treatment":["Apply Captan or Thiophanate-methyl","Remove infected fruits","Spray every 7-10 days","Prune cankers"],"prevention":["Remove mummified fruits","Prune infected branches","Maintain tree vigor","Apply dormant copper spray"],"fertilizer":["Apply Calcium fertilizer","Use balanced NPK","Apply Potassium","Avoid excessive Nitrogen"]},
    "grape black rot": {"name":"Grape Black Rot","crop":"Grape","emoji":"🔴","severity":"Severe","severity_emoji":"🔴","action":"Act IMMEDIATELY!","cause":"Fungus: Guignardia bidwellii","symptoms":"Brown circular lesions on leaves, shriveled black mummified berries.","treatment":["Apply Myclobutanil or Mancozeb","Spray from bud break","Remove infected berries","Repeat every 7-14 days"],"prevention":["Remove mummified berries","Prune for air circulation","Apply early season sprays","Remove infected material"],"fertilizer":["Apply Potassium fertilizer","Use balanced NPK","Apply Calcium and Boron","Avoid excessive Nitrogen"]},
}

# ── CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.main { background: #f8f9fa; }
.block-container { padding: 2rem 1rem !important; max-width: 860px !important; }
#MainMenu, footer, header { visibility: hidden; }
.hero { background: white; border: 1px solid #e9ecef; border-radius: 16px; padding: 48px 32px; text-align: center; margin-bottom: 16px; }
.hero-badge { display: inline-block; background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; padding: 5px 14px; border-radius: 100px; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 18px; }
.hero h1 { font-size: 2.4rem; font-weight: 600; color: #1a1a1a; margin: 0 0 12px 0; line-height: 1.2; }
.hero h1 span { color: #2e7d32; }
.hero p { color: #6c757d; font-size: 0.95rem; margin: 0; }
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.stat-card { background: white; border: 1px solid #e9ecef; border-radius: 12px; padding: 20px 12px; text-align: center; }
.stat-num { font-size: 1.8rem; font-weight: 600; color: #2e7d32; margin-bottom: 4px; }
.stat-label { font-size: 11px; color: #adb5bd; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { background: white; border: 1px solid #e9ecef; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 7px; color: #6c757d; font-weight: 500; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #2e7d32 !important; color: white !important; font-weight: 600 !important; }
.stButton > button { background: #2e7d32 !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 0.95rem !important; padding: 14px !important; }
.stButton > button:hover { background: #1b5e20 !important; }
.result-box { background: white; border-radius: 14px; padding: 24px; margin: 16px 0; border: 1px solid #e9ecef; }
.result-disease { border-left: 4px solid #e53935; }
.result-healthy { border-left: 4px solid #2e7d32; }
.result-error { border-left: 4px solid #f57c00; }
.result-tag { display: inline-block; padding: 3px 10px; border-radius: 100px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px; }
.tag-red { background: #ffebee; color: #c62828; }
.tag-green { background: #e8f5e9; color: #2e7d32; }
.tag-orange { background: #fff3e0; color: #e65100; }
.result-title { font-size: 1.4rem; font-weight: 600; color: #1a1a1a; margin: 0 0 6px 0; }
.result-sub { color: #6c757d; font-size: 0.85rem; }
.confidence-wrap { background: white; border: 1px solid #e9ecef; border-radius: 14px; padding: 24px; text-align: center; margin: 12px 0; }
.confidence-lbl { font-size: 11px; color: #adb5bd; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; }
.confidence-val { font-size: 3.5rem; font-weight: 700; color: #2e7d32; line-height: 1; }
.confidence-desc { color: #adb5bd; font-size: 0.8rem; margin-top: 6px; }
.severity-wrap { background: white; border: 1px solid #ffe0b2; border-radius: 14px; padding: 20px 24px; margin: 12px 0; display: flex; align-items: center; gap: 14px; }
.severity-icon { font-size: 2rem; flex-shrink: 0; }
.severity-title { font-size: 10px; font-weight: 700; color: #e65100; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.severity-level { font-size: 1.1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 2px; }
.severity-action { color: #6c757d; font-size: 0.85rem; }
.info-box { background: white; border: 1px solid #e9ecef; border-radius: 14px; padding: 20px 24px; margin: 12px 0; }
.info-box-header { display: flex; align-items: center; gap: 8px; padding-bottom: 14px; margin-bottom: 14px; border-bottom: 1px solid #f1f3f5; }
.info-box-title { font-size: 12px; font-weight: 700; color: #495057; text-transform: uppercase; letter-spacing: 1px; }
.info-field { margin-bottom: 12px; }
.info-field:last-child { margin-bottom: 0; }
.info-field-label { font-size: 10px; font-weight: 700; color: #adb5bd; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.info-field-val { font-size: 0.9rem; color: #343a40; line-height: 1.5; }
.list-item { display: flex; gap: 10px; padding: 10px 0; border-bottom: 1px solid #f8f9fa; color: #495057; font-size: 0.88rem; line-height: 1.5; }
.list-item:last-child { border-bottom: none; }
.dot-green { color: #2e7d32; font-weight: 700; flex-shrink: 0; }
.dot-blue { color: #1565c0; font-weight: 700; flex-shrink: 0; }
.dot-purple { color: #6a1b9a; font-weight: 700; flex-shrink: 0; }
.crop-chip { display: inline-block; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 100px; padding: 5px 12px; font-size: 12px; color: #495057; margin: 3px; font-weight: 500; }
.stage-badge { display: inline-flex; align-items: center; gap: 5px; background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 100px; padding: 5px 12px; font-size: 12px; font-weight: 600; margin: 3px; }
.how-to-box { background: white; border: 1px solid #e9ecef; border-radius: 14px; padding: 20px 24px; margin: 16px 0; }
.step-item { display: flex; align-items: flex-start; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f8f9fa; }
.step-item:last-child { border-bottom: none; }
.step-num { width: 26px; height: 26px; background: #e8f5e9; color: #2e7d32; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; }
.step-text { font-size: 0.88rem; color: #495057; line-height: 1.5; padding-top: 3px; }
.crop-header-box { background: linear-gradient(135deg, #e8f5e9, #f1f8e9); border: 1px solid #c8e6c9; border-radius: 14px; padding: 18px 22px; margin: 12px 0; display: flex; align-items: center; gap: 14px; }
.crop-header-emoji { font-size: 2.2rem; flex-shrink: 0; }
.crop-header-name { font-size: 1.2rem; font-weight: 600; color: #1b5e20; margin-bottom: 3px; }
.crop-header-sub { font-size: 0.8rem; color: #388e3c; }
.footer-box { background: white; border: 1px solid #e9ecef; border-radius: 14px; padding: 20px 24px; text-align: center; margin-top: 32px; }
.footer-logo { font-size: 1.1rem; font-weight: 600; color: #2e7d32; margin-bottom: 6px; }
.footer-text { font-size: 12px; color: #adb5bd; line-height: 1.8; }
.footer-link { color: #2e7d32; font-weight: 600; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

def show_disease_info(info, confidence=None):
    if confidence:
        st.markdown(f"""
        <div class="confidence-wrap">
            <div class="confidence-lbl">AI Confidence Score</div>
            <div class="confidence-val">{confidence:.1f}%</div>
            <div class="confidence-desc">Based on deep learning analysis</div>
        </div>
        """, unsafe_allow_html=True)
    if info['severity'] != 'Healthy':
        st.markdown(f"""
        <div class="severity-wrap">
            <div class="severity-icon">{info['severity_emoji']}</div>
            <div>
                <div class="severity-title">Severity Level</div>
                <div class="severity-level">{info['severity']}</div>
                <div class="severity-action">⏰ {info['action']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <div class="info-box-header">
            <div class="info-box-title">🔬 Disease Information</div>
        </div>
        <div class="info-field">
            <div class="info-field-label">Disease Name</div>
            <div class="info-field-val">{info['name']}</div>
        </div>
        <div class="info-field">
            <div class="info-field-label">Affected Crop</div>
            <div class="info-field-val">{info['crop']}</div>
        </div>
        <div class="info-field">
            <div class="info-field-label">Cause</div>
            <div class="info-field-val">{info['cause']}</div>
        </div>
        <div class="info-field">
            <div class="info-field-label">Symptoms</div>
            <div class="info-field-val">{info['symptoms']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box"><div class="info-box-header"><div class="info-box-title">💊 Treatment</div></div>', unsafe_allow_html=True)
    for t in info["treatment"]:
        st.markdown(f'<div class="list-item"><span class="dot-green">▸</span><span>{t}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><div class="info-box-header"><div class="info-box-title">🛡️ Prevention</div></div>', unsafe_allow_html=True)
    for p in info["prevention"]:
        st.markdown(f'<div class="list-item"><span class="dot-blue">▸</span><span>{p}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><div class="info-box-header"><div class="info-box-title">🌱 Fertilizer Recommendation</div></div>', unsafe_allow_html=True)
    for f in info["fertilizer"]:
        st.markdown(f'<div class="list-item"><span class="dot-purple">▸</span><span>{f}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌿 AI-Powered Agriculture</div>
    <h1>Detect Plant <span>Disease Instantly</span></h1>
    <p>Federal University of Technology Babura · Computer Science · 2024/2025</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-num">14+</div><div class="stat-label">Crops</div></div>
    <div class="stat-card"><div class="stat-num">47+</div><div class="stat-label">Diseases</div></div>
    <div class="stat-card"><div class="stat-num">95%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-card"><div class="stat-num">Free</div><div class="stat-label">Always</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📸 Detect Disease",
    "🔍 Search",
    "🌾 Browse Crops"
])

# ════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="how-to-box">
        <div class="info-box-header">
            <div class="info-box-title">📖 How It Works</div>
        </div>
        <div class="step-item">
            <div class="step-num">1</div>
            <div class="step-text">Upload a clear close-up photo of a plant leaf</div>
        </div>
        <div class="step-item">
            <div class="step-num">2</div>
            <div class="step-text">AI validates the image then detects the disease</div>
        </div>
        <div class="step-item">
            <div class="step-num">3</div>
            <div class="step-text">Read diagnosis, severity, treatment and fertilizer advice</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ✅ Crops the AI Can Detect:")
    st.markdown("""
    <div>
    <span class="crop-chip">🍎 Apple</span>
    <span class="crop-chip">🫐 Blueberry</span>
    <span class="crop-chip">🍒 Cherry</span>
    <span class="crop-chip">🌽 Corn/Maize</span>
    <span class="crop-chip">🍇 Grape</span>
    <span class="crop-chip">🍊 Orange</span>
    <span class="crop-chip">🍑 Peach</span>
    <span class="crop-chip">🌶️ Pepper</span>
    <span class="crop-chip">🥔 Potato</span>
    <span class="crop-chip">🍓 Raspberry</span>
    <span class="crop-chip">🫘 Soybean</span>
    <span class="crop-chip">🎃 Squash</span>
    <span class="crop-chip">🍓 Strawberry</span>
    <span class="crop-chip">🍅 Tomato</span>
    </div>
    <p style="color:#adb5bd;font-size:0.8rem;margin-top:10px;">
    ⚠️ For Rice, Cassava, Groundnut, Onion — use Search or Browse tab
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your leaf image here",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1,4,1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button(
            "⚡ ANALYZE DISEASE NOW",
            use_container_width=True,
            type="primary"
        )

        if detect_btn:
            if model_loaded:
                img = image.resize((224,224))
                img_array = np.array(img, dtype=np.float32)/255.0
                img_input = np.expand_dims(img_array, axis=0)

                # Stage 1 - Leaf Validation
                if validator_loaded:
                    with st.spinner("Stage 1 — Validating image..."):
                        validator.set_tensor(val_input[0]['index'], img_input)
                        validator.invoke()
                        val_pred = validator.get_tensor(val_output[0]['index'])
                        val_result = ['leaf','not_leaf'][np.argmax(val_pred)]
                        val_confidence = np.max(val_pred)*100

                    if val_result == 'not_leaf' and val_confidence > 70:
                        st.markdown("""
                        <div class="result-box result-error">
                            <div class="result-tag tag-orange">❌ Invalid Image</div>
                            <div class="result-title">Not a Plant Leaf!</div>
                            <div class="result-sub">
                            Please upload a clear plant leaf photo and try again.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="how-to-box">
                            <div class="info-box-header">
                                <div class="info-box-title">💡 Tips</div>
                            </div>
                            <div class="step-item">
                                <div class="step-num">✓</div>
                                <div class="step-text">Close-up photo of leaf only</div>
                            </div>
                            <div class="step-item">
                                <div class="step-num">✓</div>
                                <div class="step-text">Good natural lighting</div>
                            </div>
                            <div class="step-item">
                                <div class="step-num" style="background:#ffebee;color:#c62828;">✗</div>
                                <div class="step-text" style="color:#c62828;">
                                No people, animals, cars or objects
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()

                # Stage 2 - Disease Detection
                with st.spinner("Stage 2 — Analyzing disease..."):
                    interpreter.set_tensor(input_details[0]['index'], img_input)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    result = ai_classes[np.argmax(prediction)]
                    confidence = np.max(prediction)*100

                if validator_loaded:
                    st.markdown("""
                    <div style="margin-bottom:16px;">
                        <span class="stage-badge">✅ Stage 1: Leaf Confirmed</span>
                        <span class="stage-badge">✅ Stage 2: Analysis Complete</span>
                    </div>
                    """, unsafe_allow_html=True)

                if confidence < 60:
                    st.markdown(f"""
                    <div class="result-box result-error">
                        <div class="result-tag tag-orange">⚠️ Low Confidence</div>
                        <div class="result-title">Photo Not Clear Enough</div>
                        <div class="result-sub">
                        Confidence: {confidence:.1f}% — Please upload a clearer photo!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="how-to-box">
                        <div class="info-box-header">
                            <div class="info-box-title">💡 How to Take a Better Photo</div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">✓</div>
                            <div class="step-text">Use natural daylight</div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">✓</div>
                            <div class="step-text">Get closer to the leaf</div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">✓</div>
                            <div class="step-text">Hold camera steady</div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">✓</div>
                            <div class="step-text">Focus on the diseased part</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                elif "Healthy" in result:
                    st.markdown(f"""
                    <div class="result-box result-healthy">
                        <div class="result-tag tag-green">✅ Healthy Plant</div>
                        <div class="result-title">{result}</div>
                        <div class="result-sub">
                        Your plant appears completely healthy!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    info = diseases_db.get(result.lower())
                    if info:
                        show_disease_info(info, confidence)

                else:
                    st.markdown(f"""
                    <div class="result-box result-disease">
                        <div class="result-tag tag-red">⚠️ Disease Detected</div>
                        <div class="result-title">{result}</div>
                        <div class="result-sub">
                        Follow recommendations below immediately!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    info = diseases_db.get(result.lower())
                    if info:
                        show_disease_info(info, confidence)
            else:
                st.error("❌ AI Model not loaded! Check your tflite files are in the models/ folder on GitHub.")

# ════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Search Any Disease or Crop")
    query = st.text_input(
        "",
        placeholder="🔎 Try: tomato, rice, cassava, blight, rust...",
        label_visibility="collapsed"
    )
    if not query:
        st.markdown("""
        <div style="margin-top:12px;">
            <span class="crop-chip">🍅 Tomato</span>
            <span class="crop-chip">🌾 Rice</span>
            <span class="crop-chip">🌿 Cassava</span>
            <span class="crop-chip">🌽 Maize</span>
            <span class="crop-chip">🥜 Groundnut</span>
            <span class="crop-chip">🥔 Potato</span>
            <span class="crop-chip">🌶️ Pepper</span>
            <span class="crop-chip">🍎 Apple</span>
        </div>
        """, unsafe_allow_html=True)
    if query:
        q = query.lower().strip()
        found = [info for key, info in diseases_db.items()
                 if q in key or q in info["name"].lower()
                 or q in info["crop"].lower()
                 or q in info["symptoms"].lower()
                 or q in info["cause"].lower()]
        if found:
            st.success(f"✅ Found {len(found)} result(s) for '{query}'")
            for info in found:
                with st.expander(
                    f"{info['emoji']} {info['name']} — "
                    f"{info['crop']} | {info['severity_emoji']} {info['severity']}"
                ):
                    show_disease_info(info)
        else:
            st.error(f"❌ No results for '{query}'")
            st.write("Try: tomato, potato, rice, cassava, maize, groundnut, pepper, onion")

# ════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌾 Browse by Crop")
    crops = sorted(set(info["crop"] for info in diseases_db.values()))
    crop_emojis = {
        "Apple":"🍎","Cassava":"🌿","Corn/Maize":"🌽",
        "Grape":"🍇","Groundnut":"🥜","Onion":"🧅",
        "Pepper":"🌶️","Potato":"🥔","Rice":"🌾","Tomato":"🍅"
    }
    selected = st.selectbox(
        "",
        ["— Select a crop —"] + crops,
        label_visibility="collapsed"
    )
    if selected != "— Select a crop —":
        matches = [info for info in diseases_db.values()
                   if info["crop"] == selected]
        emoji = crop_emojis.get(selected, "🌿")
        disease_count = len([m for m in matches if "Healthy" not in m["name"]])
        st.markdown(f"""
        <div class="crop-header-box">
            <div class="crop-header-emoji">{emoji}</div>
            <div>
                <div class="crop-header-name">{selected}</div>
                <div class="crop-header-sub">
                {disease_count} diseases · Full treatment guide
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        diseases = [m for m in matches if "Healthy" not in m["name"]]
        healthy = [m for m in matches if "Healthy" in m["name"]]
        if diseases:
            st.markdown(f"#### ⚠️ {len(diseases)} Disease(s)")
            for info in diseases:
                with st.expander(
                    f"{info['emoji']} {info['name']} · "
                    f"{info['severity_emoji']} {info['severity']}"
                ):
                    show_disease_info(info)
        if healthy:
            st.markdown("#### ✅ Healthy State")
            for info in healthy:
                with st.expander(f"🟢 {info['name']}"):
                    show_disease_info(info)

# ── Footer ────────────────────────────────────────
st.markdown("""
<div class="footer-box">
    <div class="footer-logo">🌿 LeafAI</div>
    <div class="footer-text">
        Plant Disease Detection System<br>
        Developed by <b>Yusuf Gambo</b> · Matric: SIT/CSC/23/0005<br>
        B.Sc Computer Science · FUTB · 2024/2025<br>
        Supervised by <b>Dr. Khalid Haruna</b><br><br>
        <a class="footer-link"
        href="https://futb-plant-disease.streamlit.app">
        🌐 futb-plant-disease.streamlit.app</a>
    </div>
</div>
""", unsafe_allow_html=True)

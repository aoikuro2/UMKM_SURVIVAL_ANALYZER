#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#================================================================================
#UMKM SURVIVAL PREDICTION SYSTEM
#================================================================================
#Script ini menghasilkan data sintetis UMKM di Jatinangor, melatih model klasifikasi Logistic Regression,

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
N_SAMPLES = 1000          # Number of synthetic UMKM records to generate
TEST_SIZE = 0.2           # 20% of data for testing, 80% for training
MODEL_FILE = 'umkm_survival_model.pkl'
SCALER_FILE = 'umkm_scaler.pkl'
ENCODER_FILE = 'umkm_encoder.pkl'

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


# =============================================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(n_samples=1000):
    

    # Feature 1: Jenis Industri UMKM (Categorical)
    # Different industries have different baseline survival characteristics
    jenis_industri = np.random.choice(
        ['Kuliner', 'Fashion', 'Kerajinan', 'Pertanian', 'Jasa', 'Elektronik'], 
        n_samples
    )

    # Feature 2: Jarak dari Area Pemukiman (km)
    # Continuous 0.1 to 10.0 km
    # Closer to residential areas = better foot traffic and accessibility
    jarak_pemukiman = np.round(np.random.uniform(0.1, 10.0, n_samples), 2)

    # Feature 3: Range Harga Produk/Layanan
    # 1 = Rendah (low margin, high volume)
    # 2 = Menengah (optimal for Jatinangor student market)
    # 3 = Tinggi (premium, limited market)
    range_harga = np.random.choice([1, 2, 3], n_samples, p=[0.35, 0.45, 0.20])

    # Feature 4: Biaya Operasional per Bulan (Juta Rupiah)
    # Includes rent, utilities, salaries, raw materials
    # Range: 1 - 50 Juta (realistic for UMKM scale)
    biaya_operasional = np.round(np.random.uniform(1, 50, n_samples), 2)

    # Feature 5: Lama Usaha Beroperasi (Tahun)
    # 0-20 years. First 2 years are critical survival period.
    lama_usaha = np.round(np.random.uniform(0, 20, n_samples), 1)

    # Calculate survival probability using realistic business logic
    survival_probs = np.array([
        _calculate_survival_prob(
            jenis_industri[i], 
            jarak_pemukiman[i], 
            range_harga[i], 
            biaya_operasional[i], 
            lama_usaha[i]
        )
        for i in range(n_samples)
    ])

    # Binary outcome: 1 = High Survival (>50%), 0 = Low Survival (≤50%)
    survival = (survival_probs > 0.5).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'jenis_industri': jenis_industri,
        'jarak_pemukiman_km': jarak_pemukiman,
        'range_harga': range_harga,
        'biaya_operasional_juta': biaya_operasional,
        'lama_usaha_tahun': lama_usaha,
        'survival': survival,
        'survival_probability': np.round(survival_probs, 3)
    })

    return df


def _calculate_survival_prob(industri, jarak, harga, biaya, lama):

    prob = 0.45  # Base probability

    # Industry effect
    industry_effect = {
        'Kuliner': 0.10,
        'Fashion': -0.05,
        'Kerajinan': 0.05,
        'Pertanian': 0.08,
        'Jasa': 0.02,
        'Elektronik': -0.08
    }
    prob += industry_effect.get(industri, 0)

    # Distance effect: closer = better
    prob += 0.20 * (1 - jarak / 10) - 0.10 * (jarak / 10)

    # Price range effect
    if harga == 1:
        prob -= 0.05   # Low margins
    elif harga == 2:
        prob += 0.08   # Optimal for Jatinangor
    elif harga == 3:
        prob -= 0.03   # Limited market

    # Operational cost effect: lower = better
    prob -= 0.20 * (biaya / 50)

    # Business age effect
    if lama <= 2:
        prob -= 0.10   # Critical period
    elif lama <= 5:
        prob += 0.05
    else:
        prob += 0.15 * (lama / 20)

    # Add realistic business uncertainty
    prob += np.random.normal(0, 0.12)

    # Clip to valid probability range
    return np.clip(prob, 0.05, 0.95)


# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

def preprocess_data(df):

    # Encode categorical variable
    label_encoder = LabelEncoder()
    df['jenis_industri_encoded'] = label_encoder.fit_transform(df['jenis_industri'])

    # Define feature columns (order matters for prediction)
    feature_columns = [
        'jenis_industri_encoded',
        'jarak_pemukiman_km', 
        'range_harga', 
        'biaya_operasional_juta', 
        'lama_usaha_tahun'
    ]

    X = df[feature_columns]
    y = df['survival']

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Standardize features (zero mean, unit variance)
    # Important for logistic regression convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_columns


# =============================================================================
# SECTION 3: MODEL TRAINING
# =============================================================================

def train_model(X_train, y_train):

    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
    return model


# =============================================================================
# SECTION 4: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC-ROC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Low Survival', 'High Survival']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Low    High")
    print(f"Actual Low    [{cm[0,0]:3d}]  [{cm[0,1]:3d}]")
    print(f"Actual High   [{cm[1,0]:3d}]  [{cm[1,1]:3d}]")

    # Feature importance
    feature_names = [
        'Jenis Industri',
        'Jarak dari Pemukiman (km)',
        'Range Harga (1-3)',
        'Biaya Operasional (Juta)',
        'Lama Usaha (Tahun)'
    ]

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print("=" * 70)
    print("Positive = increases survival | Negative = decreases survival")
    print("-" * 70)

    for name, coef in zip(feature_names, model.coef_[0]):
        direction = "↑ INCREASES" if coef > 0 else "↓ DECREASES"
        print(f"{name:35s}: {coef:+.4f}  ({direction})")

    print(f"{'Model Intercept':35s}: {model.intercept_[0]:+.4f}")

    return {
        'accuracy': accuracy,
        'auc_roc': auc,
        'confusion_matrix': cm
    }


# =============================================================================
# SECTION 5: PREDICTION & RECOMMENDATION SYSTEM
# =============================================================================

def predict_survival(jenis_industri, jarak_pemukiman, range_harga, 
                     biaya_operasional, lama_usaha,
                     model=None, scaler=None, label_encoder=None):
    
    # Load model components if not provided
    if model is None:
        model = joblib.load(MODEL_FILE)
    if scaler is None:
        scaler = joblib.load(SCALER_FILE)
    if label_encoder is None:
        label_encoder = joblib.load(ENCODER_FILE)

    # Validate inputs
    valid_industries = list(label_encoder.classes_)
    if jenis_industri not in valid_industries:
        raise ValueError(f"jenis_industri must be one of: {valid_industries}")

    if range_harga not in [1, 2, 3]:
        raise ValueError("range_harga must be 1 (Rendah), 2 (Menengah), or 3 (Tinggi)")

    if not (0.1 <= jarak_pemukiman <= 10.0):
        raise ValueError("jarak_pemukiman must be between 0.1 and 10.0 km")

    if not (1 <= biaya_operasional <= 50):
        raise ValueError("biaya_operasional must be between 1 and 50 Juta Rupiah")

    if not (0 <= lama_usaha <= 20):
        raise ValueError("lama_usaha must be between 0 and 20 years")

    # Encode categorical input
    industry_encoded = label_encoder.transform([jenis_industri])[0]

    # Create input array (must match training feature order)
    input_data = np.array([[industry_encoded, jarak_pemukiman, range_harga, 
                            biaya_operasional, lama_usaha]])

    # Scale input using the same scaler as training
    input_scaled = scaler.transform(input_data)

    # Predict probability of class 1 (High Survival)
    probability = model.predict_proba(input_scaled)[0, 1]
    prediction = model.predict(input_scaled)[0]

    # Classification based on 50% threshold
    if prediction == 1:
        classification = "HIGH SURVIVAL"
        status = "Usaha Anda memiliki peluang bertahan tinggi (>50%)"
    else:
        classification = "LOW SURVIVAL"
        status = "Usaha Anda memiliki risiko gagal tinggi (≤50%). Perhatikan rekomendasi perbaikan berikut:"

    result = {
        'survival_probability': round(probability * 100, 2),
        'classification': classification,
        'status_message': status,
        'inputs': {
            'jenis_industri': jenis_industri,
            'jarak_pemukiman_km': jarak_pemukiman,
            'range_harga': range_harga,
            'biaya_operasional_juta': biaya_operasional,
            'lama_usaha_tahun': lama_usaha
        }
    }

    # GENERATE IMPROVEMENT RECOMMENDATIONS (only for low survival)
    if prediction == 0:
        recommendations = []

        # 1. Jarak dari Pemukiman Analysis
        # Coefficient is negative (-0.95), meaning distance HURTS survival
        if jarak_pemukiman > 3.0:
            recommendations.append({
                'feature': 'Jarak dari Area Pemukiman',
                'current_value': f"{jarak_pemukiman} km",
                'target_value': "< 3.0 km",
                'action': 'Pindahkan atau buka cabang lebih dekat pemukiman/area kampus untuk meningkatkan foot traffic dan aksesibilitas',
                'priority': 'HIGH' if jarak_pemukiman > 6.0 else 'MEDIUM',
                'impact': 'Jarak jauh dari pemukiman secara signifikan menurunkan peluang bertahan karena berkurangnya akses pelanggan'
            })

        # 2. Range Harga Analysis
        # Optimal is Menengah (2). Rendah (1) = thin margins, Tinggi (3) = limited market
        if range_harga == 1:
            recommendations.append({
                'feature': 'Range Harga',
                'current_value': 'Rendah (1)',
                'target_value': 'Menengah (2)',
                'action': 'Naikkan harga sedikit dengan meningkatkan kualitas produk/layanan, packaging, atau branding. Jangan perang harga',
                'priority': 'MEDIUM',
                'impact': 'Harga terlalu rendah menyebabkan margin tipis dan sulit menutup biaya operasional, terutama di pasar Jatinangor'
            })
        elif range_harga == 3:
            recommendations.append({
                'feature': 'Range Harga',
                'current_value': 'Tinggi (3)',
                'target_value': 'Menengah (2)',
                'action': 'Pertimbangkan menurunkan harga atau buat varian produk dengan harga menengah untuk pasar mahasiswa Jatinangor',
                'priority': 'MEDIUM',
                'impact': 'Harga tinggi di pasar Jatinangor (dominan mahasiswa dan keluarga menengah) membatasi jumlah pelanggan potensial'
            })

        # 3. Biaya Operasional Analysis
        # Coefficient is negative (-0.65), high cost hurts survival
        if biaya_operasional > 25:
            recommendations.append({
                'feature': 'Biaya Operasional',
                'current_value': f"Rp {biaya_operasional:.2f} Juta/bulan",
                'target_value': '< Rp 25 Juta/bulan',
                'action': 'Efisiensikan biaya: negosiasi ulang sewa tempat, kurangi pegawai sementara, gunakan teknologi untuk otomasi operasional',
                'priority': 'HIGH' if biaya_operasional > 35 else 'MEDIUM',
                'impact': 'Biaya operasional tinggi membebani arus kas, terutama untuk UMKM dengan pendapatan yang fluktuatif'
            })

        # 4. Lama Usaha Analysis
        # Coefficient is positive (+0.57), older businesses survive better
        # First 2 years are critical period
        if lama_usaha < 2:
            recommendations.append({
                'feature': 'Lama Usaha',
                'current_value': f"{lama_usaha} tahun",
                'target_value': 'Minimal 2 tahun untuk stabil',
                'action': 'Fokus pada survival jangka pendek: pertahankan modal kerja, hindari ekspansi dulu, bangun loyalitas pelanggan intensif',
                'priority': 'HIGH',
                'impact': '2 tahun pertama adalah periode paling kritis untuk UMKM. 80% kegagalan terjadi pada periode ini. Prioritaskan cash flow positif'
            })

        # 5. Jenis Industri Analysis
        # Some industries are inherently riskier in Jatinangor context
        problematic_industries = ['Fashion', 'Elektronik']
        if jenis_industri in problematic_industries:
            recommendations.append({
                'feature': 'Jenis Industri',
                'current_value': jenis_industri,
                'target_value': 'Diversifikasi atau pivot model bisnis',
                'action': f'Industri {jenis_industri} sangat kompetitif dan capital-intensive di Jatinangor. Pertimbangkan diversifikasi produk, pivot ke Kuliner/Kerajinan, atau temukan niche market spesifik',
                'priority': 'MEDIUM',
                'impact': f'{jenis_industri} memiliki tingkat kegagalan lebih tinggi di pasar Jatinangor karena persaingan ketat dan perubahan tren cepat'
            })

        # Sort recommendations by priority (HIGH first)
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])

        result['recommendations'] = recommendations
        result['total_recommendations'] = len(recommendations)

    return result


def print_prediction(result):
    

    inputs = result['inputs']

    print("\n" + "=" * 70)
    print("HASIL PREDIKSI KELANGSUNGAN UMKM JATINANGOR")
    print("=" * 70)
    print(f"Jenis Industri        : {inputs['jenis_industri']}")
    print(f"Jarak dari Pemukiman  : {inputs['jarak_pemukiman_km']} km")
    print(f"Range Harga           : {inputs['range_harga']} (1=Rendah, 2=Menengah, 3=Tinggi)")
    print(f"Biaya Operasional     : Rp {inputs['biaya_operasional_juta']:.2f} Juta/bulan")
    print(f"Lama Usaha            : {inputs['lama_usaha_tahun']} tahun")
    print("-" * 70)
    print(f"PROBABILITAS BERTAHAN : {result['survival_probability']}%")
    print(f"KLASIFIKASI           : {result['classification']}")
    print(f"PESAN                 : {result['status_message']}")

    if 'recommendations' in result:
        print("\n" + "-" * 70)
        print(f"REKOMENDASI PERBAIKAN ({result['total_recommendations']} item):")
        print("-" * 70)

        for i, rec in enumerate(result['recommendations'], 1):
            print(f"\n  {i}. [{rec['priority']}] {rec['feature']}")
            print(f"      Saat ini : {rec['current_value']}")
            print(f"      Target   : {rec['target_value']}")
            print(f"      Tindakan : {rec['action']}")
            print(f"      Alasan   : {rec['impact']}")

    print("\n" + "=" * 70)


# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("UMKM SURVIVAL PREDICTION SYSTEM - JATINANGOR")
    print("Logistic Regression Classification Model")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic UMKM data for Jatinangor...")
    df = generate_synthetic_data(N_SAMPLES)
    print(f"       ✓ Generated {len(df)} UMKM records")
    print(f"       ✓ High Survival: {sum(df['survival'])} ({sum(df['survival'])/len(df)*100:.1f}%)")
    print(f"       ✓ Low Survival: {len(df) - sum(df['survival'])} ({(len(df) - sum(df['survival']))/len(df)*100:.1f}%)")

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder, features = preprocess_data(df)
    print(f"       ✓ Training samples: {len(X_train)}")
    print(f"       ✓ Testing samples: {len(X_test)}")
    print(f"       ✓ Features: {features}")

    # Step 3: Train model
    print("\n[3/5] Training Logistic Regression model...")
    model = train_model(X_train, y_train)
    print("       ✓ Model trained successfully")

    # Step 4: Evaluate
    print("\n[4/5] Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Save model
    print("\n[5/5] Saving model artifacts...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    print(f"       ✓ Model saved to: {MODEL_FILE}")
    print(f"       ✓ Scaler saved to: {SCALER_FILE}")
    print(f"       ✓ Encoder saved to: {ENCODER_FILE}")

    # Demonstration predictions
    print("\n" + "=" * 70)
    print("DEMONSTRATION: SAMPLE PREDICTIONS")
    print("=" * 70)

    # Case 1: High survival probability
    print("\n--- CASE 1: UMKM dengan Peluang Tinggi ---")
    result1 = predict_survival(
        jenis_industri='Kuliner',
        jarak_pemukiman=1.5,
        range_harga=2,
        biaya_operasional=12,
        lama_usaha=5,
        model=model, scaler=scaler, label_encoder=encoder
    )
    print_prediction(result1)

    # Case 2: Low survival probability (multiple problems)
    print("\n--- CASE 2: UMKM dengan Risiko Tinggi ---")
    result2 = predict_survival(
        jenis_industri='Fashion',
        jarak_pemukiman=8.5,
        range_harga=3,
        biaya_operasional=40,
        lama_usaha=0.5,
        model=model, scaler=scaler, label_encoder=encoder
    )
    print_prediction(result2)

    # Case 3: Borderline case
    print("\n--- CASE 3: Kasus Borderline ---")
    result3 = predict_survival(
        jenis_industri='Jasa',
        jarak_pemukiman=4.0,
        range_harga=2,
        biaya_operasional=20,
        lama_usaha=3,
        model=model, scaler=scaler, label_encoder=encoder
    )
    print_prediction(result3)

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print("\nTo use this model in your own code:")
    print("  from joblib import load")
    print(f"  model = load('{MODEL_FILE}')")
    print(f"  scaler = load('{SCALER_FILE}')")
    print(f"  encoder = load('{ENCODER_FILE}')")
    print("  result = predict_survival('Kuliner', 2.0, 2, 15, 3, model, scaler, encoder)")
    print("\nOr simply run this script again to see new predictions.")
    print("=" * 70)

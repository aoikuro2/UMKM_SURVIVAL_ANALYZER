#!/usr/bin/env python3
"""
UMKM Survival Predictor for Jatinangor, Sumedang, Jawa Barat
Using Logistic Regression with 5 inputs:
  1. jenis_industri     : Kuliner, Fashion, Jasa, Retail, Kerajinan
  2. jarak_pemukiman    : meters from housing/kos area
  3. kapasitas_kursi    : number of seats
  4. target_harga       : price in thousands IDR (ribu rupiah)
  5. modal_awal         : initial capital in millions IDR (juta rupiah)

Output: Survival probability (0-100%)
  > 50% = High Survival
  < 50% = Low Survival + specific improvement recommendations

Author: AI Assistant
Context: Jatinangor is a university town (UNPAD, ITB, etc.) with student-centric market
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. DATA LOADING / GENERATION
# ============================================================
# NOTE: For production, replace this with your real survey data from UMKM in Jatinangor.
# The synthetic data below mimics realistic Jatinangor market dynamics.

def generate_synthetic_data(n_samples=1200):
    """Generate realistic synthetic UMKM data for Jatinangor."""

    # Feature 1: jenis_industri (Kuliner dominates in student area)
    jenis_industri = np.random.choice(
        ['Kuliner', 'Fashion', 'Jasa', 'Retail', 'Kerajinan'], 
        n_samples, 
        p=[0.50, 0.10, 0.15, 0.15, 0.10]
    )

    # Feature 2: jarak dari area pemukiman/kos (meters)
    # Optimal: 100-800m (walking distance for students)
    jarak_pemukiman = np.random.gamma(2.5, 600, n_samples)
    jarak_pemukiman = np.clip(jarak_pemukiman, 50, 4000)

    # Feature 3: kapasitas kursi
    # Sweet spot for Jatinangor: 15-45 seats
    kapasitas_kursi = np.random.poisson(28, n_samples)
    kapasitas_kursi = np.clip(kapasitas_kursi, 0, 120)

    # Feature 4: target harga (ribu rupiah)
    # Student budget: 15k-50k is ideal
    target_harga = np.random.lognormal(3.8, 0.55, n_samples)
    target_harga = np.clip(target_harga, 8, 300)

    # Feature 5: modal awal (juta rupiah)
    # UMKM Jatinangor: 20M-150M common range
    modal_awal = np.random.lognormal(4.2, 0.7, n_samples)
    modal_awal = np.clip(modal_awal, 5, 500)

    # Generate survival labels with realistic market logic
    survival = np.zeros(n_samples, dtype=int)

    industry_effect = {
        'Kuliner': 0.18,
        'Jasa': 0.08,
        'Retail': 0.02,
        'Fashion': -0.08,
        'Kerajinan': -0.12
    }

    for i in range(n_samples):
        prob = 0.45 + industry_effect[jenis_industri[i]]

        # Jarak effect
        if 100 <= jarak_pemukiman[i] <= 800:
            prob += 0.18
        elif 800 < jarak_pemukiman[i] <= 1500:
            prob += 0.08
        elif 1500 < jarak_pemukiman[i] <= 2500:
            prob += 0.00
        else:
            prob -= 0.15

        # Kapasitas effect
        if 15 <= kapasitas_kursi[i] <= 45:
            prob += 0.12
        elif 5 <= kapasitas_kursi[i] < 15:
            prob += 0.04
        elif kapasitas_kursi[i] > 70:
            prob -= 0.10
        elif kapasitas_kursi[i] < 5:
            prob -= 0.08

        # Harga effect
        if target_harga[i] <= 25:
            prob += 0.15
        elif target_harga[i] <= 50:
            prob += 0.10
        elif target_harga[i] <= 80:
            prob += 0.00
        elif target_harga[i] <= 120:
            prob -= 0.10
        else:
            prob -= 0.20

        # Modal effect
        if 20 <= modal_awal[i] <= 100:
            prob += 0.08
        elif modal_awal[i] < 15:
            prob -= 0.12
        elif modal_awal[i] > 250:
            prob -= 0.08

        # Add noise
        prob += np.random.normal(0, 0.07)
        prob = np.clip(prob, 0.05, 0.95)

        survival[i] = 1 if np.random.random() < prob else 0

    df = pd.DataFrame({
        'jenis_industri': jenis_industri,
        'jarak_pemukiman': jarak_pemukiman,
        'kapasitas_kursi': kapasitas_kursi,
        'target_harga': target_harga,
        'modal_awal': modal_awal,
        'survival': survival
    })

    return df


# ============================================================
# 2. MODEL PIPELINE
# ============================================================

def build_model(df):
    """Build and train logistic regression pipeline."""

    categorical_features = ['jenis_industri']
    numeric_features = ['jarak_pemukiman', 'kapasitas_kursi', 'target_harga', 'modal_awal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED))
    ])

    X = df.drop('survival', axis=1)
    y = df['survival']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("=== Model Evaluation ===")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Survival', 'High Survival']))

    return pipeline, categorical_features, numeric_features


# ============================================================
# 3. SURVIVAL ANALYZER CLASS
# ============================================================

class UMKMSurvivalAnalyzer:
    """
    Main analyzer class.
    - predict(): returns survival probability and classification
    - analyze_improvements(): if Low Survival, returns specific actionable suggestions
    """

    def __init__(self, pipeline, categorical_features, numeric_features):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps['preprocessor']
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.coefficients = pipeline.named_steps['classifier'].coef_[0]
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features

        # Jatinangor-specific optimal values
        self.optimal_values = {
            'jarak_pemukiman': 400,
            'kapasitas_kursi': 25,
            'target_harga': 35,
            'modal_awal': 60
        }
        self.best_industry = 'Kuliner'

    def predict(self, input_data):
        """Predict survival probability."""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        probability = self.pipeline.predict_proba(input_df)[0, 1]
        classification = "High Survival" if probability >= 0.5 else "Low Survival"

        return {
            'probability': probability,
            'classification': classification,
            'score_percent': round(probability * 100, 2)
        }

    def analyze_improvements(self, input_data):
        """
        Full analysis with improvement suggestions for Low Survival cases.
        Returns structured JSON-like dict.
        """
        result = self.predict(input_data)

        if result['classification'] == "High Survival":
            return {
                **result,
                'improvements_needed': False,
                'suggestions': ["Your UMKM shows strong survival potential. Maintain current strategy!"],
                'detailed_improvements': [],
                'combined_potential': result['score_percent']
            }

        if isinstance(input_data, dict):
            current = input_data.copy()
        else:
            current = input_data.iloc[0].to_dict()

        improvements = []
        base_prob = result['probability']

        # Analyze industry pivot
        current_industry = current['jenis_industri']
        if current_industry != self.best_industry:
            test_data = current.copy()
            test_data['jenis_industri'] = self.best_industry
            test_prob = self.pipeline.predict_proba(pd.DataFrame([test_data]))[0, 1]
            gain = test_prob - base_prob

            improvements.append({
                'feature': 'jenis_industri',
                'current_value': current_industry,
                'optimal_value': self.best_industry,
                'potential_gain': round(gain * 100, 2),
                'priority': 1 if gain > 0.15 else 2 if gain > 0.08 else 3,
                'suggestion': self._get_industry_suggestion(current_industry, gain)
            })

        # Analyze numeric features
        for feat in self.numeric_features:
            current_val = current[feat]
            optimal_val = self.optimal_values[feat]

            test_data = current.copy()
            test_data[feat] = optimal_val
            test_prob = self.pipeline.predict_proba(pd.DataFrame([test_data]))[0, 1]
            gain = test_prob - base_prob

            if gain > 0.01:
                improvements.append({
                    'feature': feat,
                    'current_value': round(current_val, 2),
                    'optimal_value': optimal_val,
                    'potential_gain': round(gain * 100, 2),
                    'priority': 1 if gain > 0.12 else 2 if gain > 0.06 else 3,
                    'suggestion': self._get_numeric_suggestion(feat, current_val, optimal_val, gain)
                })

        # Sort by potential gain (descending)
        improvements.sort(key=lambda x: x['potential_gain'], reverse=True)

        # Combined potential
        optimal_data = current.copy()
        optimal_data['jenis_industri'] = self.best_industry
        for feat in self.numeric_features:
            optimal_data[feat] = self.optimal_values[feat]
        optimal_prob = self.pipeline.predict_proba(pd.DataFrame([optimal_data]))[0, 1]

        return {
            **result,
            'improvements_needed': True,
            'suggestions': [imp['suggestion'] for imp in improvements],
            'detailed_improvements': improvements,
            'combined_potential': round(optimal_prob * 100, 2),
            'current_score': round(base_prob * 100, 2)
        }

    def _get_industry_suggestion(self, current, gain):
        if current == 'Fashion':
            return f"🔄 PIVOT INDUSTRI: Fashion sulit di pasar mahasiswa Jatinangor. Pertimbangkan beralih ke Kuliner (potensi naik {gain*100:.1f}%) atau tambahkan layanan delivery/makanan."
        elif current == 'Kerajinan':
            return f"🔄 PIVOT INDUSTRI: Kerajinan memiliki pasar sempit di Jatinangor. Beralih ke Kuliner (potensi naik {gain*100:.1f}%) atau Jasa (laundry, print) lebih menjanjikan."
        elif current == 'Retail':
            return f"🔄 DIVERSIFIKASI: Tambahkan produk makanan/minuman ke retail Anda. Kuliner murni memiliki survival {gain*100:.1f}% lebih tinggi di pasar mahasiswa."
        elif current == 'Jasa':
            return f"💡 OPTIMASI: Jasa Anda sudah cukup baik, tetapi Kuliner memiliki survival {gain*100:.1f}% lebih tinggi. Pertimbangkan add-on makanan jika memungkinkan."
        else:
            return f"✅ Jenis industri Anda (Kuliner) sudah optimal untuk pasar Jatinangor."

    def _get_numeric_suggestion(self, feat, current, optimal, gain):
        if feat == 'jarak_pemukiman':
            if current > 1500:
                return f"📍 LOKASI KRITIS: Jarak {current:.0f}m terlalu jauh dari pemukiman/kos. Pindah ke lokasi ~{optimal}m dari area kos mahasiswa (potensi naik {gain*100:.1f}%)."
            elif current > 800:
                return f"📍 LOKASI: Jarak {current:.0f}m kurang ideal. Dekatkan ke {optimal}m dari kos/kampus (potensi naik {gain*100:.1f}%)."
            else:
                return f"📍 LOKASI: Jarak {current:.0f}m sebenarnya dekat, tetapi mungkin terlalu dekat (biaya sewa tinggi). Optimal ~{optimal}m (potensi naik {gain*100:.1f}%)."

        elif feat == 'kapasitas_kursi':
            if current < 15:
                return f"🪑 KAPASITAS: {current:.0f} kursi terlalu sedikit untuk Kuliner. Tambah menjadi ~{optimal} kursi untuk menampung grup mahasiswa (potensi naik {gain*100:.1f}%)."
            elif current > 60:
                return f"🪑 KAPASITAS: {current:.0f} kursi terlalu besar untuk Jatinangor. Kurangi menjadi ~{optimal} kursi atau fokuskan pada takeaway (potensi naik {gain*100:.1f}%)."
            else:
                return f"🪑 KAPASITAS: {current:.0f} kursi kurang optimal. Sesuaikan menjadi ~{optimal} kursi untuk efisiensi operasional (potensi naik {gain*100:.1f}%)."

        elif feat == 'target_harga':
            if current > 80:
                return f"💰 HARGA: Rp {current:.0f} ribu terlalu mahal untuk budget mahasiswa Jatinangor. Turunkan ke Rp {optimal} ribu atau buat paket hemat (potensi naik {gain*100:.1f}%)."
            else:
                return f"💰 HARGA: Rp {current:.0f} ribu masih bisa dioptimalkan. Target Rp {optimal} ribu untuk daya tarik mahasiswa (potensi naik {gain*100:.1f}%)."

        elif feat == 'modal_awal':
            if current < 20:
                return f"💵 MODAL: Rp {current:.0f} juta terlalu kecil untuk kualitas yang kompetitif. Naikkan ke Rp {optimal} juta untuk peralatan & bahan baku (potensi naik {gain*100:.1f}%)."
            elif current > 200:
                return f"💵 MODAL: Rp {current:.0f} juta terlalu besar, tekanan ROI tinggi. Model bisnis leaner ~Rp {optimal} juta lebih aman (potensi naik {gain*100:.1f}%)."
            else:
                return f"💵 MODAL: Rp {current:.0f} juta bisa dioptimalkan. Rp {optimal} juta adalah sweet spot untuk UMKM di Jatinangor (potensi naik {gain*100:.1f}%)."

        return f"{feat}: ubah dari {current} ke {optimal} (potensi naik {gain*100:.1f}%)."


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UMKM SURVIVAL PREDICTOR - JATINANGOR, SUMEDANG")
    print("=" * 70)

    # Step 1: Generate / Load Data
    print("\n[1] Loading data...")
    df = generate_synthetic_data(n_samples=1200)
    print(f"    Dataset: {df.shape[0]} UMKM records")
    print(f"    Survival rate: {df['survival'].mean()*100:.1f}%")

    # Step 2: Train Model
    print("\n[2] Training Logistic Regression model...")
    pipeline, cat_features, num_features = build_model(df)

    # Step 3: Initialize Analyzer
    print("\n[3] Initializing analyzer...")
    analyzer = UMKMSurvivalAnalyzer(pipeline, cat_features, num_features)

    # Step 4: Test Cases
    print("\n[4] Running test predictions...")
    print("=" * 70)

    test_cases = [
        {
            'name': 'Warteg Dekat Kampus',
            'data': {
                'jenis_industri': 'Kuliner',
                'jarak_pemukiman': 350,
                'kapasitas_kursi': 24,
                'target_harga': 25,
                'modal_awal': 45
            }
        },
        {
            'name': 'Boutique Fashion Jauh dari Kos',
            'data': {
                'jenis_industri': 'Fashion',
                'jarak_pemukiman': 2800,
                'kapasitas_kursi': 80,
                'target_harga': 150,
                'modal_awal': 300
            }
        },
        {
            'name': 'Cafe Mahasiswa Harga Sedang',
            'data': {
                'jenis_industri': 'Kuliner',
                'jarak_pemukiman': 1200,
                'kapasitas_kursi': 45,
                'target_harga': 65,
                'modal_awal': 80
            }
        }
    ]

    for case in test_cases:
        print(f"\n🏪 {case['name']}")
        print("-" * 50)

        result = analyzer.analyze_improvements(case['data'])

        print(f"📊 Survival Score: {result['score_percent']}%")
        print(f"📈 Classification: {result['classification']}")

        if result['improvements_needed']:
            print(f"\n⚠️  IMPROVEMENTS NEEDED:")
            for i, sug in enumerate(result['suggestions'], 1):
                print(f"   {i}. {sug}")
            print(f"\n🎯 Combined Potential (if all optimized): {result['combined_potential']}%")
        else:
            print(f"\n✅ {result['suggestions'][0]}")

        print("=" * 70)

    # Step 5: Interactive Input
    print("\n[5] INTERACTIVE MODE")
    print("-" * 50)
    print("Enter your UMKM details below:")

    try:
        industry = input("Jenis industri (Kuliner/Fashion/Jasa/Retail/Kerajinan): ")
        jarak = float(input("Jarak dari pemukiman (meter): "))
        kursi = int(input("Kapasitas kursi: "))
        harga = float(input("Target harga (ribu rupiah): "))
        modal = float(input("Modal awal (juta rupiah): "))

        user_input = {
            'jenis_industri': industry,
            'jarak_pemukiman': jarak,
            'kapasitas_kursi': kursi,
            'target_harga': harga,
            'modal_awal': modal
        }

        result = analyzer.analyze_improvements(user_input)

        print(f"\n📊 YOUR SURVIVAL SCORE: {result['score_percent']}%")
        print(f"📈 Classification: {result['classification']}")

        if result['improvements_needed']:
            print(f"\n⚠️  IMPROVEMENTS NEEDED:")
            for i, sug in enumerate(result['suggestions'], 1):
                print(f"   {i}. {sug}")
            print(f"\n🎯 Combined Potential (if all optimized): {result['combined_potential']}%")
        else:
            print(f"\n✅ {result['suggestions'][0]}")

    except KeyboardInterrupt:
        print("\n\nExiting interactive mode.")
    except Exception as e:
        print(f"\nError: {e}")

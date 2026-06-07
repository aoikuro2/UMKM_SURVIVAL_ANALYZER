#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
UMKM SURVIVAL PREDICTOR - STREAMLIT WEB APP
Jatinangor, Sumedang, Jawa Barat, Indonesia
================================================================================

A user-friendly web interface for predicting UMKM survival probability
using a trained Logistic Regression model.

HOW TO RUN:
-----------
1. Install dependencies:
   pip install streamlit pandas numpy scikit-learn joblib

2. Make sure these 3 model files are in the SAME folder as this script:
   - umkm_survival_model.pkl
   - umkm_scaler.pkl
   - umkm_encoder.pkl

3. Run the app:
   streamlit run umkm_survival_app.py

4. Open your browser at http://localhost:8501
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="UMKM Survival Predictor - Jatinangor",
    page_icon="🏪",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS FOR BETTER LOOKS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-survival {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .low-survival {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .recommendation-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================

@st.cache_resource
def load_model_artifacts():
    """
    Load the trained model, scaler, and label encoder.
    Cached so it only loads once per session.
    """
    try:
        model = load('umkm_survival_model.pkl')
        scaler = load('umkm_scaler.pkl')
        encoder = load('umkm_encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError:
        st.error("❌ Model files not found! Make sure these files are in the same folder:")
        st.code("umkm_survival_model.pkl\numkm_scaler.pkl\numkm_encoder.pkl")
        st.stop()

model, scaler, encoder = load_model_artifacts()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def predict_survival(industri, jarak, harga, biaya, lama):
    """
    Core prediction logic.
    Returns dict with probability, classification, and recommendations.
    """
    # Encode industry
    industry_encoded = encoder.transform([industri])[0]

    # Prepare input
    input_data = np.array([[industry_encoded, jarak, harga, biaya, lama]])
    input_scaled = scaler.transform(input_data)

    # Predict
    probability = model.predict_proba(input_scaled)[0, 1]
    prediction = model.predict(input_scaled)[0]

    result = {
        'probability': round(probability * 100, 2),
        'classification': 'HIGH SURVIVAL' if prediction == 1 else 'LOW SURVIVAL',
        'is_high': prediction == 1,
        'inputs': {
            'jenis_industri': industri,
            'jarak_pemukiman_km': jarak,
            'range_harga': harga,
            'biaya_operasional_juta': biaya,
            'lama_usaha_tahun': lama
        }
    }

    # Generate recommendations for low survival
    if prediction == 0:
        recommendations = []

        if jarak > 3.0:
            recommendations.append({
                'priority': 'HIGH' if jarak > 6.0 else 'MEDIUM',
                'feature': '📍 Jarak dari Area Pemukiman',
                'current': f'{jarak} km',
                'target': '< 3.0 km',
                'action': 'Pindahkan atau buka cabang lebih dekat pemukiman/area kampus untuk meningkatkan foot traffic.',
                'why': 'Jarak jauh dari pemukiman secara signifikan menurunkan peluang bertahan.'
            })

        if harga == 1:
            recommendations.append({
                'priority': 'MEDIUM',
                'feature': '💰 Range Harga',
                'current': 'Rendah (1)',
                'target': 'Menengah (2)',
                'action': 'Naikkan harga sedikit dengan meningkatkan kualitas produk/layanan atau branding.',
                'why': 'Harga terlalu rendah menyebabkan margin tipis dan sulit menutup biaya operasional.'
            })
        elif harga == 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'feature': '💰 Range Harga',
                'current': 'Tinggi (3)',
                'target': 'Menengah (2)',
                'action': 'Pertimbangkan menurunkan harga atau buat varian produk dengan harga menengah untuk pasar mahasiswa Jatinangor.',
                'why': 'Harga tinggi di pasar Jatinangor (dominan mahasiswa) membatasi jumlah pelanggan.'
            })

        if biaya > 25:
            recommendations.append({
                'priority': 'HIGH' if biaya > 35 else 'MEDIUM',
                'feature': '💸 Biaya Operasional',
                'current': f'Rp {biaya:.1f} Juta/bulan',
                'target': '< Rp 25 Juta/bulan',
                'action': 'Efisiensikan biaya: negosiasi sewa, kurangi pegawai sementara, gunakan teknologi untuk otomasi.',
                'why': 'Biaya operasional tinggi membebani arus kas, terutama untuk UMKM dengan pendapatan fluktuatif.'
            })

        if lama < 2:
            recommendations.append({
                'priority': 'HIGH',
                'feature': '⏳ Lama Usaha',
                'current': f'{lama} tahun',
                'target': 'Minimal 2 tahun',
                'action': 'Fokus pada survival jangka pendek: pertahankan modal kerja, jangan ekspansi dulu, bangun loyalitas pelanggan.',
                'why': '2 tahun pertama adalah periode paling kritis untuk UMKM. Prioritaskan cash flow positif.'
            })

        if industri in ['Fashion', 'Elektronik']:
            recommendations.append({
                'priority': 'MEDIUM',
                'feature': '🏭 Jenis Industri',
                'current': industri,
                'target': 'Diversifikasi atau pivot',
                'action': f'Industri {industri} sangat kompetitif di Jatinangor. Pertimbangkan diversifikasi produk atau pivot ke Kuliner/Kerajinan.',
                'why': f'{industri} memiliki tingkat kegagalan lebih tinggi di pasar Jatinangor karena persaingan ketat.'
            })

        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        result['recommendations'] = recommendations
        result['total_recommendations'] = len(recommendations)

    return result


def get_probability_color(prob):
    """Return color based on probability score."""
    if prob >= 70:
        return "#28a745"  # Green
    elif prob >= 50:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def get_gauge_emoji(prob):
    """Return emoji based on probability."""
    if prob >= 70:
        return "🟢"
    elif prob >= 50:
        return "🟡"
    else:
        return "🔴"


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📊 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memprediksi peluang kelangsungan hidup UMKM di **Jatinangor, Sumedang** 
    menggunakan model **Logistic Regression**.

    **5 Input yang Dianalisis:**
    1. Jenis Industri
    2. Jarak dari Pemukiman
    3. Range Harga
    4. Biaya Operasional
    5. Lama Usaha

    **Output:**
    - Skor probabilitas bertahan (0-100%)
    - Klasifikasi: High/Low Survival
    - Rekomendasi perbaikan (jika skor ≤50%)
    """)

    st.markdown("---")
    st.markdown("### 🏆 Model Performance")
    st.markdown("""
    - **Akurasi:** 80.5%
    - **AUC-ROC:** 0.88
    - **Algoritma:** Logistic Regression
    - **Data:** 1,000 synthetic UMKM records
    """)

    st.markdown("---")
    st.markdown("*Developed for UMKM Jatinangor*")


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<div class="main-header">🏪 UMKM Survival Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediksi Kelangsungan Usaha Mikro Kecil Menengah di Jatinangor, Sumedang</div>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# INPUT FORM
# =============================================================================

st.markdown("### 📝 Masukkan Data UMKM Anda")

col1, col2 = st.columns(2)

with col1:
    jenis_industri = st.selectbox(
        "Jenis Industri *",
        options=list(encoder.classes_),
        help="Pilih jenis industri UMKM Anda"
    )

    jarak_pemukiman = st.slider(
        "Jarak dari Area Pemukiman (km) *",
        min_value=0.1,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Semakin dekat dengan pemukiman/kampus, semakin baik"
    )

    range_harga = st.selectbox(
        "Range Harga Produk/Layanan *",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 - Rendah", 2: "2 - Menengah (Optimal)", 3: "3 - Tinggi"}[x],
        help="Menengah (2) adalah sweet spot untuk pasar Jatinangor"
    )

with col2:
    biaya_operasional = st.number_input(
        "Biaya Operasional per Bulan (Juta Rupiah) *",
        min_value=1.0,
        max_value=50.0,
        value=15.0,
        step=0.5,
        help="Total biaya sewa, gaji, bahan baku, listrik, dll per bulan"
    )

    lama_usaha = st.number_input(
        "Lama Usaha Beroperasi (Tahun) *",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        help="Berapa tahun usaha ini sudah berjalan"
    )

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PREDICT BUTTON
# =============================================================================

center_col = st.columns([1, 2, 1])[1]
with center_col:
    predict_btn = st.button(
        "🔮 Prediksi Kelangsungan Usaha",
        type="primary",
        use_container_width=True
    )

st.markdown("---")

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

if predict_btn:
    with st.spinner("Menganalisis data UMKM Anda..."):
        result = predict_survival(
            jenis_industri, 
            jarak_pemukiman, 
            range_harga, 
            biaya_operasional, 
            lama_usaha
        )

    prob = result['probability']
    color = get_probability_color(prob)
    emoji = get_gauge_emoji(prob)

    # Result Header
    st.markdown(f"### {emoji} Hasil Prediksi")

    # Metrics row
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            label="Probabilitas Bertahan",
            value=f"{prob}%",
            delta=None
        )

    with m2:
        status_color = "🟢 HIGH" if result['is_high'] else "🔴 LOW"
        st.metric(
            label="Klasifikasi",
            value=status_color,
            delta=None
        )

    with m3:
        risk_level = "Rendah" if prob >= 70 else "Sedang" if prob >= 50 else "Tinggi"
        st.metric(
            label="Tingkat Risiko Gagal",
            value=risk_level,
            delta=None
        )

    # Progress bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(prob / 100, text=f"Skor Kelangsungan: {prob}%")

    # Result box
    if result['is_high']:
        st.markdown(f"""
        <div class="result-box high-survival">
            <h4>✅ {result['classification']}</h4>
            <p>Usaha Anda memiliki peluang bertahan <strong>tinggi</strong> (>50%). 
            Pertahankan strategi yang sudah berjalan dan terus evaluasi performa secara berkala.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box low-survival">
            <h4>⚠️ {result['classification']}</h4>
            <p>Usaha Anda memiliki risiko gagal <strong>tinggi</strong> (≤50%). 
            Silakan tinjau rekomendasi perbaikan di bawah ini untuk meningkatkan peluang bertahan.</p>
        </div>
        """, unsafe_allow_html=True)

    # =============================================================================
    # INPUT SUMMARY
    # =============================================================================

    with st.expander("📋 Ringkasan Input Data"):
        inputs = result['inputs']
        harga_label = {1: "Rendah", 2: "Menengah", 3: "Tinggi"}[inputs['range_harga']]

        summary_data = {
            'Parameter': ['Jenis Industri', 'Jarak Pemukiman', 'Range Harga', 'Biaya Operasional', 'Lama Usaha'],
            'Nilai': [
                inputs['jenis_industri'],
                f"{inputs['jarak_pemukiman_km']} km",
                f"{inputs['range_harga']} ({harga_label})",
                f"Rp {inputs['biaya_operasional_juta']:.1f} Juta/bulan",
                f"{inputs['lama_usaha_tahun']} tahun"
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # =============================================================================
    # RECOMMENDATIONS (Only for Low Survival)
    # =============================================================================

    if not result['is_high'] and 'recommendations' in result:
        st.markdown("---")
        st.markdown(f"### 🛠️ Rekomendasi Perbaikan ({result['total_recommendations']} item)")
        st.markdown("Berikut adalah faktor-faktor yang perlu diperbaiki untuk meningkatkan peluang kelangsungan usaha Anda:")

        for i, rec in enumerate(result['recommendations'], 1):
            priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[rec['priority']]

            st.markdown(f"""
            <div class="recommendation-card">
                <h5>{priority_color} {i}. [{rec['priority']}] {rec['feature']}</h5>
                <p><strong>Saat ini:</strong> {rec['current']} → <strong>Target:</strong> {rec['target']}</p>
                <p><strong>💡 Tindakan:</strong> {rec['action']}</p>
                <p><small><strong>📌 Mengapa penting:</strong> {rec['why']}</small></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("💡 **Tip:** Prioritaskan perbaikan dengan label **[HIGH]** terlebih dahulu karena memiliki dampak terbesar pada kelangsungan usaha Anda.")

    # =============================================================================
    # FEATURE IMPACT ANALYSIS
    # =============================================================================

    st.markdown("---")
    st.markdown("### 📊 Analisis Dampak Setiap Faktor")
    st.markdown("""
    Berdasarkan model yang telah dilatih, berikut adalah faktor-faktor yang paling mempengaruhi 
    kelangsungan UMKM di Jatinangor (diurutkan dari yang paling berpengaruh):
    """)

    impact_data = {
        'Faktor': [
            '📍 Jarak dari Pemukiman',
            '💸 Biaya Operasional',
            '⏳ Lama Usaha',
            '🏭 Jenis Industri',
            '💰 Range Harga'
        ],
        'Dampak': [
            'Sangat Negatif (-0.95)',
            'Negatif (-0.65)',
            'Positif (+0.57)',
            'Positif (+0.74)',
            'Positif (+0.20)'
        ],
        'Artinya': [
            'Semakin JAUH = semakin BERBAHAYA',
            'Semakin TINGGI = semakin BERBAHAYA',
            'Semakin LAMA = semakin AMAN',
            'Kuliner/Pertanian = lebih AMAN',
            'Menengah (2) = paling OPTIMAL'
        ]
    }

    st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <h5>🎯 Kesimpulan Analisis</h5>
        <p>Faktor <strong>jarak dari pemukiman</strong> adalah yang paling kritis. UMKM di Jatinangor 
        yang berlokasi dekat dengan pemukiman padat mahasiswa (sekitar kampus UNPAD/ITB) memiliki 
        peluang bertahan jauh lebih tinggi karena foot traffic dan aksesibilitas yang lebih baik.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Initial state - show instructions
    st.info("👆 **Silakan isi data UMKM Anda di atas, lalu klik tombol 'Prediksi Kelangsungan Usaha' untuk melihat hasil.**")

    st.markdown("---")
    st.markdown("### 📖 Panduan Pengisian")

    guide_col1, guide_col2 = st.columns(2)

    with guide_col1:
        st.markdown("""
        **Jenis Industri**
        - **Kuliner** 🍜: Paling aman (demand mahasiswa tinggi)
        - **Pertanian** 🌾: Baik (basis lokal kuat)
        - **Kerajinan** 🎨: Cukup baik (produk unik)
        - **Jasa** 🛠️: Netral
        - **Fashion** 👗: Berisiko (persaingan ketat)
        - **Elektronik** 📱: Paling berisiko (modal besar)

        **Jarak dari Pemukiman**
        - 0-3 km: Ideal ✅
        - 3-6 km: Cukup ⚠️
        - 6-10 km: Berisiko ❌
        """)

    with guide_col2:
        st.markdown("""
        **Range Harga**
        - **1 (Rendah)**: Margin tipis, sulit profit
        - **2 (Menengah)**: 🎯 Sweet spot untuk Jatinangor
        - **3 (Tinggi)**: Market terbatas (mahasiswa)

        **Biaya Operasional**
        - < Rp 15 Jt: Sangat ideal ✅
        - Rp 15-25 Jt: Normal ⚠️
        - > Rp 25 Jt: Berisiko ❌

        **Lama Usaha**
        - < 2 tahun: Periode kritis 🚨
        - 2-5 tahun: Mulai stabil
        - > 5 tahun: Cukup aman ✅
        """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <p>UMKM Survival Predictor | Jatinangor, Sumedang, Jawa Barat | Powered by Logistic Regression</p>
    <p>Model Accuracy: 80.5% | AUC-ROC: 0.88 | 1,000 Training Samples</p>
</div>
""", unsafe_allow_html=True)

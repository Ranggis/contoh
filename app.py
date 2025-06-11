import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime

# Load model dan encoder dari file .pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_gender = encoders['Gender']
le_membership = encoders['Membership_Status']

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Tipe Pelanggan", page_icon="☕", layout="centered")

# Tema terang/gelap
mode = st.sidebar.selectbox("🌜 Pilih Tema", ["🌝 Terang", "🌚 Gelap"])

if mode == "🌚 Gelap":
    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        .custom-footer {
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #888;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .custom-footer {
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #555;
        }
        </style>
    """, unsafe_allow_html=True)

# Sidebar navigasi dan bahasa
st.sidebar.title("📚 Menu")
page = st.sidebar.radio("Navigasi", ["Prediksi Individu", "Prediksi Massal", "Aturan", "Tentang"])
language = st.sidebar.radio("🌐 Bahasa", ["🇮🇩 Indonesia", "🇬🇧 English"])

st.sidebar.markdown(
    """
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .custom-button {
        background-color: #ff4b4b;
        color: white;
        padding: 12px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .custom-button:hover {
        background-color: #ff1f1f;
        transform: scale(1.05);
    }
    </style>

    <div class="button-container">
        <a href="https://ranggis.netlify.app/" target="_blank">
            <button class="custom-button">☕ Kembali ke Website</button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# Fungsi translate sederhana
def tr(id_text, en_text):
    return id_text if language == "🇮🇩 Indonesia" else en_text

if page == "Prediksi Individu":
    st.title("☕ " + tr("Prediksi Tipe Pelanggan Coffee Shop", "Customer Type Prediction for Coffee Shop"))
    st.markdown(tr("Masukkan data pelanggan untuk mengetahui tipe pelanggan.",
                  "Input customer data to classify their type."))

    with st.form("form_prediksi"):
        visits = st.number_input(tr("🔁 Frekuensi Kunjungan per Bulan", "🔁 Visit Frequency per Month"), 0, 30, 5)
        spending = st.number_input(tr("💸 Pengeluaran per Kunjungan (Rp)", "💸 Spending per Visit (Rp)"), 0, step=1000, value=75000)
        time_spent = st.slider(tr("🕒 Rata-rata Waktu di Cafe (menit)", "🕒 Average Time in Cafe (min)"), 0, 300, 90)
        gender = st.selectbox(tr("👤 Jenis Kelamin", "👤 Gender"), le_gender.classes_)
        membership = st.selectbox(tr("🎫 Status Membership", "🎫 Membership Status"), le_membership.classes_)
        submitted = st.form_submit_button(tr("🚀 Prediksi", "🚀 Predict"))

    if submitted:
        with st.spinner(tr("Memprediksi kategori pelanggan...", "Predicting customer type...")):
            time.sleep(1)

        input_df = pd.DataFrame({
            "Age": [25],
            "Gender_Encoded": [le_gender.transform([gender])[0]],
            "Visit_Frequency": [visits],
            "Spending_per_Visit": [spending],
            "Time_Spent_in_Cafe": [time_spent],
            "Membership_Status_Encoded": [le_membership.transform([membership])[0]],
        })

        input_df = input_df[['Age', 'Gender_Encoded', 'Visit_Frequency',
                             'Spending_per_Visit', 'Time_Spent_in_Cafe',
                             'Membership_Status_Encoded']]

        proba = model.predict_proba(input_df.values)[0]
        pred = model.classes_[np.argmax(proba)]

        score = min(100, int((visits * 2 + spending / 2000 + time_spent / 2)))

        st.subheader("🔍 " + tr("Hasil Prediksi", "Prediction Result"))
        avatar_url = ""

        if pred == "Royal":
            st.success("🏆 " + tr("Pelanggan ini termasuk kategori Royal!", "This customer is Royal!"))
            avatar_url = "image/kiki.jpg"
            st.balloons()
        elif pred == "Inactive":
            st.warning("💤 " + tr("Pelanggan ini termasuk kategori Inactive.", "This customer is Inactive."))
            avatar_url = "image/Amu.jpg"
        else:
            st.info("🌱 " + tr("Pelanggan ini termasuk kategori New.", "This customer is New."))
            avatar_url = "image/upi.jpg"

        st.image(avatar_url, caption=tr("Kartu Pelanggan", "Customer Avatar"), width=200)
        st.metric("📈 Loyalty Score", f"{score}/100")

        st.subheader("📊 " + tr("Probabilitas Tiap Kategori", "Category Probabilities"))
        proba_df = pd.DataFrame({
            "Kategori": model.classes_,
            "Probabilitas": np.round(proba, 2)
        })
        st.bar_chart(proba_df.set_index("Kategori"))

        st.subheader("💡 " + tr("Rekomendasi", "Recommendation"))
        if pred == "Inactive":
            st.info(tr("📌 Saran: Kirim kupon diskon agar pelanggan kembali.", "📌 Tip: Send discount coupons to re-engage the customer."))
        elif pred == "New":
            st.info(tr("📌 Saran: Tawarkan membership eksklusif.", "📌 Tip: Offer exclusive membership benefits."))
        elif pred == "Royal":
            st.info(tr("📌 Saran: Beri reward poin atau akses VIP.", "📌 Tip: Provide loyalty rewards or VIP access."))
    

elif page == "Prediksi Massal":
    st.title(tr("📂 Prediksi Massal dari File CSV", "📂 Batch Prediction from CSV"))
    uploaded = st.file_uploader(tr("Unggah file CSV dengan kolom yang dibutuhkan:", "Upload a CSV file with required columns:"), type=["csv"])
    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)

            st.subheader(tr("📋 Pratinjau Data", "📋 Data Preview"))
            st.dataframe(df_batch.head())

            # Daftar kolom yang dibutuhkan
            required_columns = ['Age', 'Gender', 'Visit_Frequency', 'Spending_per_Visit', 'Time_Spent_in_Cafe', 'Membership_Status']
            missing_columns = [col for col in required_columns if col not in df_batch.columns]

            st.subheader(tr("✅ Validasi Format CSV", "✅ CSV Format Validation"))

            # Checklist validasi kolom
            for col in required_columns:
                if col in df_batch.columns:
                    st.markdown(f"✅ {col}")
                else:
                    st.markdown(f"❌ **{col}** {tr('tidak ditemukan.', 'not found.')}")

            if missing_columns:
                st.error(tr("Beberapa kolom penting tidak ada. Silakan periksa kembali file CSV Anda.",
                            "Some required columns are missing. Please recheck your CSV file."))
            else:
                try:
                    # Cek isi kolom Gender & Membership
                    invalid_gender = df_batch[~df_batch['Gender'].isin(le_gender.classes_)]
                    invalid_membership = df_batch[~df_batch['Membership_Status'].isin(le_membership.classes_)]

                    if not invalid_gender.empty or not invalid_membership.empty:
                        st.warning(tr("Beberapa baris memiliki nilai yang tidak dikenali.", 
                                      "Some rows contain unrecognized values."))

                        if not invalid_gender.empty:
                            st.error(tr(f"Nilai tidak valid di kolom Gender:\n{invalid_gender['Gender'].unique()}",
                                        f"Invalid values in Gender column:\n{invalid_gender['Gender'].unique()}"))

                        if not invalid_membership.empty:
                            st.error(tr(f"Nilai tidak valid di kolom Membership_Status:\n{invalid_membership['Membership_Status'].unique()}",
                                        f"Invalid values in Membership_Status column:\n{invalid_membership['Membership_Status'].unique()}"))
                    else:
                        # Lanjut jika semua valid
                        df_batch['Gender_Encoded'] = le_gender.transform(df_batch['Gender'])
                        df_batch['Membership_Status_Encoded'] = le_membership.transform(df_batch['Membership_Status'])

                        X_batch = df_batch[['Age', 'Gender_Encoded', 'Visit_Frequency',
                                            'Spending_per_Visit', 'Time_Spent_in_Cafe',
                                            'Membership_Status_Encoded']]

                        df_batch['Prediksi'] = model.predict(X_batch.values)
                        df_batch['Probabilitas_Tertinggi'] = model.predict_proba(X_batch.values).max(axis=1).round(2)

                        st.success(tr("✅ Semua baris berhasil diprediksi!", "✅ All rows successfully predicted!"))
                        st.dataframe(df_batch.head())

                        csv_result = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(label=tr("⬇️ Unduh Hasil Prediksi", "⬇️ Download Prediction Results"),
                                           data=csv_result,
                                           file_name="hasil_prediksi.csv",
                                           mime="text/csv")
                except Exception as e:
                    st.error(tr(f"Gagal memproses data: {e}", f"Failed to process data: {e}"))

        except Exception as e:
            st.error(tr(f"Gagal membaca file: {e}", f"Failed to read file: {e}"))

elif page == "Aturan":
    st.title("📖 " + tr("Aturan Klasifikasi Pelanggan", "Customer Classification Rules"))
    st.markdown(tr("""
    ### 🏆 Royal:
    - Visit_Frequency ≥ 10
    - Spending_per_Visit ≥ 150000
    - Time_Spent_in_Cafe ≥ 120
    - Membership_Status = Yes

    ### 💤 Inactive:
    - Visit_Frequency ≤ 2
    - Spending_per_Visit < 35000
    - Time_Spent_in_Cafe < 40
    - Membership_Status = No

    ### 🌱 New:
    - Semua kombinasi lain di luar Royal & Inactive
    """,
    """
    ### 🏆 Royal:
    - Visit_Frequency ≥ 10
    - Spending_per_Visit ≥ 150000
    - Time_Spent_in_Cafe ≥ 120
    - Membership_Status = Yes

    ### 💤 Inactive:
    - Visit_Frequency ≤ 2
    - Spending_per_Visit < 35000
    - Time_Spent_in_Cafe < 40
    - Membership_Status = No

    ### 🌱 New:
    - All other combinations outside Royal & Inactive
    """))

elif page == "Tentang":
    st.title("👩‍💻 " + tr("Tentang Aplikasi", "About the App"))
    st.markdown(tr("""
    Aplikasi ini dibuat untuk memprediksi tipe pelanggan pada sebuah coffee shop menggunakan model machine learning Random Forest.

    Fitur:
    - Prediksi individu dan massal
    - Skor loyalitas
    - Visualisasi probabilitas
    - Rekomendasi tindakan
    - Avatar kartu pelanggan
    - Navigasi multi-halaman & mode gelap
    - Dukungan bahasa Indonesia dan Inggris

    Dibuat oleh: **Kelompok 1**
    """,
    """
    This app predicts customer types in a coffee shop using a Random Forest machine learning model.

    Features:
    - Single and batch prediction
    - Loyalty scoring
    - Probability visualization
    - Action recommendations
    - Customer avatar card
    - Multi-page layout & dark mode
    - Language support: English & Indonesian

    Created by: **Kelompok 1**
    """))

# Footer
st.markdown(f"""
<div class='custom-footer'>
    © {datetime.now().year} Coffee Prediction App — All rights reserved.
</div>
""", unsafe_allow_html=True)
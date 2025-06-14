import streamlit as st
import pandas as pd
import joblib
import io

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    layout="wide",
    page_title="Prediktor Mahasiswa DO",
    page_icon="🎓"
)

# --- 2. Memuat Model ---
# Muat model sekali dan simpan dalam cache untuk seluruh sesi.
@st.cache_resource
def muat_model():
    """
    Memuat model machine learning yang sudah dilatih dari file joblib.
    Menangani FileNotFoundError dan menghentikan aplikasi jika model tidak ditemukan.
    """
    try:
        model = joblib.load('model_rf.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: File model 'model_rf.joblib' tidak ditemukan.")
        st.info("Pastikan file model berada di direktori yang sama dengan skrip ini.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

model = muat_model()

# --- 3. Definisi Fitur ---
# Kamus ini mendefinisikan semua fitur input, tipe, opsi, dan nilai defaultnya.
# Ini menjadi satu-satunya sumber acuan untuk membuat elemen UI dan mengurutkan kolom DataFrame.
DEFINISI_FITUR = {
    # Pribadi & Keuangan
    "Marital_status": {
        "label": "Status Pernikahan",
        "type": "selectbox",
        "options": {"Lajang": 1, "Menikah": 2, "Duda/Janda": 3, "Bercerai": 4, "Berpisah Resmi": 5},
        "default": 1,
        "help": "Status pernikahan mahasiswa."
    },
    "Gender": {
        "label": "Jenis Kelamin",
        "type": "selectbox",
        "options": {"Laki-laki": 1, "Perempuan": 0},
        "default": 1,
        "help": "Jenis kelamin mahasiswa."
    },
    "Age_at_enrollment": {
        "label": "Usia saat Pendaftaran",
        "type": "number_input", "min_value": 17, "max_value": 70, "default": 19, "step": 1,
        "help": "Usia mahasiswa saat pertama kali mendaftar."
    },
    "Displaced": {
        "label": "Mahasiswa Pindahan (Luar Daerah)",
        "type": "selectbox",
        "options": {"Tidak": 0, "Ya": 1},
        "default": 0,
        "help": "Apakah mahasiswa berasal dari luar daerah/kota?"
    },
    "Debtor": {
        "label": "Memiliki Tunggakan?",
        "type": "selectbox",
        "options": {"Tidak": 0, "Ya": 1},
        "default": 0,
        "help": "Apakah mahasiswa memiliki tunggakan biaya kepada institusi?"
    },
    "Tuition_fees_up_to_date": {
        "label": "Uang Kuliah Lancar?",
        "type": "selectbox",
        "options": {"Ya": 1, "No": 0},
        "default": 1,
        "help": "Apakah pembayaran uang kuliah mahasiswa lancar/tepat waktu?"
    },
    "Scholarship_holder": {
        "label": "Penerima Beasiswa?",
        "type": "selectbox",
        "options": {"Tidak": 0, "Ya": 1},
        "default": 0,
        "help": "Apakah mahasiswa merupakan penerima beasiswa?"
    },
    # Latar Belakang Akademik
    "Application_mode": {
        "label": "Jalur Pendaftaran",
        "type": "selectbox",
        "options": {
            "Gelombang 1 - Kuota Umum": 1, "Gelombang 2 - Kuota Umum": 17, "Gelombang 3 - Kuota Umum": 18,
            "Lulusan Kursus Tinggi Lain": 7, "Di Atas 23 Tahun": 39, "Transfer": 42, "Pindah Jurusan": 43,
            "Mahasiswa Internasional (S1)": 15, "Gelombang 1 - Kuota Khusus (Pulau Azores)": 5,
            "Gelombang 1 - Kuota Khusus (Pulau Madeira)": 16, "Ordonansi No. 612/93": 2, "Ordonansi No. 854-B/99": 10,
            "Ordonansi No. 533-A/99, Item B2 (Beda Rencana)": 26, "Ordonansi No. 533-A/99, Item B3 (Institusi Lain)": 27,
            "Lulusan Diploma Spesialisasi Teknologi": 44, "Pindah Institusi/Jurusan": 51, "Lulusan Diploma Siklus Pendek": 53,
            "Pindah Institusi/Jurusan (Internasional)": 57
        },
        "default": 1,
        "help": "Jalur atau tipe pendaftaran saat masuk."
    },
    "Previous_qualification_grade": {
        "label": "Nilai Kualifikasi Sebelumnya",
        "type": "number_input", "min_value": 0.0, "max_value": 200.0, "default": 130.0, "step": 0.1,
        "help": "Nilai dari kualifikasi sebelumnya (misal: Rata-rata UN/Ijazah SMA)."
    },
    "Admission_grade": {
        "label": "Nilai Penerimaan",
        "type": "number_input", "min_value": 0.0, "max_value": 200.0, "default": 125.0, "step": 0.1,
        "help": "Nilai tes masuk universitas."
    },
    # Kinerja Semester
    "Curricular_units_1st_sem_enrolled": {
        "label": "SKS Diambil Sem. 1",
        "type": "number_input", "min_value": 0, "max_value": 50, "default": 6, "step": 1,
        "help": "Jumlah unit kurikuler (SKS) yang diambil di semester 1."
    },
    "Curricular_units_1st_sem_approved": {
        "label": "SKS Lulus Sem. 1",
        "type": "number_input", "min_value": 0, "max_value": 50, "default": 5, "step": 1,
        "help": "Jumlah unit kurikuler (SKS) yang lulus di semester 1."
    },
    "Curricular_units_1st_sem_grade": {
        "label": "IPK Sem. 1",
        "type": "number_input", "min_value": 0.0, "max_value": 20.0, "default": 12.5, "step": 0.1,
        "help": "Nilai rata-rata (IPK) untuk semester 1 (skala 0-20)."
    },
    "Curricular_units_2nd_sem_enrolled": {
        "label": "SKS Diambil Sem. 2",
        "type": "number_input", "min_value": 0, "max_value": 50, "default": 6, "step": 1,
        "help": "Jumlah unit kurikuler (SKS) yang diambil di semester 2."
    },
    "Curricular_units_2nd_sem_evaluations": {
        "label": "Jumlah Evaluasi Sem. 2",
        "type": "number_input", "min_value": 0, "max_value": 50, "default": 8, "step": 1,
        "help": "Jumlah evaluasi (tes, tugas) di semester 2."
    },
    "Curricular_units_2nd_sem_approved": {
        "label": "SKS Lulus Sem. 2",
        "type": "number_input", "min_value": 0, "max_value": 50, "default": 5, "step": 1,
        "help": "Jumlah unit kurikuler (SKS) yang lulus di semester 2."
    },
    "Curricular_units_2nd_sem_grade": {
        "label": "IPK Sem. 2",
        "type": "number_input", "min_value": 0.0, "max_value": 20.0, "default": 12.0, "step": 0.1,
        "help": "Nilai rata-rata (IPK) untuk semester 2 (skala 0-20)."
    },
    "Curricular_units_2nd_sem_without_evaluations": {
        "label": "SKS Tanpa Evaluasi Sem. 2",
        "type": "number_input", "min_value": 0, "max_value": 20, "default": 0, "step": 1,
        "help": "Jumlah SKS di semester 2 yang tidak ada evaluasinya."
    },
}

# Urutan kolom harus sama persis dengan urutan saat pelatihan model.
URUTAN_KOLOM_FITUR = list(DEFINISI_FITUR.keys())

# --- 4. Fungsi untuk Menampilkan Halaman ---

def tampilkan_halaman_beranda():
    """Menampilkan halaman selamat datang dan informasi."""
    st.title("🎓 Alat Prediksi Dropout Mahasiswa")
    st.markdown("""
        Selamat datang! Aplikasi ini menggunakan model machine learning *Random Forest* untuk memprediksi kemungkinan seorang mahasiswa akan *drop out* (DO). 
        Aplikasi ini dirancang untuk membantu dosen pembimbing akademik dan administrator mengidentifikasi mahasiswa berisiko dan memberikan dukungan tepat waktu.

        **Cara Penggunaan:**
        1.  **Prediksi Tunggal:** Buka tab ini untuk memasukkan data seorang mahasiswa dan dapatkan prediksi instan.
        2.  **Prediksi Massal:** Gunakan tab ini untuk mengunggah file CSV berisi data banyak mahasiswa dan dapatkan prediksi untuk semuanya sekaligus.

        Alat ini bersifat informasional dan sebaiknya digunakan sebagai bagian dari strategi dukungan mahasiswa yang komprehensif.
    """)
    st.markdown("---")
    st.info("Silakan pilih mode prediksi dari tab di atas untuk memulai.")

def tampilkan_halaman_prediksi_tunggal():
    """Menampilkan formulir untuk prediksi satu mahasiswa."""
    st.header("👤 Prediksi Mahasiswa Tunggal")
    st.markdown("Isi detail di bawah ini untuk memprediksi status *dropout* seorang mahasiswa.")

    data_input = {}

    # --- Formulir Input menggunakan Expander ---
    with st.expander("Informasi Pribadi & Keuangan", expanded=True):
        kol1, kol2 = st.columns(2)
        fitur_pribadi_keuangan = ["Marital_status", "Gender", "Age_at_enrollment", "Displaced", "Debtor", "Tuition_fees_up_to_date", "Scholarship_holder"]
        for i, fitur in enumerate(fitur_pribadi_keuangan):
            with kol1 if i % 2 == 0 else kol2:
                detail = DEFINISI_FITUR[fitur]
                if detail["type"] == "selectbox":
                    opsi_terpilih = st.selectbox(detail["label"], options=detail["options"].keys(), help=detail["help"])
                    data_input[fitur] = detail["options"][opsi_terpilih]
                elif detail["type"] == "number_input":
                    data_input[fitur] = st.number_input(detail["label"], min_value=detail["min_value"], max_value=detail["max_value"], value=detail["default"], step=detail.get("step", 1), help=detail["help"])

    with st.expander("Latar Belakang Akademik"):
        kol1, kol2 = st.columns(2)
        fitur_akademik = ["Application_mode", "Previous_qualification_grade", "Admission_grade"]
        for i, fitur in enumerate(fitur_akademik):
            with kol1 if i % 2 == 0 else kol2:
                detail = DEFINISI_FITUR[fitur]
                if detail["type"] == "selectbox":
                    indeks_default = list(detail["options"].values()).index(detail["default"])
                    opsi_terpilih = st.selectbox(detail["label"], options=detail["options"].keys(), index=indeks_default, help=detail["help"])
                    data_input[fitur] = detail["options"][opsi_terpilih]
                elif detail["type"] == "number_input":
                    data_input[fitur] = st.number_input(detail["label"], min_value=detail["min_value"], max_value=detail["max_value"], value=detail["default"], step=detail.get("step", 1), help=detail["help"])

    with st.expander("Kinerja per Semester"):
        kol1, kol2 = st.columns(2)
        fitur_sem1 = ["Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_grade"]
        fitur_sem2 = ["Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_without_evaluations"]
        with kol1:
            st.subheader("Semester 1")
            for fitur in fitur_sem1:
                detail = DEFINISI_FITUR[fitur]
                data_input[fitur] = st.number_input(detail["label"], min_value=detail["min_value"], max_value=detail["max_value"], value=detail["default"], step=detail.get("step", 1), help=detail["help"])
        with kol2:
            st.subheader("Semester 2")
            for fitur in fitur_sem2:
                detail = DEFINISI_FITUR[fitur]
                data_input[fitur] = st.number_input(detail["label"], min_value=detail["min_value"], max_value=detail["max_value"], value=detail["default"], step=detail.get("step", 1), help=detail["help"])

    st.markdown("---")

    if st.button("📈 Prediksi Status Mahasiswa", type="primary"):
        df_input = pd.DataFrame([data_input])
        # Pastikan urutan kolom sudah benar
        df_input = df_input[URUTAN_KOLOM_FITUR]

        try:
            prediksi = model.predict(df_input)
            prediksi_proba = model.predict_proba(df_input)
            
            st.subheader("📊 Hasil Prediksi")
            kol1, kol2 = st.columns(2)

            with kol1:
                if prediksi[0] == 0: # Asumsi 0 adalah Dropout
                    st.error("Status: Diprediksi akan **DROP OUT**")
                    st.metric(label="Probabilitas Dropout", value=f"{prediksi_proba[0][0]:.2%}")
                else: # Asumsi 1 adalah Lulus/Aktif
                    st.success("Status: Diprediksi akan **LULUS / TETAP AKTIF**")
                    st.metric(label="Probabilitas Lanjut", value=f"{prediksi_proba[0][1]:.2%}")

            with kol2:
                st.subheader("Analisis Faktor Risiko")
                ada_risiko = False
                if data_input.get("Debtor") == 1:
                    st.warning("🚩 Risiko Tinggi: Mahasiswa memiliki tunggakan.")
                    ada_risiko = True
                if data_input.get("Tuition_fees_up_to_date") == 0:
                    st.warning("🚩 Risiko Tinggi: Pembayaran uang kuliah tidak lancar.")
                    ada_risiko = True
                if data_input.get("Curricular_units_2nd_sem_approved", 0) < 3:
                     st.warning("🚩 Risiko Sedang: Jumlah SKS lulus di semester 2 rendah.")
                     ada_risiko = True
                if not ada_risiko:
                    st.info("Tidak ada faktor risiko tinggi yang terdeteksi secara eksplisit. Prediksi model didasarkan pada interaksi semua fitur.")

            with st.expander("Lihat Data yang Disubmit"):
                st.dataframe(df_input)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

def tampilkan_halaman_prediksi_batch():
    """Menampilkan halaman untuk unggah CSV dan prediksi massal."""
    st.header("📄 Prediksi Mahasiswa Massal (Batch)")
    st.markdown("Unggah file CSV berisi data mahasiswa untuk memprediksi status *dropout* beberapa mahasiswa sekaligus.")

    st.subheader("1. Unduh Template CSV")
    st.info("Pastikan file CSV Anda memiliki nama dan urutan kolom yang sama persis dengan template.")
    
    # Buat DataFrame contoh untuk template
    df_contoh = pd.DataFrame({kol: [detail["default"]] for kol, detail in DEFINISI_FITUR.items()})
    df_contoh = df_contoh[URUTAN_KOLOM_FITUR]
    csv_contoh = df_contoh.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Unduh Template CSV",
        data=csv_contoh,
        file_name="template_prediksi_mahasiswa.csv",
        mime="text/csv",
        help="Klik untuk mengunduh template dengan kolom yang benar."
    )

    st.markdown("---")
    st.subheader("2. Unggah File CSV Anda")
    file_diunggah = st.file_uploader(
        "Pilih sebuah file CSV",
        type=["csv"],
        help="Pastikan kolom sesuai template: " + ", ".join(URUTAN_KOLOM_FITUR)
    )

    if file_diunggah:
        try:
            df_input = pd.read_csv(file_diunggah)
            st.success("File berhasil diunggah!")
            st.write("Pratinjau Data:")
            st.dataframe(df_input.head())

            # Validasi kolom
            kolom_hilang = [kol for kol in URUTAN_KOLOM_FITUR if kol not in df_input.columns]
            if kolom_hilang:
                st.error(f"Kolom wajib berikut tidak ditemukan di file Anda: {', '.join(kolom_hilang)}")
                return

            st.subheader("3. Dapatkan Prediksi")
            if st.button("Jalankan Prediksi Massal", type="primary"):
                with st.spinner("Sedang memproses prediksi..."):
                    df_untuk_prediksi = df_input[URUTAN_KOLOM_FITUR]
                    
                    prediksi = model.predict(df_untuk_prediksi)
                    probabilitas = model.predict_proba(df_untuk_prediksi)

                    df_hasil = df_input.copy()
                    # Asumsi 0: Dropout, 1: Lulus/Aktif
                    df_hasil['Prediksi_Status'] = ['Dropout' if p == 0 else 'Lulus/Aktif' for p in prediksi]
                    df_hasil['Probabilitas_Dropout'] = [f"{proba[0]:.2%}" for proba in probabilitas]

                    st.subheader("4. Hasil Prediksi")
                    st.dataframe(df_hasil)

                    # Siapkan hasil untuk diunduh
                    csv_hasil = df_hasil.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh Hasil Prediksi (CSV)",
                        data=csv_hasil,
                        file_name="hasil_prediksi_dropout_mahasiswa.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.warning("Pastikan CSV Anda diformat dengan benar dan berisi data numerik yang valid.")

# --- 5. Logika Utama Aplikasi ---
# Gunakan tab untuk navigasi
tab_beranda, tab_prediksi_tunggal, tab_prediksi_batch = st.tabs(["🏠 Beranda", "👤 Prediksi Tunggal", "📄 Prediksi Massal"])

with tab_beranda:
    tampilkan_halaman_beranda()

with tab_prediksi_tunggal:
    tampilkan_halaman_prediksi_tunggal()

with tab_prediksi_batch:
    tampilkan_halaman_prediksi_batch()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("🛠️ Dibuat dengan **Streamlit**")
st.sidebar.caption("©Mochamad Zikri Abdilah")
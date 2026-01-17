import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import re
# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AI Tematik Analiz Aracı", layout="wide")

st.title("📊 AI Destekli Tematik Veri Analizi")
st.markdown("""
Bu araç, CSV dosyanızdaki verileri analiz eder, temaları ve alt temaları belirler, 
doğrudan alıntılar yapar ve 'Major' (Bölüm) kırılımına göre görselleştirir.
""")

# --- 1. API ANAHTARI YÖNETİMİ (Secrets Entegrasyonu) ---
api_key = None

# Önce Streamlit Secrets içinde anahtar var mı diye bakıyoruz
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # Yoksa (örneğin secrets dosyası oluşturulmadıysa) manuel giriş ister
    with st.sidebar:
        st.warning("⚠️ 'secrets.toml' dosyası bulunamadı.")
        api_key = st.text_input("Google Gemini API Key", type="password")

# --- 2. DOSYA YÜKLEME VE AYARLAR ---
with st.sidebar:
    st.header("Veri Ayarları")
    # HATA ÇÖZÜMÜ: Kullanıcının ayırıcıyı seçmesine izin veriyoruz
    separator = st.selectbox(
        "CSV Ayırıcı (Separator)", 
        options=[";", ",", "\t"], 
        index=0, # Varsayılan olarak noktalı virgül (Excel/Türkiye standardı)
        help="Dosyanız Excel çıktısıysa genelde ';' (noktalı virgül) kullanılır."
    )
    uploaded_file = st.file_uploader("CSV Dosyasını Yükle", type=["csv"])

if uploaded_file and api_key:
    try:
        # CSV OKUMA (Hata toleranslı)
        df = pd.read_csv(uploaded_file, sep=separator, engine='python', on_bad_lines='skip')
        
        st.write("### 📋 Veri Önizlemesi")
        st.dataframe(df.head(3))

        # --- 3. SÜTUN SEÇİMİ ---
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            text_column = st.selectbox("Analiz Edilecek Metin Sütunu (Örn: Görüşler)", df.columns)
        with col2:
            major_column = st.selectbox("Kırılım Sütunu (Örn: Major/Bölüm)", df.columns)

        # --- 4. ANALİZ İŞLEMİ ---
        if st.button("🚀 Analizi Başlat"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            with st.spinner('Yapay zeka verileri okuyor, kodluyor ve analiz ediyor...'):
                
                # Veriyi JSON formatına hazırlama
                data_input = []
                # Veri çok büyükse ilk 100-200 satırı alabilirsiniz. Şimdilik hepsini alıyoruz.
                for index, row in df.iterrows():
                    data_input.append({
                        "id": index,
                        "major": str(row[major_column]), # String'e çeviriyoruz hata olmaması için
                        "text": str(row[text_column])
                    })
                
                # Prompt (İstem)
                prompt = f"""
                Sen uzman bir nitel veri analistisin (Qualitative Data Analyst). 
                Aşağıdaki veri setini Tematik Analiz yöntemiyle incele.

                GÖREVLER:
                1. Katılımcı görüşlerinden ana temaları ve bunların alt temalarını belirle.
                2. Her tema için, o fikri en iyi ifade eden çarpıcı "doğrudan alıntılar" seç. Alıntıyı yapan kişinin Major'ını belirt.
                3. Hangi temanın hangi "Major" (bölüm) tarafından ne kadar zikredildiğini (frekansını) say.

                ÇIKTI FORMATI (SADECE SAF JSON):
                Cevabın kesinlikle ve sadece aşağıdaki JSON formatında olmalı. Başka açıklama yazma.
                
                {{
                    "analiz_ozeti": "Analizin genel sonucunu özetleyen profesyonel bir paragraf.",
                    "temalar": [
                        {{
                            "tema_adi": "Tema Başlığı",
                            "toplam_frekans": 15,
                            "alt_temalar": ["Alt tema 1", "Alt tema 2"],
                            "major_dagilimi": {{"Bilgisayar Müh": 10, "Mimarlık": 5, "Diğer": 0}},
                            "ornek_alintilar": [
                                {{"alinti": "Ders yükü çok fazlaydı...", "major": "Bilgisayar Müh"}},
                                {{"alinti": "Stüdyo dersleri yorucu...", "major": "Mimarlık"}}
                            ]
                        }}
                    ]
                }}

                VERİ SETİ:
                {json.dumps(data_input, ensure_ascii=False)}
                """

try:
                    response = model.generate_content(prompt)
                    
                    # --- GÜÇLENDİRİLMİŞ TEMİZLİK KODU (REGEX) ---
                    # Bu kod, yapay zeka ne kadar geveze olursa olsun 
                    # metnin içinden sadece { ile başlayıp } ile biten JSON kısmını cımbızla çeker.
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    
                    if match:
                        cleaned_text = match.group(0)
                    else:
                        # Eğer regex bulamazsa manuel temizliği dene
                        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
                    
                    try:
                        result = json.loads(cleaned_text)
                        
                        # --- SONUÇLARI GÖSTER ---
                        st.success("Analiz Tamamlandı!")
                        
                        # 1. Özet
                        st.subheader("📝 Yönetici Özeti")
                        st.info(result.get("analiz_ozeti", "Özet bulunamadı."))
                        
                        # 2. Grafikleştirme
                        temalar = result.get("temalar", [])
                        chart_data = []
                        
                        for t in temalar:
                            # Grafik verisi hazırlama
                            dagilim = t.get("major_dagilimi", {})
                            for maj, count in dagilim.items():
                                chart_data.append({
                                    "Tema": t["tema_adi"], 
                                    "Bölüm (Major)": maj, 
                                    "Frekans": count
                                })
                        
                        if chart_data:
                            st.write("---")
                            st.subheader("📊 Temaların Bölümlere Göre Dağılımı")
                            df_chart = pd.DataFrame(chart_data)
                            fig = px.bar(
                                df_chart, 
                                x="Tema", 
                                y="Frekans", 
                                color="Bölüm (Major)", 
                                barmode="group",
                                title="Tema ve Bölüm İlişkisi",
                                text_auto=True
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # 3. Detaylar ve Alıntılar
                        st.write("---")
                        st.subheader("🔍 Tema Detayları ve Alıntılar")
                        
                        for tema in temalar:
                            with st.expander(f"📌 {tema['tema_adi']} (Toplam: {tema['toplam_frekans']})"):
                                st.markdown(f"**Alt Temalar:** {', '.join(tema.get('alt_temalar', []))}")
                                st.markdown("#### 🗣️ Doğrudan Alıntılar")
                                for alinti in tema.get('ornek_alintilar', []):
                                    st.markdown(f"> *\"{alinti['alinti']}\"*")
                                    st.caption(f"— Bölüm: {alinti['major']}")

                    except json.JSONDecodeError:
                        st.error("AI yanıtı hala uygun formatta değil. Ham veri aşağıdadır:")
                        st.code(cleaned_text, language='json') # Hata olursa kod bloğu içinde göster

                except Exception as e:
                    st.error(f"API Hatası: {e}")

    except Exception as e:
        st.error("Dosya okunurken bir hata oluştu.")
        st.warning("Lütfen sol menüden 'CSV Ayırıcı' seçeneğini değiştirip tekrar deneyin (Örn: ; yerine , seçin).")
        st.error(f"Teknik Hata: {e}")

elif not api_key:
    st.info("Lütfen '.streamlit/secrets.toml' dosyasını oluşturun veya sol menüden API anahtarınızı girin.")

elif not uploaded_file:
    st.info("Lütfen analiz edilecek CSV dosyasını yükleyin.")





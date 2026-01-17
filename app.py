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

# --- 1. API ANAHTARI YÖNETİMİ ---
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        api_key = st.text_input("Google Gemini API Key", type="password")

# --- 2. DOSYA YÜKLEME VE AYARLAR ---
with st.sidebar:
    st.header("Veri Ayarları")
    separator = st.selectbox(
        "CSV Ayırıcı (Separator)", 
        options=[";", ",", "\t"], 
        index=0, 
        help="Dosyanız Excel çıktısıysa genelde ';' (noktalı virgül) kullanılır."
    )
    uploaded_file = st.file_uploader("CSV Dosyasını Yükle", type=["csv"])

if uploaded_file and api_key:
    try:
        # CSV OKUMA
        df = pd.read_csv(uploaded_file, sep=separator, engine='python', on_bad_lines='skip')
        
        st.write("### 📋 Veri Önizlemesi")
        st.dataframe(df.head(3))

        # --- 3. SÜTUN SEÇİMİ ---
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            text_column = st.selectbox("Analiz Edilecek Metin Sütunu", df.columns)
        with col2:
            major_column = st.selectbox("Kırılım Sütunu (Major/Bölüm)", df.columns)

        # --- 4. ANALİZ İŞLEMİ ---
        if st.button("🚀 Analizi Başlat"):
            genai.configure(api_key=api_key)
            # Eğer 'gemini-1.5-flash' hata verirse 'gemini-pro' kullanabilirsiniz.
            model = genai.GenerativeModel('gemini-1.5-flash')

            with st.spinner('Yapay zeka verileri okuyor, kodluyor ve analiz ediyor...'):
                
                # Veri Hazırlığı
                data_input = []
                for index, row in df.iterrows():
                    data_input.append({
                        "id": index,
                        "major": str(row[major_column]), 
                        "text": str(row[text_column])
                    })
                
                # Prompt
                prompt = f"""
                Sen uzman bir nitel veri analistisin. Aşağıdaki veri setini analiz et.

                GÖREVLER:
                1. Katılımcı görüşlerinden ana temaları ve alt temaları belirle.
                2. Her tema için çarpıcı "doğrudan alıntılar" seç ve alıntıyı yapanın Major'ını belirt.
                3. Hangi temanın hangi "Major" (bölüm) tarafından ne kadar zikredildiğini say.

                ÇIKTI FORMATI (SADECE JSON):
                Cevabın kesinlikle ve sadece aşağıdaki JSON formatında olmalı. Markdown kullanma.
                
                {{
                    "analiz_ozeti": "Genel değerlendirme paragrafı...",
                    "temalar": [
                        {{
                            "tema_adi": "Tema Başlığı",
                            "toplam_frekans": 15,
                            "alt_temalar": ["Alt 1", "Alt 2"],
                            "major_dagilimi": {{"Bölüm A": 10, "Bölüm B": 5}},
                            "ornek_alintilar": [
                                {{"alinti": "Örnek cümle...", "major": "Bölüm A"}}
                            ]
                        }}
                    ]
                }}

                VERİ SETİ:
                {json.dumps(data_input, ensure_ascii=False)}
                """

                try:
                    # API ÇAĞRISI
                    response = model.generate_content(prompt)
                    
                    # --- REGEX İLE TEMİZLİK (Hata Çözümü) ---
                    # Yanıtın içinden sadece { ile başlayıp } ile biten JSON kısmını alır.
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    
                    if match:
                        cleaned_text = match.group(0)
                    else:
                        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
                    
                    # JSON PARSE
                    try:
                        result = json.loads(cleaned_text)
                        
                        # --- SONUÇLARI GÖSTER ---
                        st.success("Analiz Tamamlandı!")
                        
                        # Özet
                        st.subheader("📝 Yönetici Özeti")
                        st.info(result.get("analiz_ozeti", "Özet yok"))
                        
                        # Grafik
                        temalar = result.get("temalar", [])
                        chart_data = []
                        for t in temalar:
                            for maj, count in t.get("major_dagilimi", {}).items():
                                chart_data.append({
                                    "Tema": t["tema_adi"], 
                                    "Bölüm": maj, 
                                    "Frekans": count
                                })
                        
                        if chart_data:
                            st.write("---")
                            st.subheader("📊 Temaların Bölümlere Göre Dağılımı")
                            df_chart = pd.DataFrame(chart_data)
                            fig = px.bar(df_chart, x="Tema", y="Frekans", color="Bölüm", barmode="group", text_auto=True)
                            st.plotly_chart(fig, use_container_width=True)

                        # Detaylar
                        st.write("---")
                        st.subheader("🔍 Detaylar ve Alıntılar")
                        for tema in temalar:
                            with st.expander(f"📌 {tema['tema_adi']} ({tema['toplam_frekans']})"):
                                st.markdown(f"**Alt Temalar:** {', '.join(tema.get('alt_temalar', []))}")
                                st.markdown("#### 🗣️ Alıntılar")
                                for alinti in tema.get('ornek_alintilar', []):
                                    st.markdown(f"> *\"{alinti['alinti']}\"*")
                                    st.caption(f"— {alinti['major']}")

                    except json.JSONDecodeError:
                        st.error("JSON format hatası. Ham yanıt:")
                        st.code(cleaned_text)

                except Exception as e:
                    st.error(f"API Bağlantı Hatası: {e}")

    except Exception as e:
        st.error("Dosya okunurken hata oluştu. Ayırıcıyı değiştirmeyi deneyin.")
        st.error(str(e))

elif not api_key:
    st.info("Lütfen API anahtarınızı girin.")

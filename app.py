import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px

# Sayfa Ayarları
st.set_page_config(page_title="AI Tematik Analiz Aracı", layout="wide")

st.title("📊 AI Destekli Tematik Veri Analizi")
st.markdown("""
Bu araç, CSV dosyanızdaki açık uçlu yanıtları analiz eder, temaları belirler 
ve katılımcıların 'Major' (Bölüm/Branş) bilgilerine göre kırılımlar sunar.
""")

# 1. API Anahtarı Girişi (Güvenlik için Sidebar'da)
with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.info("API anahtarınızı Google AI Studio'dan alabilirsiniz.")

# 2. Dosya Yükleme
uploaded_file = st.file_uploader("Veri Setinizi Yükleyin (CSV)", type=["csv"])

if uploaded_file and api_key:
    # Veriyi Oku
    df = pd.read_csv(uploaded_file)
    st.write("Veri Önizlemesi:", df.head(3))

    # 3. Sütun Seçimi
    col1, col2 = st.columns(2)
    with col1:
        text_column = st.selectbox("Analiz Edilecek Metin Sütunu (Örn: Görüşler)", df.columns)
    with col2:
        major_column = st.selectbox("Kırılım Sütunu (Örn: Major/Bölüm)", df.columns)

    if st.button("Analizi Başlat"):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            with st.spinner('Yapay zeka verileri okuyor, temaları çıkarıyor ve analiz ediyor... Bu işlem verinin boyutuna göre 1-2 dakika sürebilir.'):
                
                # Veriyi metne dönüştür (Token limitini aşmamak için büyük veride örneklem alınabilir)
                # Burada veriyi JSON benzeri bir yapıda modele sunuyoruz ki kırılım yapabilsin.
                data_input = []
                for index, row in df.iterrows():
                    data_input.append({
                        "id": index,
                        "major": row[major_column],
                        "text": row[text_column]
                    })
                
                # Prompt Mühendisliği
                prompt = f"""
                Sen uzman bir nitel veri analistisin. Aşağıdaki veri setini analiz et.
                
                GÖREVLER:
                1. Katılımcı görüşlerinden ana temaları ve alt temaları belirle.
                2. Her tema için katılımcıların ifadelerinden çarpıcı "doğrudan alıntılar" seç (hangi Major'dan olduğunu belirt).
                3. Hangi temanın hangi "Major" (bölüm) tarafından ne kadar zikredildiğini say.
                
                ÇIKTI FORMATI (KESİNLİKLE SADECE JSON):
                Cevabın sadece aşağıdaki yapıda saf bir JSON olmalı, başında veya sonunda markdown (```json) olmamalı:
                
                {{
                    "analiz_ozeti": "Genel bir değerlendirme paragrafı...",
                    "temalar": [
                        {{
                            "tema_adi": "Tema Başlığı",
                            "toplam_frekans": 15,
                            "alt_temalar": ["Alt tema 1", "Alt tema 2"],
                            "major_dagilimi": {{"Bilgisayar Müh": 10, "Mimarlık": 5}},
                            "ornek_alintilar": [
                                {{"alinti": "Dersler çok yoğundu...", "major": "Bilgisayar Müh"}},
                                {{"alinti": "Tasarım odaklıydı...", "major": "Mimarlık"}}
                            ]
                        }}
                    ]
                }}

                VERİ SETİ:
                {json.dumps(data_input, ensure_ascii=False)}
                """

                response = model.generate_content(prompt)
                
                # JSON Temizliği (Bazen model markdown tagleri ekleyebilir)
                cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
                result = json.loads(cleaned_text)

                # --- SONUÇLARI GÖSTER ---
                
                st.success("Analiz Tamamlandı!")
                
                # Genel Özet
                st.subheader("📝 Yönetici Özeti")
                st.write(result.get("analiz_ozeti", ""))
                st.divider()

                # Temaları Döngüye Al ve Göster
                temalar = result.get("temalar", [])
                
                # Grafik için veri hazırlığı
                chart_data = []
                for t in temalar:
                    for major, count in t["major_dagilimi"].items():
                        chart_data.append({"Tema": t["tema_adi"], "Major": major, "Frekans": count})
                
                # Grafik Çizimi (Plotly)
                if chart_data:
                    st.subheader("📊 Temaların Majorlara Göre Dağılımı")
                    df_chart = pd.DataFrame(chart_data)
                    fig = px.bar(df_chart, x="Tema", y="Frekans", color="Major", barmode="group", title="Tema ve Bölüm İlişkisi")
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("🔍 Tema Detayları ve Alıntılar")

                for tema in temalar:
                    with st.expander(f"📌 {tema['tema_adi']} (Toplam: {tema['toplam_frekans']})"):
                        st.markdown(f"**Alt Temalar:** {', '.join(tema['alt_temalar'])}")
                        
                        st.markdown("**Doğrudan Alıntılar:**")
                        for alinti in tema['ornek_alintilar']:
                            st.info(f"🗣️ \"{alinti['alinti']}\" \n\n— *{alinti.get('major', 'Belirsiz')}*")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
            st.warning("Veri seti çok büyükse veya API yanıtı bozuksa bu hata alınabilir. Lütfen daha küçük bir veri setiyle deneyin.")

elif not api_key:
    st.warning("Lütfen sol menüden API anahtarınızı giriniz.")
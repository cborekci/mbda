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
Bu araç, nitel verilerinizi analiz eder. **Genel Analiz** sekmesinde tüm veri setinin özetini, 
**Bölüm Bazlı Analiz** sekmesinde ise seçtiğiniz bölüme özel detayları görebilirsiniz.
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
        help="Excel çıktıları için genelde ';' kullanılır."
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
            model = genai.GenerativeModel('gemini-2.5-flash')

            with st.spinner('Yapay zeka verileri okuyor, temaları kodluyor ve analiz ediyor...'):
                
                # Veri Hazırlığı
                data_input = []
                # Veri setindeki tüm benzersiz bölümleri alalım
                unique_majors = df[major_column].unique().tolist()
                
                for index, row in df.iterrows():
                    data_input.append({
                        "id": index,
                        "major": str(row[major_column]), 
                        "text": str(row[text_column])
                    })
                
                # --- GÜNCELLENMİŞ PROMPT ---
                prompt = f"""
                Sen uzman bir nitel veri analistisin. Aşağıdaki veri setini analiz et.

                GÖREVLER:
                1. Bütünsel Analiz: Katılımcı görüşlerinden ana temaları belirle.
                2. Detaylandırma: Her ana tema için 2-4 adet açıklayıcı "alt tema" belirle.
                3. Alıntılama: Her tema için çarpıcı "doğrudan alıntılar" seç ve alıntıyı yapanın Major'ını (Bölümünü) mutlaka belirt.
                4. Frekans: Hangi temanın hangi "Major" (bölüm) tarafından ne kadar zikredildiğini say.

                ÇIKTI FORMATI (SADECE JSON):
                Cevabın kesinlikle ve sadece aşağıdaki JSON formatında olmalı. Markdown kullanma.
                
                {{
                    "analiz_ozeti": "Veri setinin genelindeki eğilimleri anlatan 1 paragraf özet.",
                    "temalar": [
                        {{
                            "tema_adi": "Tema Başlığı (Örn: Müfredat Yetersizliği)",
                            "toplam_frekans": 25,
                            "alt_temalar": ["Teorik ders yoğunluğu", "Pratik eksikliği", "Güncel olmayan içerik"],
                            "major_dagilimi": {{"Bilgisayar Müh": 15, "Mimarlık": 10}},
                            "ornek_alintilar": [
                                {{"alinti": "Dersler çok teorik...", "major": "Bilgisayar Müh"}},
                                {{"alinti": "Atölye saatleri az...", "major": "Mimarlık"}}
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
                    
                    # Regex ile temizlik
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    cleaned_text = match.group(0) if match else response.text.replace("```json", "").replace("```", "").strip()
                    
                    # JSON PARSE
                    result = json.loads(cleaned_text)
                    st.success("Analiz Tamamlandı!")

                    # --- YENİ ARAYÜZ YAPISI: SEKMELER (TABS) ---
                    tab1, tab2 = st.tabs(["📊 Genel Analiz", "🎓 Bölüm (Major) Kırılımı"])

                    # --- SEKME 1: GENEL ANALİZ ---
                    with tab1:
                        st.subheader("📝 Yönetici Özeti")
                        st.info(result.get("analiz_ozeti", "Özet yok"))
                        
                        st.divider()
                        
                        # Grafik Verisi Hazırlama
                        temalar = result.get("temalar", [])
                        chart_data = []
                        for t in temalar:
                            for maj, count in t.get("major_dagilimi", {}).items():
                                chart_data.append({
                                    "Tema": t["tema_adi"], 
                                    "Bölüm": maj, 
                                    "Frekans": count
                                })
                        
                        # 1. İSTEK: YIĞILIMLI ÇUBUK GRAFİĞİ (Stacked Bar Chart)
                        if chart_data:
                            st.subheader("📈 Temaların Bölümlere Göre Yığılımlı Dağılımı")
                            df_chart = pd.DataFrame(chart_data)
                            fig = px.bar(
                                df_chart, 
                                x="Tema", 
                                y="Frekans", 
                                color="Bölüm", 
                                title="Tema Frekansları (Bölüm Kırılımlı)",
                                text_auto=True
                            )
                            # Stacked (Yığılımlı) olması için layout güncellemesi
                            fig.update_layout(barmode='stack', xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)

                        st.divider()
                        st.subheader("🧩 Temalar ve Alt Temalar")
                        
                        # 2. İSTEK: ALT TEMALAR VE GENEL GÖRÜNÜM
                        for tema in temalar:
                            with st.expander(f"📌 {tema['tema_adi']} (Toplam: {tema['toplam_frekans']})"):
                                # Alt temaları madde işaretli liste olarak gösterme
                                st.markdown("**Alt Temalar:**")
                                for sub in tema.get('alt_temalar', []):
                                    st.markdown(f"- {sub}")
                                
                                st.markdown("---")
                                st.markdown("**Örnek Alıntılar:**")
                                for alinti in tema.get('ornek_alintilar', []):
                                    st.markdown(f"> *\"{alinti['alinti']}\"*")
                                    st.caption(f"— {alinti['major']}")

                    # --- SEKME 2: BÖLÜM (MAJOR) BAZLI ANALİZ ---
                    with tab2:
                        st.subheader("🔍 Bölüm Bazlı Detaylandırma")
                        
                        # Bölüm Seçim Kutusu
                        # JSON'dan gelen verilerdeki tüm bölümleri toplayalım
                        available_majors = set()
                        for t in temalar:
                            available_majors.update(t.get("major_dagilimi", {}).keys())
                        
                        selected_major = st.selectbox("İncelemek istediğiniz Bölümü (Major) Seçin:", list(available_majors))

                        if selected_major:
                            st.markdown(f"### 🎓 {selected_major} Bölümü İçin Bulgular")
                            
                            major_has_data = False
                            for tema in temalar:
                                # Bu tema bu bölümde hiç geçmiş mi?
                                major_count = tema.get("major_dagilimi", {}).get(selected_major, 0)
                                
                                if major_count > 0:
                                    major_has_data = True
                                    # Karta benzer görünüm
                                    with st.container():
                                        st.markdown(f"#### {tema['tema_adi']}")
                                        st.write(f"Bu bölümden katılım sıklığı: **{major_count}**")
                                        
                                        # Sadece bu bölüme ait alıntıları filtrele
                                        major_quotes = [q['alinti'] for q in tema.get('ornek_alintilar', []) if q.get('major') == selected_major]
                                        
                                        if major_quotes:
                                            st.markdown("**Bu bölümden gelen ifadeler:**")
                                            for q in major_quotes:
                                                st.info(f"🗣️ {q}")
                                        else:
                                            st.markdown("*Bu tema için bu bölümden doğrudan alıntı seçilmemiş.*")
                                        
                                        st.divider()
                            
                            if not major_has_data:
                                st.warning(f"{selected_major} bölümü için belirgin bir tema verisi bulunamadı.")

                except json.JSONDecodeError:
                    st.error("AI yanıtı JSON formatında değil. Ham veri:")
                    st.code(cleaned_text)
                except Exception as e:
                    st.error(f"İşlem Hatası: {e}")

    except Exception as e:
        st.error("Dosya yüklenirken hata oluştu. Lütfen 'Ayırıcı'yı değiştirmeyi deneyin.")
        st.error(str(e))

elif not api_key:
    st.warning("Lütfen API anahtarınızı girin.")

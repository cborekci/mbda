import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import re

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="InsightAI - Thematic Analysis", layout="wide", page_icon="🟣")

# --- STİL (CSS) ---
# Kart görünümleri ve başlıklar için ufak dokunuşlar
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

# --- 1. API ANAHTARI (SADECE BURASI SIDEBAR'DA KALDI) ---
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 Authentication")
        api_key = st.text_input("Gemini API Key", type="password")

# --- BAŞLIK ---
st.title("🟣 InsightAI")
st.markdown("### Automated Thematic Analysis & Field Segmentation")
st.markdown("Upload your dataset below. The system will auto-detect the text and group columns.")

# --- 2. MERKEZİ DOSYA YÜKLEME (KONFİGÜRASYON YOK) ---
file_container = st.container()
with file_container:
    uploaded_file = st.file_uploader("", type=["csv"], help="Upload a standard CSV file.")

# --- UYGULAMA MANTIĞI ---
if uploaded_file and api_key:
    try:
        # OTOMATİK AYIRICI TESPİTİ VE OKUMA
        # Pandas'ın python motoru genelde ayırıcıyı tahmin edebilir ama biz standart okuyalım
        # Eğer veri düzgün okunmazsa, kullanıcı dosyasını ; veya , standardına getirmeli.
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        except:
            df = pd.read_csv(uploaded_file, sep=";", engine='python', on_bad_lines='skip')

        # --- 3. OTOMATİK SÜTUN TESPİTİ (AUTO-DETECT) ---
        # Mantık: Ortalama karakter sayısı en yüksek olan sütun "Metin"dir.
        # Benzersiz değer sayısı daha az olan (kategorik) sütun "Major"dur.
        
        cols = df.columns
        if len(cols) < 2:
            st.error("Error: CSV must have at least 2 columns (Group and Text).")
            st.stop()
            
        # Basit sezgisel tespit
        text_column = None
        major_column = None
        
        max_avg_len = 0
        for col in cols:
            # Sütun string tipindeyse ortalama uzunluğuna bak
            if df[col].dtype == object or df[col].dtype == str:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > max_avg_len:
                    max_avg_len = avg_len
                    text_column = col
        
        # Text column dışındaki ilk sütunu major kabul edelim (veya en az unique değere sahip olanı)
        remaining_cols = [c for c in cols if c != text_column]
        if remaining_cols:
            major_column = remaining_cols[0] # Genelde ilk sütun ID veya Majordur.
        else:
            major_column = cols[0] # Fallback

        st.info(f"✅ **Auto-Detected Structure:** Grouping by **'{major_column}'** | Analyzing Text from **'{text_column}'**")
        
        # ANALİZ BUTONU
        if st.button("🚀 Start AI Analysis", type="primary"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            with st.spinner('InsightAI is processing data, counting frequencies, and extracting quotes...'):
                
                # VERİYİ HAZIRLA
                data_input = []
                # Token limitini korumak için max 1000 satır (opsiyonel, şu an hepsi)
                for index, row in df.iterrows():
                    data_input.append({
                        "id": index,
                        "group": str(row[major_column]), 
                        "text": str(row[text_column])
                    })
                
                # --- PROMPT GÜNCELLEMESİ (Sub-theme Counts) ---
                prompt = f"""
                You are InsightAI, an expert qualitative data analyst. 
                Analyze the following dataset. 

                **CRITICAL RULE:** ALL OUTPUT MUST BE IN ENGLISH.

                TASKS:
                1. **General Overview:** Summary paragraph (approx 100 words).
                2. **Thematic Coding:** Identify main themes.
                3. **Sub-themes with Counts:** For each theme, identify sub-themes AND count how many comments fall into that sub-theme.
                4. **Quantification:** Count theme mentions by "Group".
                5. **Quotes:** Select impactful quotes per theme, labeled by "Group".

                OUTPUT FORMAT (STRICT JSON):
                {{
                    "overview": "Executive summary string...",
                    "themes": [
                        {{
                            "id": 1,
                            "name": "Main Theme Title",
                            "definition": "Short definition.",
                            "total_count": 50,
                            "sub_themes": [
                                {{"name": "Sub-theme A", "count": 30}},
                                {{"name": "Sub-theme B", "count": 20}}
                            ],
                            "group_distribution": {{"Group A": 30, "Group B": 20}},
                            "quotes": [
                                {{"text": "Quote text...", "group": "Group A"}}
                            ]
                        }}
                    ]
                }}

                DATASET:
                {json.dumps(data_input, ensure_ascii=False)}
                """

                try:
                    # API ÇAĞRISI
                    response = model.generate_content(prompt)
                    
                    # TEMİZLİK
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    cleaned_text = match.group(0) if match else response.text.replace("```json", "").replace("```", "").strip()
                    
                    result = json.loads(cleaned_text)
                    
                    st.success("Analysis Complete!")
                    
                    # --- SEKME YAPISI ---
                    tab_overview, tab_breakdown = st.tabs(["📊 General Overview", "🎓 Detailed Field Breakdown"])

                    # ==================================================
                    # TAB 1: GENEL BAKIŞ
                    # ==================================================
                    with tab_overview:
                        # 1. ÖZET
                        st.markdown("### 📝 Executive Summary")
                        st.info(result.get("overview", "No summary."))
                        st.divider()

                        # 2. GRAFİK (YATAY STACKED)
                        st.markdown("### 📉 Dominant Themes Landscape")
                        themes = result.get("themes", [])
                        chart_data = []
                        for t in themes:
                            for grp, count in t.get("group_distribution", {}).items():
                                chart_data.append({"Theme": t["name"], "Group": grp, "Count": count})
                        
                        if chart_data:
                            df_chart = pd.DataFrame(chart_data)
                            fig = px.bar(
                                df_chart, x="Count", y="Theme", color="Group", orientation='h', 
                                title="Theme Distribution by Field", text_auto=True, 
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        st.divider()

                        # 3. TEMA KARTLARI (SUB-THEME COUNTS İLE)
                        st.markdown("### 🧩 Theme Analysis")
                        for t in themes:
                            with st.expander(f"📌 {t['name']} (Total: {t['total_count']})", expanded=True):
                                st.write(f"_{t.get('definition', '')}_")
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown("**Sub-Themes & Frequencies:**")
                                    # Alt temaları ve sayılarını yazdır
                                    for sub in t.get("sub_themes", []):
                                        # Eğer API string döndürdüyse (eski format koruması)
                                        if isinstance(sub, str):
                                            st.markdown(f"• {sub}")
                                        else:
                                            # Yeni format: Obje {name, count}
                                            st.markdown(f"• **{sub['name']}** ({sub['count']} mentions)")
                                with c2:
                                    st.markdown("**Key Voices:**")
                                    for q in t.get("quotes", []):
                                        st.caption(f"🗣️ \"{q['text']}\" — *{q['group']}*")

                    # ==================================================
                    # TAB 2: DETAYLI BÖLÜM ANALİZİ (DROP-DOWN YOK)
                    # ==================================================
                    with tab_breakdown:
                        st.subheader("🔍 Full Breakdown by Field")
                        st.markdown("Below is the detailed thematic distribution for every field found in the dataset.")
                        
                        # Tüm unique grupları bul
                        all_groups = sorted(list(set(g for t in themes for g in t.get("group_distribution", {}).keys())))
                        
                        for group in all_groups:
                            # Her bölüm için bir kapsayıcı
                            with st.container():
                                st.markdown(f"## 🎓 {group}")
                                
                                has_data = False
                                for t in themes:
                                    count = t.get("group_distribution", {}).get(group, 0)
                                    if count > 0:
                                        has_data = True
                                        # Temayı ve o bölümdeki yoğunluğunu göster
                                        st.markdown(f"**{t['name']}** (Frequency: {count})")
                                        
                                        # İlerleme çubuğu (O temanın toplamına göre bu bölümün payı)
                                        ratio = count / t['total_count']
                                        st.progress(ratio)
                                        
                                        # Sadece bu gruba ait alıntıları çek
                                        group_quotes = [q['text'] for q in t.get("quotes", []) if q.get("group") == group]
                                        if group_quotes:
                                            for gq in group_quotes:
                                                st.info(f"🗣️ \"{gq}\"")
                                        else:
                                            st.caption("*No direct quotes selected for this specific theme/group.*")
                                        
                                        st.markdown("---")
                                
                                if not has_data:
                                    st.warning(f"No significant themes detected for {group}.")
                            
                            # Gruplar arası büyük boşluk
                            st.write("##") 

                except Exception as e:
                    st.error(f"AI Processing Error: {e}")

    except Exception as e:
        st.error("Error reading file. Please ensure it is a standard CSV/Excel export.")
        st.error(str(e))

elif not api_key:
    # Boş durum (Başlangıç ekranı)
    st.info("👋 Please enter your API Key in the sidebar to start.")


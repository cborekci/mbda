import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import re

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="InsightAI - Thematic Analysis", layout="wide", page_icon="🟣")

# --- STİL (CSS) ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

# --- 1. API ANAHTARI ---
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
st.markdown("""
**Required CSV Format:**
1. Column: **Participant ID** (e.g., O1, O2)
2. Column: **Major/Field** (e.g., Science, Math)
3. Column: **Opinion/Text** (The feedback to analyze)
""")

# --- 2. DOSYA YÜKLEME ---
file_container = st.container()
with file_container:
    uploaded_file = st.file_uploader("", type=["csv"], help="Upload your CSV file with the 3-column format.")

# --- UYGULAMA MANTIĞI ---
if uploaded_file and api_key:
    try:
        # CSV OKUMA
        # Header varsa kullanır, yoksa da ilk satırı header sanabilir ama biz index ile erişeceğiz.
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        except:
            df = pd.read_csv(uploaded_file, sep=";", engine='python', on_bad_lines='skip')

        # --- 3. SÜTUN SABİTLEME (MANUEL MAPPING) ---
        if len(df.columns) < 3:
            st.error("❌ Error: The file must have at least 3 columns (ID, Major, Text).")
            st.stop()
        
        # Sütun isimleri ne olursa olsun:
        # 0. index -> ID
        # 1. index -> Major (Group)
        # 2. index -> Text
        
        # Veriyi temiz bir dataframe'e çekelim
        df_clean = pd.DataFrame({
            "ID": df.iloc[:, 0].astype(str),
            "Group": df.iloc[:, 1].astype(str),
            "Text": df.iloc[:, 2].astype(str)
        })

        st.info(f"✅ **File Loaded:** Processing **{len(df_clean)}** rows. Analyzing text from column 3, grouped by column 2.")
        
        # ANALİZ BUTONU
        if st.button("🚀 Start AI Analysis", type="primary"):
            genai.configure(api_key=api_key)
            
            # --- ÖNEMLİ DÜZELTME: JSON MODE ---
            # response_mime_type="application/json" komutu modelin sadece geçerli JSON üretmesini zorlar.
            # Bu, "Expecting ',' delimiter" hatasını çözer.
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config={"response_mime_type": "application/json"}
            )

            with st.spinner('InsightAI is analyzing themes, sub-themes, and frequencies...'):
                
                # VERİYİ HAZIRLA
                data_input = []
                for index, row in df_clean.iterrows():
                    data_input.append({
                        "id": row["ID"],
                        "group": row["Group"], 
                        "text": row["Text"]
                    })
                
                # --- PROMPT ---
                prompt = f"""
                You are InsightAI, an expert qualitative data analyst. 
                Analyze the following dataset provided in JSON format.

                **CRITICAL RULE:** ALL OUTPUT MUST BE IN ENGLISH.

                TASKS:
                1. **General Overview:** Write an executive summary paragraph (approx 100 words).
                2. **Thematic Coding:** Identify main themes from the comments.
                3. **Sub-themes & Counts:** For each theme, identify sub-themes AND count exactly how many comments belong to each sub-theme.
                4. **Quantification:** Count how many times each theme is mentioned by each "Group" (Major).
                5. **Quotes:** Select impactful quotes for each theme, labeled by "Group".

                OUTPUT FORMAT (You must return a valid JSON object):
                {{
                    "overview": "Executive summary string...",
                    "themes": [
                        {{
                            "id": 1,
                            "name": "Main Theme Title",
                            "definition": "Short definition of the theme.",
                            "total_count": 50,
                            "sub_themes": [
                                {{"name": "Sub-theme A", "count": 30}},
                                {{"name": "Sub-theme B", "count": 20}}
                            ],
                            "group_distribution": {{"Science": 30, "Math": 20}},
                            "quotes": [
                                {{"text": "Sample quote...", "group": "Science"}}
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
                    
                    # TEMİZLİK (JSON Mode kullandığımız için regex'e gerek kalmayabilir ama güvenlik için tutuyoruz)
                    # Model doğrudan JSON döndüreceği için markdown tagleri temizlemek yeterli.
                    cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
                    
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

                        # 3. TEMA KARTLARI
                        st.markdown("### 🧩 Theme Analysis")
                        for t in themes:
                            with st.expander(f"📌 {t['name']} (Total: {t['total_count']})", expanded=True):
                                st.write(f"_{t.get('definition', '')}_")
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown("**Sub-Themes & Frequencies:**")
                                    for sub in t.get("sub_themes", []):
                                        if isinstance(sub, dict):
                                             st.markdown(f"• **{sub.get('name', 'Unknown')}** ({sub.get('count', 0)})")
                                        else:
                                            st.markdown(f"• {sub}")
                                with c2:
                                    st.markdown("**Key Voices:**")
                                    for q in t.get("quotes", []):
                                        st.caption(f"🗣️ \"{q['text']}\" — *{q['group']}*")

                    # ==================================================
                    # TAB 2: DETAYLI BÖLÜM ANALİZİ
                    # ==================================================
                    with tab_breakdown:
                        st.subheader("🔍 Full Breakdown by Field")
                        
                        # Tüm unique grupları bul
                        all_groups = sorted(list(set(g for t in themes for g in t.get("group_distribution", {}).keys())))
                        
                        for group in all_groups:
                            with st.container():
                                st.markdown(f"## 🎓 {group}")
                                
                                has_data = False
                                for t in themes:
                                    count = t.get("group_distribution", {}).get(group, 0)
                                    if count > 0:
                                        has_data = True
                                        st.markdown(f"**{t['name']}** (Frequency: {count})")
                                        
                                        # Progress bar (Zero division hatasını önle)
                                        total = t.get('total_count', 1)
                                        if total == 0: total = 1
                                        
                                        ratio = count / total
                                        st.progress(ratio)
                                        
                                        # Alıntılar
                                        group_quotes = [q['text'] for q in t.get("quotes", []) if q.get("group") == group]
                                        if group_quotes:
                                            for gq in group_quotes:
                                                st.info(f"🗣️ \"{gq}\"")
                                        else:
                                            st.caption("*No direct quotes selected for this specific theme/group.*")
                                        
                                        st.markdown("---")
                                
                                if not has_data:
                                    st.warning(f"No significant themes detected for {group}.")
                            
                            st.write("##") 

                except Exception as e:
                    st.error(f"AI Processing Error: {e}")
                    st.warning("If the error persists, check if your CSV file has strange characters or encoding issues.")

    except Exception as e:
        st.error("Error reading file. Please upload a CSV with 3 columns.")
        st.error(str(e))

elif not api_key:
    st.info("👋 Please enter your API Key in the sidebar to start.")

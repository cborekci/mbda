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
1. Column: **Participant ID**
2. Column: **Major/Field**
3. Column: **Opinion/Text**
""")

# --- 2. DOSYA YÜKLEME ---
file_container = st.container()
with file_container:
    uploaded_file = st.file_uploader("", type=["csv"], help="Upload your CSV file with the 3-column format.")

# --- UYGULAMA MANTIĞI ---
if uploaded_file and api_key:
    try:
        # CSV OKUMA
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        except:
            df = pd.read_csv(uploaded_file, sep=";", engine='python', on_bad_lines='skip')

        if len(df.columns) < 3:
            st.error("❌ Error: The file must have at least 3 columns (ID, Major, Text).")
            st.stop()
        
        # Sütunları string'e çevir ve temizle
        df_clean = pd.DataFrame({
            "ID": df.iloc[:, 0].astype(str),
            "Group": df.iloc[:, 1].astype(str),
            "Text": df.iloc[:, 2].astype(str)
        })

        st.info(f"✅ **File Loaded:** Processing **{len(df_clean)}** rows.")
        
        if st.button("🚀 Start AI Analysis (High Precision)", type="primary"):
            genai.configure(api_key=api_key)
            
            # --- GÜNCELLEME 1: Token Limitini Artırdık ---
            # max_output_tokens değerini 15.000'e çıkardık (Flash modeli bunu destekler).
            generation_config = {
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 15000, 
                "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                generation_config=generation_config
            )

            with st.spinner('Sanitizing data and analyzing themes (This may take a moment)...'):
                
                # VERİ TEMİZLİĞİ (Tırnak İşaretleri Sorunu İçin)
                data_input = []
                for index, row in df_clean.iterrows():
                    safe_text = row["Text"].replace('"', "'").replace("\n", " ").strip()
                    safe_group = row["Group"].replace('"', "'").strip()
                    
                    data_input.append({
                        "id": row["ID"],
                        "group": safe_group, 
                        "text": safe_text
                    })
                
                # --- GÜNCELLEME 2: Prompt Sınırlamaları ---
                # Çıktının kesilmemesi için "Top 5 sub-themes" ve "Max 2 quotes" sınırlarını ekledik.
                prompt = f"""
                You are InsightAI, an expert qualitative data analyst. 
                Analyze the following dataset provided in JSON format.

                **CRITICAL RULES:**
                1. ALL OUTPUT MUST BE IN ENGLISH.
                2. Return ONLY valid JSON.
                3. Do not use markdown formatting.

                TASKS:
                1. **General Overview:** Executive summary (approx 100 words).
                2. **Thematic Coding:** Identify main themes from the comments.
                3. **Sub-themes:** Identify the **TOP 5** sub-themes for each main theme and count frequencies.
                4. **Quantification:** Count theme mentions by "Group".
                5. **Quotes:** Select **MAXIMUM 2** impactful quotes per theme, labeled by "Group".

                OUTPUT JSON STRUCTURE:
                {{
                    "overview": "Summary text...",
                    "themes": [
                        {{
                            "id": 1,
                            "name": "Theme Name",
                            "definition": "Description...",
                            "total_count": 0,
                            "sub_themes": [
                                {{"name": "Sub 1", "count": 0}}
                            ],
                            "group_distribution": {{"Group A": 0, "Group B": 0}},
                            "quotes": [
                                {{"text": "Quote...", "group": "Group A"}}
                            ]
                        }}
                    ]
                }}

                DATASET:
                {json.dumps(data_input, ensure_ascii=False)}
                """

                try:
                    response = model.generate_content(prompt)
                    
                    text_to_parse = response.text.replace("```json", "").replace("```", "").strip()
                    
                    result = json.loads(text_to_parse)
                    
                    st.success("Analysis Complete!")
                    
                    # --- SEKME YAPISI ---
                    tab_overview, tab_breakdown = st.tabs(["📊 General Overview", "🎓 Detailed Field Breakdown"])

                    # TAB 1: Genel Bakış
                    with tab_overview:
                        st.markdown("### 📝 Executive Summary")
                        st.info(result.get("overview", "No summary."))
                        st.divider()

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

                        st.markdown("### 🧩 Theme Analysis")
                        for t in themes:
                            with st.expander(f"📌 {t['name']} (Total: {t['total_count']})", expanded=True):
                                st.write(f"_{t.get('definition', '')}_")
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    st.markdown("**Top Sub-Themes:**")
                                    for sub in t.get("sub_themes", []):
                                        if isinstance(sub, dict):
                                             st.markdown(f"• **{sub.get('name', 'Unknown')}** ({sub.get('count', 0)})")
                                        else:
                                            st.markdown(f"• {sub}")
                                with c2:
                                    st.markdown("**Key Voices:**")
                                    for q in t.get("quotes", []):
                                        st.caption(f"🗣️ \"{q['text']}\" — *{q['group']}*")

                    # TAB 2: Detaylı Analiz
                    with tab_breakdown:
                        st.subheader("🔍 Full Breakdown by Field")
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
                                        
                                        total = t.get('total_count', 1)
                                        if total == 0: total = 1
                                        ratio = count / total
                                        st.progress(ratio)
                                        
                                        group_quotes = [q['text'] for q in t.get("quotes", []) if q.get("group") == group]
                                        if group_quotes:
                                            for gq in group_quotes:
                                                st.info(f"🗣️ \"{gq}\"")
                                        else:
                                            st.caption("*No direct quotes selected.*")
                                        st.markdown("---")
                                
                                if not has_data:
                                    st.warning(f"No significant themes detected for {group}.")
                            st.write("##") 

                except json.JSONDecodeError as e:
                    st.error("JSON Parsing Error. The AI response was truncated or malformed.")
                    st.error(f"Details: {e}")
                    # Debug için kesik metni gösterme (Opsiyonel)
                    # st.text(text_to_parse) 
                except Exception as e:
                    st.error(f"Processing Error: {e}")

    except Exception as e:
        st.error("Error reading file.")
        st.error(str(e))

elif not api_key:
    st.info("👋 Please enter your API Key in the sidebar to start.")

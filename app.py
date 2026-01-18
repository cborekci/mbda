import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import re

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="InsightAI - Thematic Analysis", layout="wide", page_icon="🟣")

# --- CSS STİLİ ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: 600;}
    .stAlert {margin-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

# --- 1. API ANAHTARI YÖNETİMİ ---
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
**Format Requirement:**
1. Column: **ID**
2. Column: **Major/Field**
3. Column: **Text/Opinion**
""")

# --- 2. DOSYA YÜKLEME ---
file_container = st.container()
with file_container:
    uploaded_file = st.file_uploader("", type=["csv"], help="Upload standard CSV.")

# --- UYGULAMA MANTIĞI ---
if uploaded_file and api_key:
    try:
        # CSV OKUMA (Esnek Ayırıcı)
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        except:
            df = pd.read_csv(uploaded_file, sep=";", engine='python', on_bad_lines='skip')

        if len(df.columns) < 3:
            st.error("❌ Error: File needs at least 3 columns.")
            st.stop()

        # TEMİZ VERİ ÇERÇEVESİ OLUŞTURMA
        df_clean = pd.DataFrame({
            "ID": df.iloc[:, 0].astype(str),
            "Group": df.iloc[:, 1].astype(str),
            "Text": df.iloc[:, 2].astype(str)
        })

        # --- GÜNCELLEME 1: AKILLI ÖRNEKLEM (Sampling) ---
        # Veri seti çok büyükse API yanıtı kesilir (Truncation Error). 
        # Bu yüzden 750 satırdan fazlasını rastgele örnekliyoruz.
        total_rows = len(df_clean)
        if total_rows > 750:
            df_analyzed = df_clean.sample(n=750, random_state=42)
            st.warning(f"⚠️ Dataset is large ({total_rows} rows). Analyzing a random sample of **750 rows** to ensure API stability.")
        else:
            df_analyzed = df_clean
            st.info(f"✅ Processing all **{total_rows}** rows.")

        if st.button("🚀 Start AI Analysis (High Precision)", type="primary"):
            genai.configure(api_key=api_key)
            
            # --- MODEL AYARLARI ---
            generation_config = {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_output_tokens": 8192, # Output limit
                "response_mime_type": "application/json",
            }
            
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash", 
                generation_config=generation_config
            )

            with st.spinner('Sanitizing data and generating insights...'):
                
                # JSON Input Hazırlığı (Temizlenmiş)
                data_input = []
                for index, row in df_analyzed.iterrows():
                    # Tırnak işaretlerini temizle ki JSON bozulmasın
                    safe_text = row["Text"].replace('"', "'").replace("\n", " ").strip()
                    safe_group = row["Group"].replace('"', "'").strip()
                    data_input.append({"group": safe_group, "text": safe_text})
                
                # --- GÜNCELLEME 2: PROMPT KISITLAMALARI ---
                # "TOP 5 Themes" ve "Max 2 Quotes" diyerek çıktının boyutunu kontrol altına alıyoruz.
                prompt = f"""
                You are InsightAI, an expert data analyst. Analyze the provided dataset (JSON).

                **CRITICAL RULES:**
                1. Output ONLY valid JSON. No markdown.
                2. Language: ENGLISH ONLY.
                3. **BE CONCISE.** Do not generate huge text blocks.

                TASKS:
                1. **Overview:** Executive summary (max 50 words).
                2. **Themes:** Identify the **TOP 5** most dominant themes.
                3. **Sub-themes:** For each theme, list top 3 sub-themes with counts.
                4. **Distribution:** Count mentions per Group.
                5. **Quotes:** Select **EXACTLY 2** representative quotes per theme (Total, not per group).

                OUTPUT FORMAT:
                {{
                    "overview": "Short summary...",
                    "themes": [
                        {{
                            "name": "Theme Title",
                            "definition": "Short definition",
                            "total_count": 100,
                            "sub_themes": [{{"name": "Sub A", "count": 10}}],
                            "group_distribution": {{"Group A": 50, "Group B": 50}},
                            "quotes": [
                                {{"text": "Quote 1...", "group": "Group A"}},
                                {{"text": "Quote 2...", "group": "Group B"}}
                            ]
                        }}
                    ]
                }}

                DATASET:
                {json.dumps(data_input, ensure_ascii=False)}
                """

                try:
                    response = model.generate_content(prompt)
                    
                    # Yanıtı temizle
                    cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
                    
                    # JSON Dönüşümü
                    result = json.loads(cleaned_text)
                    
                    st.success("Analysis Complete!")
                    
                    # --- GÖRSELLEŞTİRME ---
                    tab_overview, tab_breakdown = st.tabs(["📊 General Overview", "🎓 Field Breakdown"])

                    # TAB 1
                    with tab_overview:
                        st.markdown("### 📝 Executive Summary")
                        st.info(result.get("overview", "No summary."))
                        st.divider()

                        st.markdown("### 📉 Theme Landscape")
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

                        st.markdown("### 🧩 Theme Details")
                        for t in themes:
                            with st.expander(f"📌 {t['name']} ({t.get('total_count', 0)})", expanded=True):
                                st.write(f"_{t.get('definition', '')}_")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**Top Sub-Themes:**")
                                    for sub in t.get("sub_themes", []):
                                        if isinstance(sub, dict):
                                            st.markdown(f"• {sub.get('name')} ({sub.get('count')})")
                                        else:
                                            st.markdown(f"• {sub}")
                                with c2:
                                    st.markdown("**Key Quotes:**")
                                    for q in t.get("quotes", []):
                                        st.caption(f"🗣️ \"{q.get('text')}\" — *{q.get('group')}*")

                    # TAB 2
                    with tab_breakdown:
                        st.subheader("🔍 Breakdown by Field")
                        # Tüm grupları topla
                        all_groups = sorted(list(set(g for t in themes for g in t.get("group_distribution", {}).keys())))
                        
                        for group in all_groups:
                            with st.container():
                                st.markdown(f"## 🎓 {group}")
                                found = False
                                for t in themes:
                                    cnt = t.get("group_distribution", {}).get(group, 0)
                                    if cnt > 0:
                                        found = True
                                        st.markdown(f"**{t['name']}** ({cnt})")
                                        # İlerleme çubuğu
                                        total = t.get('total_count', 1) or 1
                                        st.progress(min(cnt / total, 1.0))
                                
                                if not found:
                                    st.caption("No major themes recorded for this group.")
                                st.divider()

                except json.JSONDecodeError:
                    st.error("⚠️ Data Too Large or Complex for Single Pass.")
                    st.error("The AI response was truncated. Please try reducing your dataset size manually or rely on the automated sampling.")
                    with st.expander("See Raw Output (Truncated)"):
                        st.text(cleaned_text)
                except Exception as e:
                    st.error(f"Processing Error: {e}")

    except Exception as e:
        st.error(f"File Error: {e}")

elif not api_key:
    st.info("👋 Enter API Key to start.")

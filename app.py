import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import time

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="InsightAI - Pro (Batch Processing)", layout="wide", page_icon="🟣")

# --- CSS ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    .stProgress > div > div > div > div {background-color: #6c5ce7;}
    </style>
""", unsafe_allow_html=True)

# --- 1. API GİRİŞİ ---
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 Authentication")
        api_key = st.text_input("Gemini API Key", type="password")

# --- BAŞLIK ---
st.title("🟣 InsightAI Pro")
st.markdown("### Large Scale Thematic Analysis (Batch Processing)")
st.info("ℹ️ This version splits large datasets into chunks to avoid AI output limits.")

# --- 2. DOSYA YÜKLEME ---
uploaded_file = st.file_uploader("Upload CSV (Unlimited Rows)", type=["csv"])

# --- YARDIMCI FONKSİYONLAR ---
def chunk_dataframe(df, chunk_size=300):
    """Veriyi belirtilen satır sayısına göre parçalar."""
    return [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]

# --- UYGULAMA MANTIĞI ---
if uploaded_file and api_key:
    try:
        # CSV OKUMA
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip')
        except:
            df = pd.read_csv(uploaded_file, sep=";", engine='python', on_bad_lines='skip')

        if len(df.columns) < 3:
            st.error("❌ Error: File needs 3 columns (ID, Major, Text).")
            st.stop()

        # VERİ TEMİZLEME
        df_clean = pd.DataFrame({
            "ID": df.iloc[:, 0].astype(str),
            "Group": df.iloc[:, 1].astype(str),
            "Text": df.iloc[:, 2].astype(str)
        })

        total_rows = len(df_clean)
        # 400 satırlık parçalara böl (Flash modeli için güvenli aralık)
        BATCH_SIZE = 400 
        chunks = chunk_dataframe(df_clean, BATCH_SIZE)
        
        st.write(f"📊 **Dataset:** {total_rows} rows | **Batches:** {len(chunks)} chunks of ~{BATCH_SIZE} rows.")

        if st.button("🚀 Start Batch Analysis", type="primary"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config={"response_mime_type": "application/json", "temperature": 0.0}
            )

            # --- AŞAMA 1: PARÇALI ANALİZ (MAP) ---
            intermediate_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(chunks):
                status_text.markdown(f"⏳ **Processing Batch {i+1}/{len(chunks)}...**")
                
                # Veriyi hazırla
                chunk_data = []
                for _, row in chunk.iterrows():
                    chunk_data.append({
                        "group": row["Group"].replace('"', "'").strip(), 
                        "text": row["Text"].replace('"', "'").replace("\n", " ").strip()
                    })
                
                # Ara Prompt (Sadece temaları ve sayıları çıkar)
                chunk_prompt = f"""
                Analyze this PARTIAL dataset (Batch {i+1}).
                Return valid JSON. English Only.

                TASKS:
                1. Identify main themes in this batch.
                2. Count theme frequency per Group.
                3. Select 1 best quote per theme.

                OUTPUT FORMAT:
                [
                    {{
                        "theme": "Theme Name",
                        "group_counts": {{"Math": 5, "Science": 3}},
                        "quote": {{"text": "...", "group": "..."}}
                    }}
                ]

                DATA:
                {json.dumps(chunk_data, ensure_ascii=False)}
                """
                
                try:
                    response = model.generate_content(chunk_prompt)
                    chunk_json = json.loads(response.text)
                    intermediate_results.extend(chunk_json) # Sonuçları havuza at
                except Exception as e:
                    st.error(f"Error in Batch {i+1}: {e}")
                
                # İlerleme çubuğunu güncelle
                progress_bar.progress((i + 1) / len(chunks))

            status_text.success("✅ All batches processed. Merging results...")

            # --- AŞAMA 2: BİRLEŞTİRME (REDUCE) ---
            # Şimdi tüm ara sonuçları tek bir prompt ile birleştiriyoruz.
            
            final_prompt = f"""
            You are an expert Data Analyst. I have analyzed a large dataset in batches. 
            Below is the list of ALL intermediate findings merged together.
            
            YOUR JOB:
            1. **Merge** similar themes (e.g., "Teacher Training" and "Educating Teachers" should be one).
            2. **Sum** the counts for the merged themes accurately.
            3. **Select** the best representative quotes from the list.
            4. **Finalize** the top 5 dominant themes.

            INPUT DATA (Intermediate Results):
            {json.dumps(intermediate_results, ensure_ascii=False)}

            FINAL OUTPUT FORMAT (Valid JSON, English):
            {{
                "overview": "Executive summary of the whole analysis...",
                "themes": [
                    {{
                        "name": "Final Theme Title",
                        "definition": "Description",
                        "total_count": 0,
                        "sub_themes": ["Sub 1", "Sub 2"], 
                        "group_distribution": {{"Group A": 0, "Group B": 0}},
                        "quotes": [
                            {{"text": "Quote...", "group": "Group A"}}
                        ]
                    }}
                ]
            }}
            """

            with st.spinner('Synthesizing final report...'):
                try:
                    final_response = model.generate_content(final_prompt)
                    final_result = json.loads(final_response.text)
                    
                    # --- SONUÇLARI GÖSTER (Eski kodla aynı yapı) ---
                    st.divider()
                    
                    tab1, tab2 = st.tabs(["📊 General Overview", "🎓 Detailed Breakdown"])
                    
                    # TAB 1
                    with tab1:
                        st.markdown("### 📝 Executive Summary")
                        st.info(final_result.get("overview"))
                        
                        themes = final_result.get("themes", [])
                        
                        # Grafik
                        chart_data = []
                        for t in themes:
                            for grp, count in t.get("group_distribution", {}).items():
                                chart_data.append({"Theme": t["name"], "Group": grp, "Count": count})
                        
                        if chart_data:
                            df_chart = pd.DataFrame(chart_data)
                            fig = px.bar(df_chart, x="Count", y="Theme", color="Group", orientation='h', 
                                         title="Consolidated Theme Distribution", text_auto=True,
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Kartlar
                        for t in themes:
                            with st.expander(f"📌 {t['name']} (Total: {t.get('total_count')})"):
                                st.write(f"_{t.get('definition')}_")
                                st.markdown("**Sub-Themes:** " + ", ".join(t.get("sub_themes", [])))
                                st.markdown("**Quotes:**")
                                for q in t.get("quotes", []):
                                    st.caption(f"🗣️ \"{q.get('text')}\" ({q.get('group')})")

                    # TAB 2
                    with tab2:
                        st.markdown("### 🔍 Field Breakdown")
                        all_groups = sorted(list(set(g for t in themes for g in t.get("group_distribution", {}).keys())))
                        for group in all_groups:
                            with st.container():
                                st.markdown(f"**🎓 {group}**")
                                for t in themes:
                                    cnt = t.get("group_distribution", {}).get(group, 0)
                                    if cnt > 0:
                                        st.write(f"- {t['name']}: **{cnt}**")
                                st.divider()

                except Exception as e:
                    st.error(f"Final Merge Error: {e}")
                    st.text(final_response.text) # Debug için

    except Exception as e:
        st.error(f"System Error: {e}")

elif not api_key:
    st.info("👋 Enter API Key to start.")

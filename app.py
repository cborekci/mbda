import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.express as px
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="InsightAI - Thematic Analysis", layout="wide", page_icon="📊")

# --- HEADER & TITLE ---
col1, col2 = st.columns([1, 5])
with col1:
    # Placeholder for a logo if you have one, or an emoji
    st.markdown("# 🟣") 
with col2:
    st.title("InsightAI")
    st.markdown("Automated Thematic Analysis & Field Segmentation")

# --- 1. API KEY MANAGEMENT ---
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 Authentication")
        api_key = st.text_input("Enter Google Gemini API Key", type="password")

# --- 2. DATA UPLOAD & SETTINGS (SIDEBAR) ---
with st.sidebar:
    st.divider()
    st.header("📂 Data Settings")
    
    separator = st.selectbox(
        "CSV Separator", 
        options=[";", ",", "\t"], 
        index=0, 
        help="Select ';' for Excel-exported CSVs (common in Europe/Turkey)."
    )
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.info("Tip: Ensure your CSV has a text column and a grouping column (e.g., Major/Department).")

# --- MAIN APP LOGIC ---
if uploaded_file and api_key:
    try:
        # LOAD DATA
        df = pd.read_csv(uploaded_file, sep=separator, engine='python', on_bad_lines='skip')
        
        # PREVIEW
        with st.expander("🔎 Preview Raw Data", expanded=False):
            st.dataframe(df.head(3), use_container_width=True)

        # COLUMN SELECTION
        st.divider()
        st.subheader("⚙️ Configuration")
        c1, c2 = st.columns(2)
        with c1:
            text_column = st.selectbox("Select Text Column (Feedback/Response)", df.columns)
        with c2:
            major_column = st.selectbox("Select Grouping Column (Major/Dept)", df.columns)

        # START ANALYSIS BUTTON
        if st.button("🚀 Start AI Analysis", type="primary"):
            genai.configure(api_key=api_key)
            # Using 1.5 Flash for speed and large context
            model = genai.GenerativeModel('gemini-1.5-flash')

            with st.spinner('InsightAI is processing your data, identifying themes, and generating visualizations...'):
                
                # PREPARE DATA FOR AI
                data_input = []
                for index, row in df.iterrows():
                    data_input.append({
                        "id": index,
                        "group": str(row[major_column]), 
                        "text": str(row[text_column])
                    })
                
                # --- PROMPT ENGINEERING (ENGLISH ENFORCED) ---
                prompt = f"""
                You are InsightAI, an expert qualitative data analyst. 
                Analyze the following dataset regardless of its original language.

                **CRITICAL RULE:** ALL OUTPUT MUST BE IN ENGLISH. TRANSLATE IF NECESSARY.

                TASKS:
                1. **General Overview:** Write a professional summary paragraph (approx 100 words) capturing the main sentiment and trends.
                2. **Thematic Coding:** Identify the main themes emerging from the participants' feedback.
                3. **Sub-themes:** For each main theme, identify 2-4 sub-themes.
                4. **Quantification:** Count how many times each theme is mentioned by each "Group" (Major/Department).
                5. **Direct Quotes:** Select impactful direct quotes for each theme. Always label which "Group" the quote came from.

                OUTPUT FORMAT (STRICT JSON ONLY):
                {{
                    "overview": "A concise, high-level executive summary of the entire analysis in English...",
                    "themes": [
                        {{
                            "id": 1,
                            "name": "Theme Title (e.g., Curriculum Gaps)",
                            "definition": "A short 1-sentence description of what this theme implies.",
                            "total_count": 45,
                            "sub_themes": ["Lack of practice", "Outdated books"],
                            "group_distribution": {{"Computer Eng": 30, "Architecture": 15}},
                            "quotes": [
                                {{"text": "We need more labs...", "group": "Computer Eng"}},
                                {{"text": "Design studios are short...", "group": "Architecture"}}
                            ]
                        }}
                    ]
                }}

                DATASET:
                {json.dumps(data_input, ensure_ascii=False)}
                """

                try:
                    # API CALL
                    response = model.generate_content(prompt)
                    
                    # REGEX CLEANING
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    cleaned_text = match.group(0) if match else response.text.replace("```json", "").replace("```", "").strip()
                    
                    # JSON PARSING
                    result = json.loads(cleaned_text)
                    
                    # --- DASHBOARD UI ---
                    st.success("Analysis Complete!")
                    
                    # TABS FOR ORGANIZATION
                    tab_overview, tab_breakdown = st.tabs(["📊 General Overview", "🎓 Field Comparison"])

                    # ==================================================
                    # TAB 1: GENERAL OVERVIEW
                    # ==================================================
                    with tab_overview:
                        
                        # 1. EXECUTIVE SUMMARY CARD
                        st.markdown("### 📝 Executive Summary")
                        st.info(result.get("overview", "No summary provided."))
                        
                        st.divider()

                        # 2. STACKED BAR CHART (HORIZONTAL)
                        st.markdown("### 📉 Dominant Themes")
                        
                        themes = result.get("themes", [])
                        chart_data = []
                        
                        for t in themes:
                            for grp, count in t.get("group_distribution", {}).items():
                                chart_data.append({
                                    "Theme": t["name"],
                                    "Group": grp,
                                    "Count": count
                                })
                        
                        if chart_data:
                            df_chart = pd.DataFrame(chart_data)
                            # Horizontal Stacked Bar Chart
                            fig = px.bar(
                                df_chart, 
                                x="Count", 
                                y="Theme", 
                                color="Group", 
                                orientation='h', # Horizontal
                                title="Distribution of Themes by Field",
                                text_auto=True,
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)

                        st.divider()

                        # 3. DETAILED THEME CARDS
                        st.markdown("### 🧩 Theme Details & Sub-breakdowns")
                        
                        for t in themes:
                            # Creating a "Card" look using container and border (if supported) or expander
                            with st.expander(f"📌 {t['name']} (Total Mentions: {t['total_count']})", expanded=True):
                                st.markdown(f"*{t.get('definition', '')}*")
                                
                                c_sub, c_quotes = st.columns([1, 2])
                                
                                with c_sub:
                                    st.markdown("**Sub-Themes:**")
                                    for sub in t.get("sub_themes", []):
                                        st.markdown(f"• {sub}")
                                
                                with c_quotes:
                                    st.markdown("**Key Voices:**")
                                    for q in t.get("quotes", []):
                                        st.markdown(f"> \"{q['text']}\"")
                                        st.caption(f"— {q['group']}")

                    # ==================================================
                    # TAB 2: FIELD COMPARISON
                    # ==================================================
                    with tab_breakdown:
                        st.subheader("🔍 Filter by Field (Major)")
                        
                        # Get unique groups
                        all_groups = set()
                        for t in themes:
                            all_groups.update(t.get("group_distribution", {}).keys())
                        
                        selected_group = st.selectbox("Select a Field to Deep Dive:", list(all_groups))

                        if selected_group:
                            st.markdown(f"### Results for: **{selected_group}**")
                            
                            found_data = False
                            for t in themes:
                                count = t.get("group_distribution", {}).get(selected_group, 0)
                                if count > 0:
                                    found_data = True
                                    with st.container():
                                        st.markdown(f"#### {t['name']}")
                                        st.progress(count / t['total_count'] if t['total_count'] > 0 else 0)
                                        st.write(f"Frequency in this field: **{count}**")
                                        
                                        # Filter quotes for this specific group
                                        group_quotes = [q['text'] for q in t.get("quotes", []) if q.get("group") == selected_group]
                                        
                                        if group_quotes:
                                            for gq in group_quotes:
                                                st.info(f"🗣️ \"{gq}\"")
                                        st.divider()
                            
                            if not found_data:
                                st.warning(f"No specific data points found for {selected_group}.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.markdown("Check if the API Key is valid or if the data size is too large.")

    except Exception as e:
        st.error("Error reading the file.")
        st.warning("Try changing the 'CSV Separator' in the sidebar.")
        st.error(f"Details: {e}")

elif not api_key:
    st.info("👋 Welcome to InsightAI. Please enter your API Key in the sidebar to begin.")

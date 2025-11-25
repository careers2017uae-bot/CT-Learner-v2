"""
CT Learner Pro - Enhanced UX with Data Analytics & Sentence Highlighting
Single-file Streamlit app with improved HCI principles.
"""

import os
import io
import re
import json
import math
import tempfile
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# File extraction
import docx
import pdfplumber

# NLP & HF
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------
# Configuration & Lexica
# ---------------------
DEFAULT_HF_MODEL = "j-hartmann/emotion-english-roberta-large"

# Color schemes for better UX
COLOR_SCHEME = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "warning": "#ffbb78",
    "danger": "#d62728",
    "info": "#17becf",
    "light": "#f8f9fa",
    "dark": "#343a40"
}

# ---------------------
# Paul's Critical Thinking Rubric with highlighting patterns
# ---------------------
PAUL_CT_RUBRIC = {
    "Clarity": {
        "description": "Demonstrate clarity in conversation; provide examples to illustrate the point as appropriate.",
        "feedback_q": "Could you elaborate further; give an example or illustrate what you mean?",
        "patterns": ["for example", "for instance", "e.g.", "such as", "to illustrate", "in other words", "specifically"],
        "color": "#FF6B6B"
    },
    "Accuracy": {
        "description": "Provide accurate and verifiable information to support the ideas/position.",
        "feedback_q": "How could we check on that; verify or test; find out if that is true?",
        "patterns": ["http", "www.", "cite", "according to", "%", "data", "study", "research", "survey", "statistics", "source"],
        "color": "#4ECDC4"
    },
    "Relevance": {
        "description": "Respond to the issues/question/problem with related information. Avoid irrelevant details.",
        "feedback_q": "How does that relate to the problem; bear on the question; help us with the issue?",
        "patterns": ["related to", "regarding", "pertaining to", "in relation to", "connected to"],
        "color": "#45B7D1"
    },
    "Significance": {
        "description": "Able to identify the central idea. Contribute with important and new points.",
        "feedback_q": "Is this the most important problem to consider? Which of these facts are most important?",
        "patterns": ["main", "central", "important", "key", "primary", "crucial", "essential", "significant"],
        "color": "#96CEB4"
    },
    "Logic": {
        "description": "Organize each piece of information in a logical order so it makes sense to others.",
        "feedback_q": "Does all this make sense together? Does what you say follow from the evidence?",
        "patterns": ["therefore", "because", "thus", "hence", "however", "but", "consequently", "as a result", "so that"],
        "color": "#FFEAA7"
    },
    "Precision": {
        "description": "Select specific information, stay focused and avoid redundancy.",
        "feedback_q": "Could you be more specific; be more exact; give more details?",
        "patterns": ["specifically", "exactly", "precisely", "in particular", "specifically", "detailed"],
        "color": "#DDA0DD"
    },
    "Fairness": {
        "description": "Demonstrate open-mindedness, consider pros and cons and challenge assumptions.",
        "feedback_q": "Am I sympathetically representing the viewpoints of others? Do I have vested interests?",
        "patterns": ["on the other hand", "although", "consider", "pros and cons", "however", "both", "despite", "alternatively"],
        "color": "#98D8C8"
    },
    "Depth": {
        "description": "Being thorough; examine the intricacies in the argument.",
        "feedback_q": "What are some of the complexities of this question? What difficulties must we deal with?",
        "patterns": ["because", "although", "since", "whereas", "in depth", "intricacy", "complex", "complexity", "thorough"],
        "color": "#F7DC6F"
    },
    "Breadth": {
        "description": "Able to offer / consider alternative views or solutions.",
        "feedback_q": "Do we need another perspective? What are alternative ways?",
        "patterns": ["alternatively", "another view", "different perspective", "other view", "in contrast", "on the contrary"],
        "color": "#BB8FCE"
    }
}

# ---------------------
# Enhanced Helper Functions
# ---------------------
def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def extract_text_from_docx_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        f.write(b); f.flush()
        tmp = f.name
    try:
        doc = docx.Document(tmp)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

def extract_text_from_pdf_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b); f.flush()
        tmp = f.name
    try:
        text_pages = []
        with pdfplumber.open(tmp) as pdf:
            for p in pdf.pages:
                text_pages.append(p.extract_text() or "")
        return "\n".join(text_pages)
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_split(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"\w+['-]?\w*|\w+", s.lower())

# ---------------------
# Enhanced CT Rubric with Sentence Highlighting
# ---------------------
def highlight_ct_sentences(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns dictionary mapping CT standard -> list of (sentence, color)
    """
    highlighted = {standard: [] for standard in PAUL_CT_RUBRIC.keys()}
    sents = sentence_split(text)
    
    for sent in sents:
        sent_lower = sent.lower()
        for standard, data in PAUL_CT_RUBRIC.items():
            for pattern in data["patterns"]:
                if pattern in sent_lower:
                    highlighted[standard].append((sent, data["color"]))
                    break  # Only highlight once per standard per sentence
    
    return highlighted

def heuristic_ct_scores(text: str) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[Tuple[str, str]]]]:
    """
    Enhanced to return highlighted sentences
    """
    sents = sentence_split(text)
    tokens = tokenize_simple(text)
    word_count = len(tokens)
    scores = {}
    suggestions = {}
    
    # Get highlighted sentences
    highlighted = highlight_ct_sentences(text)
    
    # Calculate scores (same as before)
    clarity_indicators = ["for example", "for instance", "e.g.", "such as", "to illustrate"]
    clarity_score = 1.0 if any(phrase in text.lower() for phrase in clarity_indicators) else (0.3 if word_count < 50 else 0.5)
    scores["Clarity"] = clarity_score
    suggestions["Clarity"] = PAUL_CT_RUBRIC["Clarity"]["feedback_q"]

    accuracy_indicators = ["http", "www.", "cite", "according to", "%", "data", "study", "reported", "survey"]
    accuracy_score = 1.0 if any(ind in text.lower() for ind in accuracy_indicators) else 0.4
    scores["Accuracy"] = accuracy_score
    suggestions["Accuracy"] = PAUL_CT_RUBRIC["Accuracy"]["feedback_q"]

    if sents:
        first = tokenize_simple(sents[0])
        overlap_counts = sum(1 for sent in sents[1:] if any(w in tokenize_simple(sent) for w in first[:5]))
        relevance_score = min(1.0, (overlap_counts+1) / max(1, len(sents)))
    else:
        relevance_score = 0.0
    scores["Relevance"] = relevance_score
    suggestions["Relevance"] = PAUL_CT_RUBRIC["Relevance"]["feedback_q"]

    sign_ind = ["main", "central", "important", "key", "primary"]
    sign_score = 1.0 if any(w in text.lower() for w in sign_ind) else min(0.9, 0.6 + 0.01 * (word_count/100))
    scores["Significance"] = sign_score
    suggestions["Significance"] = PAUL_CT_RUBRIC["Significance"]["feedback_q"]

    connectors = ["therefore", "because", "thus", "hence", "however", "but", "consequently", "as a result", "so that"]
    logic_score = min(1.0, sum(1 for c in connectors if c in text.lower()) * 0.25)
    scores["Logic"] = logic_score
    suggestions["Logic"] = PAUL_CT_RUBRIC["Logic"]["feedback_q"]

    hedges = ["maybe", "perhaps", "might", "could", "seems", "appears"]
    precision_score = max(0.0, 1.0 - 0.2 * sum(1 for h in hedges if h in text.lower()))
    if word_count < 40:
        precision_score *= 0.5
    scores["Precision"] = precision_score
    suggestions["Precision"] = PAUL_CT_RUBRIC["Precision"]["feedback_q"]

    fairness_ind = ["on the other hand", "although", "consider", "pros and cons", "however", "both", "despite"]
    fairness_score = 1.0 if any(p in text.lower() for p in fairness_ind) else 0.45
    scores["Fairness"] = fairness_score
    suggestions["Fairness"] = PAUL_CT_RUBRIC["Fairness"]["feedback_q"]

    depth_ind = ["because", "although", "since", "whereas", "in depth", "intricacy", "complex", "complexity"]
    depth_score = min(1.0, 0.25 * sum(1 for d in depth_ind if d in text.lower()) + 0.3)
    scores["Depth"] = depth_score
    suggestions["Depth"] = PAUL_CT_RUBRIC["Depth"]["feedback_q"]

    breadth_ind = ["alternatively", "another view", "different perspective", "other view", "in contrast"]
    breadth_score = 1.0 if any(p in text.lower() for p in breadth_ind) else 0.4
    scores["Breadth"] = breadth_score
    suggestions["Breadth"] = PAUL_CT_RUBRIC["Breadth"]["feedback_q"]

    for k in scores:
        scores[k] = float(max(0.0, min(1.0, scores[k])))
    
    return scores, suggestions, highlighted

# ---------------------
# Data Visualization Functions
# ---------------------
def create_ct_heatmap(ct_scores_list: List[Dict[str, float]], filenames: List[str]) -> go.Figure:
    """Create heatmap of CT scores across all submissions"""
    standards = list(PAUL_CT_RUBRIC.keys())
    scores_matrix = []
    
    for ct_scores in ct_scores_list:
        row = [ct_scores.get(std, 0) for std in standards]
        scores_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=scores_matrix,
        x=standards,
        y=filenames,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Critical Thinking Scores Heatmap",
        xaxis_title="CT Standards",
        yaxis_title="Submissions",
        height=400
    )
    return fig

def create_comparison_bar_chart(ct_scores: Dict[str, float], student_name: str) -> go.Figure:
    """Create bar chart comparing CT scores"""
    standards = list(ct_scores.keys())
    scores = list(ct_scores.values())
    
    colors = ['crimson' if x < 0.5 else 'steelblue' for x in scores]
    
    fig = go.Figure(data=[
        go.Bar(x=standards, y=scores, marker_color=colors)
    ])
    
    fig.update_layout(
        title=f"Critical Thinking Analysis - {student_name}",
        xaxis_title="CT Standards",
        yaxis_title="Score (0-1)",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

# ---------------------
# Transformer Model (cached) - Keeping for potential future use
# ---------------------
@st.cache_resource
def load_transformer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return {"tok": tokenizer, "model": model, "device": device}
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {e}")
        raise

def safe_extract_all_files(files) -> List[Dict[str, Any]]:
    out = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        try:
            b = f.read()
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(b)
            elif name.lower().endswith(".docx"):
                text = extract_text_from_docx_bytes(b)
            else:
                text = extract_text_from_txt_bytes(b)
            text = clean_text(text)
            if not text:
                st.warning(f"Warning: extracted empty text from {name}. If this is a scanned PDF, OCR is required.")
            out.append({"filename": name, "text": text})
        except Exception as e:
            st.error(f"Failed to extract {name}: {e}")
            out.append({"filename": name, "text": ""})
    return out

# ---------------------
# Enhanced Streamlit UI with Better UX
# ---------------------
def main():
    # Page configuration with custom theme
    st.set_page_config(
        page_title="CT Learner Pro", 
        layout="wide", 
        initial_sidebar_state="expanded",
        page_icon="🧠"
    )
    
    # Custom CSS for better styling
    st.markdown(f"""
    <style>
    .main-header {{
        font-size: 2.5rem;
        color: {COLOR_SCHEME['primary']};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {COLOR_SCHEME['light']};
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLOR_SCHEME['primary']};
        margin: 0.5rem 0;
    }}
    .highlight-sentence {{
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid;
    }}
    .progress-bar {{
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin: 0.5rem 0;
    }}
    .progress-fill {{
        height: 100%;
        border-radius: 4px;
        background-color: {COLOR_SCHEME['primary']};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header with better visual design
    st.markdown('<h1 class="main-header">🧠 CT Learner Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Advanced Critical Thinking Analysis for Student Submissions
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with improved organization
    with st.sidebar:
        st.header("📁 Upload & Settings")
        
        # File upload with better feedback
        uploaded = st.file_uploader(
            "Choose student submissions", 
            accept_multiple_files=True, 
            type=['txt','pdf','docx'],
            help="Upload multiple files for batch analysis"
        )
        
        st.markdown("---")
        st.subheader("⚙️ Analysis Settings")
        
        # Model settings in expander - simplified since emotion analysis removed
        with st.expander("Advanced Configuration", expanded=False):
            batch_size = st.number_input("Processing batch size", value=8, min_value=1, max_value=64)
        
        # System info
        st.markdown("---")
        st.subheader("💻 System Info")
        device_status = "✅ GPU Available" if torch.cuda.is_available() else "ℹ️ CPU Mode"
        st.write(f"**Device:** {device_status}")
        
        # Run button with prominent styling
        st.markdown("---")
        run_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    # Interactive Rubric Explorer
    with st.sidebar.expander("🎯 CT Rubric Guide", expanded=False):
        selected_standard = st.selectbox("Explore standards:", list(PAUL_CT_RUBRIC.keys()))
        if selected_standard:
            std_data = PAUL_CT_RUBRIC[selected_standard]
            st.markdown(f"**Description:** {std_data['description']}")
            st.markdown(f"**Feedback Prompt:** {std_data['feedback_q']}")
            st.markdown(f"**Color Code:** `{std_data['color']}`")
            st.markdown("**Patterns:** " + ", ".join(f"`{p}`" for p in std_data['patterns']))
    
    # Main workflow with better progress indicators
    if run_btn:
        if not uploaded:
            st.error("❌ Please upload at least one file to begin analysis.")
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: File processing
        status_text.text("📂 Processing uploaded files...")
        submissions = safe_extract_all_files(uploaded)
        progress_bar.progress(30)
        
        texts = [s["text"] for s in submissions]
        
        # Step 2: CT analysis with highlighting
        status_text.text("💭 Evaluating critical thinking...")
        ct_scores_all = []
        ct_suggestions_all = []
        ct_highlights_all = []
        for t in texts:
            s, sug, highlights = heuristic_ct_scores(t)
            ct_scores_all.append(s)
            ct_suggestions_all.append(sug)
            ct_highlights_all.append(highlights)
        progress_bar.progress(80)
        
        # Step 3: Data assembly
        status_text.text("📊 Compiling results...")
        results_data = []
        rows = []
        for meta, ct_scores, ct_suggest, ct_highlights in zip(
            submissions, ct_scores_all, ct_suggestions_all, ct_highlights_all):
            
            row = {
                "filename": meta.get("filename", "untitled"),
                "word_count": len(meta.get("text", "").split()),
                "avg_ct_score": np.mean(list(ct_scores.values())) if ct_scores else 0.0,
                "ct_scores": json.dumps(ct_scores),
                "ct_suggestions": json.dumps(ct_suggest),
                "text_preview": meta.get("text","")[:500]
            }
            rows.append(row)
            results_data.append((meta, ct_scores, ct_suggest, ct_highlights))
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Results section with tabs for better organization
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Submissions", "📤 Export"])
        
        with tab1:
            # Dashboard with key metrics and visualizations
            st.subheader("Executive Summary")
            
            # Key metrics - removed emotion-related metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_ct = np.mean([np.mean(list(s.values())) for s in ct_scores_all])
                st.metric("Average CT Score", f"{avg_ct:.2f}")
            with col2:
                total_words = sum(len(s["text"].split()) for s in submissions)
                st.metric("Total Words", f"{total_words:,}")
            with col3:
                high_ct = sum(1 for scores in ct_scores_all if np.mean(list(scores.values())) > 0.7)
                st.metric("High CT Scores", f"{high_ct}/{len(submissions)}")
            
            # Visualizations - removed emotion charts
            st.markdown("#### 📊 CT Scores Overview")
            fig = create_ct_heatmap(ct_scores_all, [s["filename"] for s in submissions])
            st.plotly_chart(fig, use_container_width=True)
            
            # CT standards performance
            st.markdown("#### 🎯 CT Standards Performance")
            standards_avg = {}
            for std in PAUL_CT_RUBRIC.keys():
                std_scores = [s.get(std, 0) for s in ct_scores_all]
                standards_avg[std] = np.mean(std_scores) if std_scores else 0
            
            fig = px.bar(x=list(standards_avg.keys()), y=list(standards_avg.values()),
                        title="Average Scores by CT Standard",
                        labels={'x': 'CT Standard', 'y': 'Average Score'})
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Detailed submission analysis
            st.subheader("Detailed Submission Analysis")
            
            for i, (meta, ct_scores, ct_suggest, ct_highlights) in enumerate(results_data):
                with st.expander(f"📄 {i+1}. {meta.get('filename','untitled')}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### 📖 Text with CT Highlights")
                        text = meta.get("text", "")
                        
                        # Display text with CT highlights
                        for sent in sentence_split(text):
                            highlighted = False
                            for standard, sentences in ct_highlights.items():
                                if any(sent == h_sent for h_sent, _ in sentences):
                                    color = PAUL_CT_RUBRIC[standard]["color"]
                                    st.markdown(
                                        f'<div class="highlight-sentence" style="border-left-color: {color}; background-color: {color}20;">'
                                        f'<strong>{standard}:</strong> {sent}'
                                        f'</div>', 
                                        unsafe_allow_html=True
                                    )
                                    highlighted = True
                                    break
                            if not highlighted:
                                st.write(sent)
                    
                    with col2:
                        # CT scores
                        st.markdown("#### 💭 CT Scores")
                        fig = create_comparison_bar_chart(ct_scores, meta.get('filename', 'Student'))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feedback suggestions
                        st.markdown("#### 💡 Feedback Suggestions")
                        for standard, suggestion in ct_suggest.items():
                            score = ct_scores.get(standard, 0)
                            if score < 0.6:
                                st.info(f"**{standard}:** {suggestion}")
        
        with tab3:
            # Export section
            st.subheader("📤 Export Results")
            
            # Create comprehensive DataFrame
            df_summary = pd.DataFrame(rows)
            
            # Display preview
            st.markdown("#### Preview of Export Data")
            st.dataframe(df_summary[["filename", "word_count", "avg_ct_score", "text_preview"]])
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_bytes = df_summary.to_csv(index=False).encode("utf-8")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "📥 Download CSV", 
                    data=csv_bytes, 
                    file_name=f"ctlearner_results_{timestamp}.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel Export
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                    # Main results
                    df_summary.to_excel(writer, index=False, sheet_name="Results")
                    
                    # CT scores detailed
                    ct_details = []
                    for i, (meta, ct_scores, ct_suggest) in enumerate(zip(submissions, ct_scores_all, ct_suggestions_all)):
                        for standard, score in ct_scores.items():
                            ct_details.append({
                                "Filename": meta["filename"],
                                "CT_Standard": standard,
                                "Score": score,
                                "Suggestion": ct_suggest[standard]
                            })
                    pd.DataFrame(ct_details).to_excel(writer, index=False, sheet_name="CT_Details")
                
                st.download_button(
                    "📊 Download Excel", 
                    data=towrite.getvalue(), 
                    file_name=f"ctlearner_results_{timestamp}.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Additional export options
            st.markdown("#### Additional Reports")
            
            # CT Improvement Report
            if st.button("📋 Generate CT Improvement Report", use_container_width=True):
                improvement_data = []
                for meta, ct_scores in zip(submissions, ct_scores_all):
                    weak_areas = [std for std, score in ct_scores.items() if score < 0.6]
                    strong_areas = [std for std, score in ct_scores.items() if score >= 0.7]
                    
                    improvement_data.append({
                        "Filename": meta["filename"],
                        "Overall_CT_Score": np.mean(list(ct_scores.values())),
                        "Weak_Areas": ", ".join(weak_areas) if weak_areas else "None",
                        "Strong_Areas": ", ".join(strong_areas) if strong_areas else "None",
                        "Priority_Level": "High" if len(weak_areas) > 3 else "Medium" if len(weak_areas) > 1 else "Low"
                    })
                
                improvement_df = pd.DataFrame(improvement_data)
                st.dataframe(improvement_df, use_container_width=True)
                
                # Download improvement report
                csv_improvement = improvement_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Improvement Report", 
                    data=csv_improvement, 
                    file_name=f"ct_improvement_report_{timestamp}.csv", 
                    mime="text/csv"
                )

        st.success("🎉 Analysis complete! Explore the results in the tabs above.")
        
    else:
        # Welcome state - show when no analysis has been run
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Getting Started")
            st.markdown("""
            1. **Upload** student submissions (TXT, PDF, or DOCX)
            2. **Configure** analysis settings in the sidebar
            3. **Click** 'Start Analysis' to begin processing
            4. **Explore** results in the interactive dashboard
            
            ### 📊 What You'll Get:
            - **Critical Thinking Assessment**: Automated scoring using Paul's Rubric
            - **Sentence Highlighting**: Visual indicators of CT standards in text
            - **Interactive Visualizations**: Charts and heatmaps for data insights
            - **Exportable Reports**: CSV and Excel downloads for further analysis
            - **Actionable Feedback**: Specific improvement suggestions for students
            """)
        
        with col2:
            st.subheader("🎯 CT Standards Covered")
            for standard in list(PAUL_CT_RUBRIC.keys())[:5]:
                st.markdown(f"✅ **{standard}**")
            if len(PAUL_CT_RUBRIC) > 5:
                with st.expander("See all standards"):
                    for standard in list(PAUL_CT_RUBRIC.keys())[5:]:
                        st.markdown(f"✅ **{standard}**")

# Run the app
if __name__ == "__main__":
    main()

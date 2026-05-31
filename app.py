from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from disease_profiles import get_profile, severity_level
from model import MissingModelDependency, generate_gradcam_plus_plus, load_hybrid_model, predict_leaf

import urllib.request

MODEL_URL = "https://huggingface.co/Mesbah74/aleaf-stenet-model/resolve/main/aleaf_best.pth"

if not os.path.exists("aleaf_best.pth"):
    urllib.request.urlretrieve(
        MODEL_URL,
        "aleaf_best.pth"
    )


MODEL_PATH = Path(os.getenv("ALEAF_MODEL_PATH", "aleaf_best.pth"))
RESEARCH_TITLE = (
    "ALeaf-STENet: Hybrid Deep Learning Framework for Apple Leaf Disease "
    "Detection and Severity Assessment"
)
FOOTER_TEXT = (
    "ALeaf-STENet: Advanced Severity Tracking via Ensemble Networks | "
    "Developed by Mesbah Uddin Bhuiyan"
)


st.set_page_config(
    page_title="ALeaf-STENet: Apple Leaf Disease Diagnosis",
    page_icon="AL",
    layout="wide",
    initial_sidebar_state="expanded",
)


SCAB_PROFILE = get_profile("Scab")
DEFAULT_DIAGNOSIS = {
    "class_name": "Scab",
    "diseaseName": "Scab",
    "scientificName": "Venturia inaequalis",
    "confidence": 96.4,
    "severity": 42.0,
    "severityLevel": "Moderate",
    "symptoms": SCAB_PROFILE["symptoms"],
    "treatment": {
        "preventive": SCAB_PROFILE["preventive"],
        "organic": SCAB_PROFILE["organic"],
        "chemical": SCAB_PROFILE["chemical"],
    },
    "gradCamMetadata": {
        "description": (
            "STENet high-attention coefficients isolate significant visual anomalies "
            "near the mid-rib and lower blade segments."
        ),
        "hotspots": [
            {"x": 38, "y": 44, "radius": 22, "intensity": 0.95},
            {"x": 65, "y": 55, "radius": 18, "intensity": 0.75},
            {"x": 50, "y": 28, "radius": 12, "intensity": 0.60},
        ],
    },
    "researchNotes": SCAB_PROFILE["research_notes"],
    "top_predictions": [
        {"class_name": "Scab", "confidence": 96.4},
        {"class_name": "Rust", "confidence": 2.1},
        {"class_name": "Brown spot", "confidence": 1.0},
    ],
    "device": "demo",
    "val_acc": 98.39963420210334,
}


def init_state() -> None:
    defaults = {
        "active_tab": "Research",
        "previous_tab": None,
        "diagnosis": DEFAULT_DIAGNOSIS,
        "selected_leaf_name": "",
        "selected_leaf_uri": None,
        "selected_leaf_bytes": None,
        "last_upload_hash": None,
        "model_error": None,
        "validation_passed": None,
        "validation_message": "",
        "gradcam_uri": None,
        "gradcam_bytes": None,
        "gradcam_error": None,
        "arch_image_uri": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def leaf_svg(size: int = 28) -> str:
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M50 14C66 28 86 45 50 88C14 45 34 28 50 14Z" stroke="currentColor" stroke-width="6" fill="rgba(16,185,129,.16)"/>
      <path d="M50 14C50 36 50 61 50 88" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-dasharray="8 8"/>
    </svg>
    """


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
        :root {
          --bg: #040D0B;
          --panel: rgba(10, 25, 14, 0.68);
          --panel-strong: #061612;
          --panel-soft: #0A1F1B;
          --border: #1A3A32;
          --emerald: #10B981;
          --emerald-soft: #2da06e;
          --teal: #14B8A6;
          --text: #e2e8f0;
          --muted: #94a3b8;
        }
        * { box-sizing: border-box; }
        html, body, [data-testid="stAppViewContainer"], .stApp {
          background: var(--bg);
          color: var(--text);
          font-family: "Space Grotesk", "Inter", sans-serif;
        }
        [data-testid="stHeader"], #MainMenu, footer { visibility: hidden; height: 0; }
        [data-testid="stSidebar"] {
          background: #061612;
          border-right: 1px solid var(--border);
          font-family: "JetBrains Mono", monospace;
        }
        [data-testid="stSidebar"] [class*="material"],
        [data-testid="stSidebar"] button span,
        [data-testid="stSidebar"] [data-testid*="Icon"] {
          font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
          font-feature-settings: "liga" !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
          background: rgba(10, 31, 27, .55);
          border: 1px solid rgba(26, 58, 50, .9);
          border-radius: 12px;
          padding: 12px 14px;
          margin-bottom: 10px;
          color: #cbd5e1;
          transition: all .2s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
          border-color: rgba(16,185,129,.75);
          color: #34d399;
          background: rgba(16,185,129,.08);
        }
        .block-container { padding-top: 1rem; padding-bottom: 2.2rem; max-width: 1280px; }
        [data-testid="column"] { min-width: 0; }
        h1, h2, h3, h4, h5, h6 {
          font-family: "JetBrains Mono", monospace;
          letter-spacing: 0;
        }
        .topbar {
          min-height: 64px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          background: #061612;
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 12px 18px;
          box-shadow: 0 18px 42px rgba(0,0,0,.32);
          position: sticky;
          top: 0;
          z-index: 30;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .logo-mark {
          width: 42px;
          height: 42px;
          display: grid;
          place-items: center;
          border-radius: 10px;
          color: #020605;
          background: linear-gradient(135deg, #34d399, #0d9488);
          box-shadow: 0 12px 28px rgba(16,185,129,.22);
        }
        .brand-name {
          display: flex;
          align-items: center;
          gap: 8px;
          font-family: "JetBrains Mono", monospace;
          font-size: 20px;
          font-weight: 900;
          color: white;
          line-height: 1;
        }
        .version {
          font-size: 10px;
          color: #34d399;
          background: rgba(16,185,129,.12);
          border: 1px solid rgba(16,185,129,.28);
          padding: 3px 6px;
          border-radius: 6px;
          font-weight: 800;
        }
        .subtitle {
          margin-top: 4px;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px;
          color: #34d399;
          letter-spacing: .12em;
          text-transform: uppercase;
          font-weight: 800;
          opacity: .86;
        }
        .status-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
        .status-pill {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          border-radius: 8px;
          border: 1px solid rgba(16,185,129,.28);
          background: rgba(6, 78, 59, .34);
          color: #a7f3d0;
          padding: 6px 10px;
          font-family: "JetBrains Mono", monospace;
          font-size: 11px;
          font-weight: 800;
        }
        .pulse-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #34d399;
          box-shadow: 0 0 0 rgba(52,211,153,.8);
          animation: pulse 1.7s infinite;
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(52,211,153,.62); }
          70% { box-shadow: 0 0 0 9px rgba(52,211,153,0); }
          100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
        }
        .glass-panel {
          background: var(--panel);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(26, 58, 50, .92);
          border-radius: 18px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, .37);
        }
        .hero {
          position: relative;
          overflow: hidden;
          padding: 42px;
          border-radius: 24px;
          border: 1px solid var(--border);
          background:
            radial-gradient(circle at 20% 20%, rgba(16,185,129,.13), transparent 32%),
            linear-gradient(180deg, #0A1F1B 0%, #040D0B 100%);
          box-shadow: 0 22px 60px rgba(0,0,0,.36);
        }
        .hero:before {
          content: "";
          position: absolute;
          inset: 0;
          background-image: radial-gradient(#10B981 1px, transparent 1px);
          background-size: 16px 16px;
          opacity: .12;
          pointer-events: none;
        }
        .hero > * { position: relative; z-index: 1; }
        .eyebrow {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border: 1px solid rgba(52,211,153,.25);
          background: rgba(16,185,129,.10);
          color: #a7f3d0;
          border-radius: 999px;
          padding: 6px 12px;
          font-family: "JetBrains Mono", monospace;
          font-size: 12px;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: .04em;
        }
        .hero-title {
          margin: 22px 0 8px 0;
          color: white;
          font-family: "JetBrains Mono", monospace;
          font-size: clamp(34px, 6vw, 56px);
          line-height: 1.02;
          font-weight: 900;
        }
        .hero-subtitle {
          color: rgba(52,211,153,.92);
          font-size: 19px;
          font-weight: 650;
          margin-bottom: 16px;
        }
        .body-copy { color: #cbd5e1; font-size: 14px; line-height: 1.72; }
        .metric-card, .feature-card, .dark-card {
          background: rgba(0,0,0,.28);
          border: 1px solid rgba(255,255,255,.06);
          border-radius: 14px;
          padding: 16px;
        }
        .feature-card {
          min-height: 218px;
          background: rgba(10, 25, 14, .65);
          border-color: rgba(26, 58, 50, .95);
        }
        .feature-icon {
          width: 48px;
          height: 48px;
          display: grid;
          place-items: center;
          border: 1px solid rgba(16,185,129,.22);
          color: #34d399;
          background: rgba(16,185,129,.10);
          border-radius: 12px;
          margin-bottom: 16px;
        }
        .section-label {
          display: flex;
          align-items: center;
          gap: 8px;
          color: var(--emerald-soft);
          font-family: "JetBrains Mono", monospace;
          font-size: 12px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: .12em;
          margin-bottom: 12px;
        }
        .section-label.no-margin { margin-bottom: 0; }
        .diagnosis-card {
          padding: 20px;
          overflow: hidden;
        }
        .diagnosis-card-head {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 14px;
        }
        .card-subtext {
          color: #94a3b8;
          font-family: "Space Grotesk", "Inter", sans-serif;
          font-size: 12px;
          line-height: 1.45;
          margin-top: 5px;
        }
        .status-badge {
          flex: 0 0 auto;
          border: 1px solid rgba(16,185,129,.25);
          border-radius: 8px;
          background: rgba(16,185,129,.10);
          color: #a7f3d0;
          font-family: "JetBrains Mono", monospace;
          font-size: 9px;
          font-weight: 900;
          letter-spacing: .08em;
          text-transform: uppercase;
          padding: 5px 8px;
          white-space: nowrap;
        }
        [data-testid="stFileUploader"] {
          width: 100%;
          background: rgba(4,13,11,.95);
          border: 2px dashed rgba(16,185,129,.30);
          border-radius: 14px;
          padding: 16px;
        }
        [data-testid="stFileUploader"] section {
          width: 100%;
          border: 0;
          padding: 4px;
        }
        [data-testid="stFileUploader"] button, .stButton button, .stDownloadButton button {
          background: linear-gradient(90deg, #10b981, #2dd4bf);
          color: #020605;
          border: 0;
          border-radius: 12px;
          font-family: "JetBrains Mono", monospace;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: .04em;
          box-shadow: 0 14px 28px rgba(16,185,129,.14);
          white-space: normal;
          min-height: 42px;
        }
        .stButton button:hover, .stDownloadButton button:hover, [data-testid="stFileUploader"] button:hover {
          background: linear-gradient(90deg, #34d399, #5eead4);
          color: #020605;
          border: 0;
        }
        .specimen-frame {
          aspect-ratio: 1 / 1;
          width: 100%;
          background: rgba(0,0,0,.62);
          border: 1px solid var(--border);
          border-radius: 14px;
          overflow: hidden;
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 280px;
        }
        .specimen-frame img { width: 100%; height: 100%; object-fit: cover; }
        .scan-line {
          position: absolute;
          left: 0;
          right: 0;
          top: 0;
          height: 2px;
          background: #34d399;
          box-shadow: 0 0 18px rgba(52,211,153,.8);
          opacity: .55;
          animation: scan 2.2s ease-in-out infinite;
        }
        @keyframes scan {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(275px); }
        }
        .placeholder {
          color: #64748b;
          text-align: center;
          padding: 28px;
          font-family: "JetBrains Mono", monospace;
        }
        .meta-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          padding: 10px;
          background: #020605;
          border: 1px solid var(--border);
          border-radius: 10px;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px;
        }
        .meta-grid span { color: #64748b; }
        .meta-grid strong {
          display: block;
          color: #34d399;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .result-header {
          display: grid;
          grid-template-columns: 2fr 1fr 1fr;
          gap: 18px;
          padding: 24px;
          border-radius: 18px;
          border: 1px solid var(--border);
          background: linear-gradient(90deg, #0C241E, #061612);
          box-shadow: 0 8px 32px rgba(0,0,0,.35);
        }
        .result-title {
          font-family: "JetBrains Mono", monospace;
          font-size: clamp(26px, 4vw, 34px);
          line-height: 1.08;
          font-weight: 900;
          color: white;
          margin: 10px 0 6px 0;
          overflow-wrap: anywhere;
        }
        .small-badge {
          display: inline-block;
          color: var(--emerald-soft);
          background: rgba(16,185,129,.10);
          border: 1px solid rgba(16,185,129,.20);
          border-radius: 6px;
          padding: 3px 8px;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: .1em;
        }
        .metric-value {
          color: #2dd4bf;
          font-family: "JetBrains Mono", monospace;
          font-size: clamp(27px, 4vw, 34px);
          line-height: 1.1;
          font-weight: 900;
        }
        .severity-value {
          font-family: "JetBrains Mono", monospace;
          font-size: clamp(31px, 5vw, 42px);
          line-height: 1.1;
          font-weight: 900;
        }
        .bar {
          width: 100%;
          height: 7px;
          margin-top: 10px;
          background: #27272a;
          border: 1px solid #18181b;
          border-radius: 999px;
          overflow: hidden;
        }
        .bar-fill { height: 100%; border-radius: 999px; }
        .heatmap {
          position: relative;
          aspect-ratio: 1 / 1;
          min-height: 260px;
          background: rgba(0,0,0,.62);
          border: 1px solid rgba(6,78,59,.75);
          border-radius: 14px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .heatmap img {
          width: 100%;
          height: 100%;
          object-fit: contain;
          opacity: 1;
          filter: saturate(1.15);
          background: #020605;
        }
        .hotspot {
          position: absolute;
          transform: translate(-50%, -50%);
          border-radius: 50%;
          border: 1px dashed rgba(248,113,113,.48);
          animation: pulseHotspot 1.8s infinite;
        }
        @keyframes pulseHotspot {
          0%, 100% { opacity: .68; transform: translate(-50%, -50%) scale(.96); }
          50% { opacity: 1; transform: translate(-50%, -50%) scale(1.04); }
        }
        .map-caption {
          position: absolute;
          bottom: 12px;
          left: 12px;
          right: 12px;
          background: rgba(0,0,0,.72);
          border: 1px solid rgba(16,185,129,.25);
          color: #34d399;
          border-radius: 8px;
          padding: 6px 8px;
          font-family: "JetBrains Mono", monospace;
          font-size: 9px;
          font-weight: 800;
        }
        .warning-note {
          border: 1px solid rgba(251,146,60,.35);
          background: rgba(124,45,18,.20);
          color: #fdba74;
          border-radius: 8px;
          padding: 8px 10px;
          margin-top: 10px;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px;
          line-height: 1.45;
        }
        .validation-card {
          border: 1px solid rgba(16,185,129,.28);
          background: rgba(6,78,59,.22);
          border-radius: 12px;
          padding: 14px;
          margin: 10px 0 14px;
          color: #d1fae5;
          font-family: "JetBrains Mono", monospace;
          font-size: 12px;
          line-height: 1.55;
        }
        .validation-card.invalid {
          border-color: rgba(248,113,113,.35);
          background: rgba(127,29,29,.22);
          color: #fee2e2;
        }
        .validation-title {
          color: #34d399;
          font-size: 13px;
          font-weight: 900;
          margin-bottom: 8px;
        }
        .validation-card.invalid .validation-title { color: #fca5a5; }
        .validation-card ul {
          margin: 4px 0 0;
          padding-left: 18px;
        }
        .camera-capture-card {
          position: relative;
          overflow: hidden;
          border: 1px solid rgba(45,212,191,.30);
          border-radius: 14px;
          background:
            radial-gradient(circle at 12% 20%, rgba(45,212,191,.16), transparent 28%),
            rgba(6, 22, 18, .92);
          padding: 14px;
          margin: 12px 0 10px;
          font-family: "JetBrains Mono", monospace;
          box-shadow: 0 14px 32px rgba(0,0,0,.25);
        }
        .camera-capture-card:before {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(120deg, transparent 0%, rgba(52,211,153,.14) 42%, transparent 70%);
          transform: translateX(-110%);
          animation: cameraSweep 3.8s ease-in-out infinite;
          pointer-events: none;
        }
        .camera-capture-content {
          position: relative;
          z-index: 1;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .camera-lens {
          width: 42px;
          height: 42px;
          flex: 0 0 auto;
          border-radius: 14px;
          border: 1px solid rgba(45,212,191,.38);
          background: rgba(16,185,129,.14);
          display: grid;
          place-items: center;
          box-shadow: 0 0 0 rgba(45,212,191,.2);
          animation: lensPulse 2.2s ease-in-out infinite;
        }
        .camera-lens:before {
          content: "";
          width: 16px;
          height: 16px;
          border-radius: 50%;
          border: 3px solid #5eead4;
          box-shadow: inset 0 0 12px rgba(94,234,212,.35);
        }
        .camera-title {
          color: #f8fafc;
          font-size: 13px;
          font-weight: 900;
          letter-spacing: .08em;
          text-transform: uppercase;
        }
        .camera-copy {
          color: #94a3b8;
          font-family: "Space Grotesk", "Inter", sans-serif;
          font-size: 12px;
          line-height: 1.45;
          margin-top: 4px;
        }
        [data-testid="stCameraInput"] {
          border: 1px solid rgba(45,212,191,.22);
          border-radius: 14px;
          background: rgba(4,13,11,.75);
          padding: 12px;
        }
        @keyframes cameraSweep {
          0%, 52% { transform: translateX(-110%); }
          78%, 100% { transform: translateX(110%); }
        }
        @keyframes lensPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(45,212,191,.16); }
          50% { box-shadow: 0 0 0 10px rgba(45,212,191,0); }
        }
        ul.tiny-list, ol.tiny-list {
          color: #cbd5e1;
          font-size: 12px;
          line-height: 1.58;
          margin: 0;
          padding-left: 18px;
        }
        .treatment-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 22px;
        }
        .treatment-label {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          border-radius: 8px;
          border: 1px solid rgba(16,185,129,.25);
          background: rgba(6,78,59,.42);
          color: #34d399;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px;
          font-weight: 900;
          text-transform: uppercase;
          padding: 5px 8px;
          margin-bottom: 10px;
        }
        .pipeline {
          min-height: 500px;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: auto;
          background: rgba(0,0,0,.62);
          border: 1px solid #111827;
          border-radius: 18px;
          padding: 24px;
        }
        .pipeline-box {
          width: 850px;
          aspect-ratio: 16 / 10;
          background: #020605;
          border: 1px solid rgba(6,78,59,.8);
          border-radius: 14px;
          padding: 32px;
          position: relative;
          box-shadow: 0 28px 80px rgba(0,0,0,.52);
          transform-origin: center;
        }
        .pipeline-grid {
          height: 100%;
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 16px;
          align-items: center;
        }
        .stage-card {
          background: #0A1F1B;
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 16px;
          min-height: 235px;
          font-family: "JetBrains Mono", monospace;
        }
        .stage-chip {
          display: inline-block;
          color: #34d399;
          background: rgba(16,185,129,.10);
          border: 1px solid rgba(16,185,129,.28);
          border-radius: 6px;
          padding: 3px 6px;
          font-size: 9px;
          font-weight: 900;
          margin-bottom: 12px;
        }
        .stage-title {
          display: block;
          color: #f8fafc;
          font-size: 12px;
          font-weight: 900;
          margin-bottom: 16px;
        }
        .mini-line {
          background: rgba(6,78,59,.46);
          border: 1px solid rgba(6,78,59,.55);
          border-radius: 6px;
          color: #cbd5e1;
          font-size: 9px;
          padding: 5px;
          margin-bottom: 6px;
        }
        .footer-strip {
          color: var(--emerald-soft);
          border-top: 1px solid var(--border);
          padding: 12px 4px 0;
          font-family: "JetBrains Mono", monospace;
          font-size: 10px;
          font-weight: 800;
          display: flex;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
        }
        @media (max-width: 1100px) {
          .result-header { grid-template-columns: 1fr 1fr; }
          .result-header > div:first-child { grid-column: 1 / -1; }
        }
        @media (max-width: 900px) {
          .topbar { height: auto; align-items: flex-start; flex-direction: column; padding: 14px; }
          .status-row { justify-content: flex-start; }
          .hero { padding: 24px; }
          .result-header { grid-template-columns: 1fr; }
          .treatment-grid { grid-template-columns: 1fr; }
          .pipeline-box { width: 760px; }
        }
        @media (max-width: 680px) {
          .block-container {
            padding-left: .75rem;
            padding-right: .75rem;
            padding-bottom: 1.5rem;
          }
          .topbar { gap: 12px; }
          .brand { align-items: flex-start; }
          .brand-name {
            flex-wrap: wrap;
            font-size: 16px;
            line-height: 1.25;
          }
          .subtitle { font-size: 9px; line-height: 1.4; }
          .status-row, .status-pill { width: 100%; }
          .status-pill { justify-content: center; }
          .hero { padding: 18px; border-radius: 18px; }
          .hero-title { font-size: 32px; }
          .hero-subtitle { font-size: 16px; }
          .glass-panel, .result-header { border-radius: 14px; }
          .diagnosis-card, .result-header, .metric-card, .feature-card, .dark-card {
            padding: 14px;
          }
          .diagnosis-card-head {
            flex-direction: column;
            align-items: flex-start;
          }
          .section-label {
            font-size: 10px;
            letter-spacing: .08em;
            line-height: 1.45;
          }
          [data-testid="stFileUploader"] { padding: 12px; }
          [data-testid="stFileUploader"] section { padding: 0; }
          [data-testid="stFileUploader"] button,
          .stButton button,
          .stDownloadButton button {
            width: 100%;
            min-height: 44px;
          }
          .specimen-frame, .heatmap { min-height: 220px; }
          .meta-grid { grid-template-columns: 1fr; }
          .map-caption { line-height: 1.4; }
          .treatment-grid { gap: 14px; }
          ul.tiny-list, ol.tiny-list {
            font-size: 12px;
            line-height: 1.65;
            padding-left: 16px;
          }
        }
        @media print {
          [data-testid="stSidebar"], .topbar, .stButton, .stDownloadButton, [data-testid="stFileUploader"] {
            display: none !important;
          }
          html, body, .stApp, [data-testid="stAppViewContainer"] {
            background: white !important;
            color: black !important;
          }
          .glass-panel, .result-header, .metric-card, .dark-card {
            background: white !important;
            color: black !important;
            border-color: #444 !important;
            box-shadow: none !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def to_data_uri(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def html_list(items: list[str], ordered: bool = False) -> str:
    tag = "ol" if ordered else "ul"
    lines = "".join(f"<li>{html.escape(str(item))}</li>" for item in items)
    return f'<{tag} class="tiny-list">{lines}</{tag}>'


def _pdf_safe(text: object) -> str:
    raw = str(text)
    raw = "".join(ch if 32 <= ord(ch) <= 126 else "-" for ch in raw)
    return raw.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_color(color: tuple[int, int, int]) -> str:
    return " ".join(f"{channel / 255:.3f}" for channel in color)


def _pdf_text(
    text: object,
    x: float,
    y: float,
    size: float = 10,
    font: str = "F1",
    color: tuple[int, int, int] = (226, 232, 240),
) -> str:
    return (
        f"BT /{font} {size:.1f} Tf {_pdf_color(color)} rg "
        f"{x:.1f} {y:.1f} Td ({_pdf_safe(text)}) Tj ET"
    )


def _pdf_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: tuple[int, int, int],
    stroke: tuple[int, int, int] | None = None,
) -> str:
    if stroke:
        return (
            f"q {_pdf_color(fill)} rg {_pdf_color(stroke)} RG "
            f"{x:.1f} {y:.1f} {width:.1f} {height:.1f} re B Q"
        )
    return f"q {_pdf_color(fill)} rg {x:.1f} {y:.1f} {width:.1f} {height:.1f} re f Q"


def _pdf_stream(commands: list[str]) -> bytes:
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    return b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"


def _prepare_pdf_image(image_bytes: bytes | None) -> dict | None:
    if not image_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return {"bytes": buffer.getvalue(), "width": image.width, "height": image.height}
    except Exception:
        return None


def _pdf_image_object(image_info: dict) -> bytes:
    image_bytes = image_info["bytes"]
    header = (
        f"<< /Type /XObject /Subtype /Image /Width {image_info['width']} "
        f"/Height {image_info['height']} /ColorSpace /DeviceRGB "
        f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(image_bytes)} >>\n"
    ).encode("ascii")
    return header + b"stream\n" + image_bytes + b"\nendstream"


def _pdf_draw_image(name: str, image_info: dict, x: float, y: float, width: float, height: float) -> str:
    scale = min(width / image_info["width"], height / image_info["height"])
    draw_width = image_info["width"] * scale
    draw_height = image_info["height"] * scale
    draw_x = x + (width - draw_width) / 2
    draw_y = y + (height - draw_height) / 2
    return f"q {draw_width:.1f} 0 0 {draw_height:.1f} {draw_x:.1f} {draw_y:.1f} cm /{name} Do Q"


def _pdf_report_header(commands: list[str], page_label: str) -> None:
    commands.extend(
        [
            _pdf_rect(0, 0, 612, 842, (255, 255, 255)),
            _pdf_rect(0, 760, 612, 82, (236, 253, 245)),
            _pdf_text("ALeaf-STENet Diagnosis Report", 38, 812, 18, "F2", (6, 78, 59)),
            _pdf_text(RESEARCH_TITLE, 38, 792, 8.4, "F1", (21, 128, 61)),
            _pdf_text(page_label, 512, 812, 9, "F2", (13, 148, 136)),
            _pdf_text(FOOTER_TEXT, 38, 24, 8.4, "F1", (45, 160, 110)),
        ]
    )


def _pdf_placeholder(commands: list[str], x: float, y: float, width: float, height: float, label: str) -> None:
    commands.extend(
        [
            _pdf_rect(x, y, width, height, (248, 250, 252), (203, 213, 225)),
            _pdf_text(label, x + 22, y + (height / 2), 10, "F2", (100, 116, 139)),
        ]
    )


def _pdf_detail_lines(diagnosis: dict, specimen_name: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = [
        ("Diagnosis Output", "section"),
        (f"System Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "body"),
        (f"Specimen Field Key: {specimen_name or 'DEMO_LEAF_SCAB_Specimen'}", "body"),
        (f"Detected Pathogen Class: {diagnosis['diseaseName']}", "body"),
        (f"Scientific Classification: {diagnosis['scientificName']}", "body"),
        (f"STENet Match Confidence: {diagnosis['confidence']}%", "body"),
        (f"Tissue Infection Ratio: {diagnosis['severity']}% ({diagnosis['severityLevel']})", "body"),
        (f"Device: {diagnosis.get('device', 'unknown')}", "body"),
        ("", "blank"),
        ("Top Class Probabilities", "section"),
    ]
    lines.extend(
        (f"- {pred['class_name']}: {pred['confidence']}%", "body")
        for pred in diagnosis.get("top_predictions", [])
    )
    lines.extend(
        [
            ("", "blank"),
            ("Grad-CAM++ Explanation", "section"),
            (diagnosis["gradCamMetadata"]["description"], "body"),
            (f"Hotspot Nodes: {len(diagnosis['gradCamMetadata']['hotspots'])}", "body"),
            ("", "blank"),
            ("Symptom Attribution Overview", "section"),
        ]
    )
    lines.extend((f"- {item}", "body") for item in diagnosis["symptoms"])
    lines.extend([("", "blank"), ("Treatment Plan: Preventive Care", "section")])
    lines.extend((f"- {item}", "body") for item in diagnosis["treatment"]["preventive"])
    lines.append(("", "blank"))
    lines.append(("Treatment Plan: Organic Control", "section"))
    lines.extend((f"- {item}", "body") for item in diagnosis["treatment"]["organic"])
    lines.append(("", "blank"))
    lines.append(("Treatment Plan: Chemical Control", "section"))
    lines.extend((f"- {item}", "body") for item in diagnosis["treatment"]["chemical"])
    lines.extend([("", "blank"), ("Research Notes", "section"), (diagnosis["researchNotes"], "body")])
    return lines


def _pdf_wrapped_lines(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    wrapped: list[tuple[str, str]] = []
    for text, kind in items:
        if kind == "blank":
            wrapped.append(("", "blank"))
            continue
        width = 74 if kind == "section" else 88
        chunks = textwrap.wrap(str(text), width=width) or [""]
        for index, chunk in enumerate(chunks):
            wrapped.append((chunk if index == 0 else f"  {chunk}", kind if index == 0 else "body"))
    return wrapped


def _pdf_text_page(lines: list[tuple[str, str]], page_number: int) -> bytes:
    commands: list[str] = []
    _pdf_report_header(commands, f"Page {page_number}")
    y = 724.0
    for text, kind in lines:
        if kind == "blank":
            y -= 9
            continue
        if kind == "section":
            commands.append(_pdf_text(text, 42, y, 11, "F2", (52, 211, 153)))
            y -= 17
        else:
            commands.append(_pdf_text(text, 48, y, 9.2, "F1", (51, 65, 85)))
            y -= 13.4
    return _pdf_stream(commands)


def build_pdf_report(
    diagnosis: dict,
    specimen_name: str,
    specimen_bytes: bytes | None = None,
    gradcam_bytes: bytes | None = None,
) -> bytes:
    """Create a color PDF report with specimen, Grad-CAM++, and diagnosis output."""

    specimen_image = _prepare_pdf_image(specimen_bytes)
    gradcam_image = _prepare_pdf_image(gradcam_bytes)

    objects: list[bytes] = []

    def add_object(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog_ref = add_object(b"")
    pages_ref = add_object(b"")
    font_ref = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    bold_ref = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    image_refs: list[tuple[str, int, dict]] = []
    if specimen_image:
        image_refs.append(("Im1", add_object(_pdf_image_object(specimen_image)), specimen_image))
    if gradcam_image:
        image_refs.append(("Im2", add_object(_pdf_image_object(gradcam_image)), gradcam_image))

    def resources() -> str:
        xobjects = ""
        if image_refs:
            entries = " ".join(f"/{name} {ref} 0 R" for name, ref, _info in image_refs)
            xobjects = f" /XObject << {entries} >>"
        return f"<< /Font << /F1 {font_ref} 0 R /F2 {bold_ref} 0 R >>{xobjects} >>"

    page_refs: list[int] = []
    commands: list[str] = []
    _pdf_report_header(commands, "Page 1")
    commands.extend(
        [
            _pdf_text("Color Input Image", 42, 730, 11, "F2", (52, 211, 153)),
            _pdf_text("Grad-CAM++ Attention Map", 324, 730, 11, "F2", (45, 212, 191)),
            _pdf_rect(38, 474, 254, 242, (248, 250, 252), (203, 213, 225)),
            _pdf_rect(320, 474, 254, 242, (248, 250, 252), (203, 213, 225)),
        ]
    )
    if specimen_image:
        commands.append(_pdf_draw_image("Im1", specimen_image, 48, 486, 234, 218))
    else:
        _pdf_placeholder(commands, 48, 486, 234, 218, "No input image available")
    if gradcam_image:
        commands.append(_pdf_draw_image("Im2", gradcam_image, 330, 486, 234, 218))
    else:
        _pdf_placeholder(commands, 330, 486, 234, 218, "No Grad-CAM++ image available")

    severity = float(diagnosis["severity"])
    severity_width = max(0, min(170, severity * 1.7))
    commands.extend(
        [
            _pdf_rect(38, 348, 536, 104, (240, 253, 250), (153, 246, 228)),
            _pdf_text("Detected Pathogen Class", 54, 424, 8.8, "F1", (148, 163, 184)),
            _pdf_text(diagnosis["diseaseName"], 54, 404, 16, "F2", (15, 23, 42)),
            _pdf_text(diagnosis["scientificName"], 54, 386, 9.5, "F1", (21, 128, 61)),
            _pdf_text(f"Confidence: {diagnosis['confidence']}%", 362, 420, 12, "F2", (45, 212, 191)),
            _pdf_text(f"Severity: {severity}% ({diagnosis['severityLevel']})", 362, 394, 12, "F2", (251, 146, 60)),
            _pdf_rect(362, 374, 170, 8, (226, 232, 240)),
            _pdf_rect(362, 374, severity_width, 8, (251, 146, 60)),
            _pdf_text(f"Specimen: {specimen_name or 'DEMO_LEAF_SCAB_Specimen'}", 54, 362, 8.6, "F1", (51, 65, 85)),
        ]
    )

    y = 314
    commands.append(_pdf_text("Top Class Probabilities", 42, y, 11, "F2", (52, 211, 153)))
    for pred in diagnosis.get("top_predictions", []):
        y -= 18
        commands.append(_pdf_text(f"{pred['class_name']}: {pred['confidence']}%", 54, y, 9.5, "F1", (51, 65, 85)))

    y -= 28
    commands.append(_pdf_text("Grad-CAM++ Explanation", 42, y, 11, "F2", (45, 212, 191)))
    for line in textwrap.wrap(diagnosis["gradCamMetadata"]["description"], width=92):
        y -= 14
        commands.append(_pdf_text(line, 54, y, 8.8, "F1", (51, 65, 85)))

    first_content_ref = add_object(_pdf_stream(commands))
    page_refs.append(
        add_object(
            (
                f"<< /Type /Page /Parent {pages_ref} 0 R /MediaBox [0 0 612 842] "
                f"/Resources {resources()} /Contents {first_content_ref} 0 R >>"
            ).encode("ascii")
        )
    )

    all_lines = _pdf_wrapped_lines(_pdf_detail_lines(diagnosis, specimen_name))
    lines_per_page = 47
    for offset in range(0, len(all_lines), lines_per_page):
        page_number = len(page_refs) + 1
        content_ref = add_object(_pdf_text_page(all_lines[offset : offset + lines_per_page], page_number))
        page_refs.append(
            add_object(
                (
                    f"<< /Type /Page /Parent {pages_ref} 0 R /MediaBox [0 0 612 842] "
                    f"/Resources {resources()} /Contents {content_ref} 0 R >>"
                ).encode("ascii")
            )
        )

    objects[catalog_ref - 1] = f"<< /Type /Catalog /Pages {pages_ref} 0 R >>".encode("ascii")
    kids = " ".join(f"{page_ref} 0 R" for page_ref in page_refs)
    objects[pages_ref - 1] = f"<< /Type /Pages /Kids [ {kids} ] /Count {len(page_refs)} >>".encode("ascii")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(pdf)


def estimate_hotspots(image: Image.Image, severity: float, max_points: int = 3) -> list[dict]:
    if severity <= 3:
        return []

    sample = image.convert("RGB").resize((160, 160))
    arr = np.asarray(sample).astype("float32") / 255.0
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    lesion_score = (red * 0.62 + blue * 0.34) - (green * 0.42)
    yellow_score = np.abs(red - green) * 0.30
    gray_score = (1.0 - np.std(arr, axis=2)) * 0.12
    score = lesion_score + yellow_score + gray_score
    score = score - score.min()
    if score.max() > 0:
        score = score / score.max()

    blocks = []
    block = 20
    for y in range(0, 160, block):
        for x in range(0, 160, block):
            patch = score[y : y + block, x : x + block]
            blocks.append((float(patch.mean()), x + block / 2, y + block / 2))

    blocks.sort(reverse=True, key=lambda item: item[0])
    hotspots = []
    for value, x, y in blocks[:max_points]:
        if value <= 0.08:
            continue
        hotspots.append(
            {
                "x": round((x / 160) * 100, 1),
                "y": round((y / 160) * 100, 1),
                "radius": round(min(28, max(12, severity / 3.2)), 1),
                "intensity": round(min(0.98, max(0.45, value)), 2),
            }
        )
    return hotspots


def connected_components(mask: np.ndarray, max_components: int = 8) -> list[dict]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            area = 0
            min_x = max_x = start_x
            min_y = max_y = start_y

            while stack:
                y, x = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and not visited[next_y, next_x]
                        and mask[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

            components.append(
                {
                    "area": area,
                    "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                }
            )
            components.sort(key=lambda item: item["area"], reverse=True)
            components = components[:max_components]

    return components


def validate_apple_leaf_image(image: Image.Image) -> bool:
    width, height = image.size
    if min(width, height) < 96 or width * height < 18000:
        return False

    sample = image.convert("RGB").resize((160, 160))
    arr = np.asarray(sample).astype("float32") / 255.0
    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    dynamic_range = float(gray.max() - gray.min())

    if brightness < 0.08 or brightness > 0.96:
        return False
    if contrast < 0.025 or dynamic_range < 0.09:
        return False

    gradient_y, gradient_x = np.gradient(gray)
    sharpness = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))
    if sharpness < 0.0025:
        return False

    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 0.001)

    green_mask = (
        (green > red * 0.92)
        & (green > blue * 0.95)
        & (green > 0.10)
        & (saturation > 0.055)
    )
    yellow_green_mask = (
        (green > 0.13)
        & (red > 0.12)
        & (np.abs(red - green) < 0.42)
        & (green > blue * 0.86)
        & (saturation > 0.045)
        & (gray < 0.82)
    )
    brown_lesion_mask = (
        (red > 0.16)
        & (green > 0.08)
        & (blue < green * 0.92)
        & (red > blue * 1.18)
        & (saturation > 0.10)
        & (gray < 0.80)
    )
    vegetation_mask = green_mask | yellow_green_mask | brown_lesion_mask
    green_ratio = float(green_mask.mean())
    vegetation_ratio = float(vegetation_mask.mean())

    if green_ratio < 0.006 and vegetation_ratio < 0.035:
        return False
    if vegetation_ratio < 0.025:
        return False

    component_mask = Image.fromarray(vegetation_mask.astype("uint8") * 255).resize((96, 96))
    component_array = np.asarray(component_mask) > 0
    components = connected_components(component_array)
    if not components:
        return False

    image_area = component_array.size
    vegetation_area = int(component_array.sum())
    largest = components[0]
    largest_area = int(largest["area"])
    largest_ratio = largest_area / image_area
    dominant_ratio = largest_area / max(vegetation_area, 1)
    significant_components = sum(1 for item in components if item["area"] / image_area > 0.012)

    min_x, min_y, max_x, max_y = largest["bbox"]
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    bbox_area = max(bbox_width * bbox_height, 1)
    bbox_ratio = bbox_area / image_area
    fill_ratio = largest_area / bbox_area
    aspect_ratio = bbox_width / max(bbox_height, 1)

    if largest_ratio < 0.018:
        return False
    if dominant_ratio < 0.24 and vegetation_ratio < 0.24:
        return False
    if significant_components > 12 and largest_ratio < 0.10:
        return False
    if not (0.10 <= aspect_ratio <= 10.0):
        return False
    if bbox_ratio < 0.018:
        return False
    if fill_ratio < 0.045:
        return False

    red_orange_ratio = float(((red > 0.50) & (red > green * 1.22) & (red > blue * 1.25)).mean())
    white_flat_ratio = float(((gray > 0.82) & (saturation < 0.12)).mean())
    if red_orange_ratio > 0.60 and green_ratio < 0.08:
        return False
    if white_flat_ratio > 0.86 and vegetation_ratio < 0.25:
        return False

    leaf_score = 0
    leaf_score += vegetation_ratio >= 0.025
    leaf_score += vegetation_ratio >= 0.055
    leaf_score += green_ratio >= 0.006
    leaf_score += largest_ratio >= 0.018
    leaf_score += largest_ratio >= 0.045
    leaf_score += dominant_ratio >= 0.24 or vegetation_ratio >= 0.45
    leaf_score += significant_components <= 12
    leaf_score += 0.10 <= aspect_ratio <= 10.0
    leaf_score += bbox_ratio >= 0.018
    leaf_score += fill_ratio >= 0.045
    leaf_score += contrast >= 0.025
    leaf_score += sharpness >= 0.0025

    return leaf_score >= 7


def build_diagnosis(prediction: dict, image: Image.Image) -> dict:
    class_name = prediction["class_name"]
    profile = get_profile(class_name)
    severity = float(prediction["severity"])
    hotspots = prediction.get("gradcam_hotspots") or estimate_hotspots(image, severity)
    disease_name = "Healthy Apple Foliage" if class_name.lower() == "health" else class_name

    return {
        "class_name": class_name,
        "diseaseName": disease_name,
        "scientificName": profile["scientific_name"],
        "confidence": prediction["confidence"],
        "severity": severity,
        "severityLevel": severity_level(severity),
        "symptoms": profile["symptoms"],
        "treatment": {
            "preventive": profile["preventive"],
            "organic": profile["organic"],
            "chemical": profile["chemical"],
        },
        "gradCamMetadata": {
            "description": (
                f"Hybrid ALeaf-STENet attention summary for {class_name}: "
                f"{len(hotspots)} high-density tissue focus node(s) were derived "
                "from the uploaded specimen for visual severity explanation."
            ),
            "hotspots": hotspots,
        },
        "researchNotes": profile["research_notes"],
        "top_predictions": prediction["top_predictions"],
        "device": prediction.get("device", "unknown"),
        "val_acc": prediction.get("val_acc"),
        "raw_severity": prediction.get("raw_severity"),
    }


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    return load_hybrid_model(model_path)


def run_inference(uploaded_file) -> None:
    payload = uploaded_file.getvalue()
    file_hash = hashlib.sha256(payload).hexdigest()
    if st.session_state.last_upload_hash == file_hash:
        return

    st.session_state.model_error = None
    st.session_state.selected_leaf_name = uploaded_file.name
    st.session_state.selected_leaf_bytes = payload
    st.session_state.selected_leaf_uri = to_data_uri(payload, uploaded_file.type or "image/png")
    st.session_state.gradcam_uri = None
    st.session_state.gradcam_bytes = None
    st.session_state.gradcam_error = None
    st.session_state.validation_passed = None
    st.session_state.validation_message = ""
    st.session_state.last_upload_hash = file_hash

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:  # noqa: BLE001
        st.session_state.validation_passed = False
        st.session_state.validation_message = "Apple Leaf Not Detected"
        return

    if not validate_apple_leaf_image(image):
        st.session_state.validation_passed = False
        st.session_state.validation_message = "Apple Leaf Not Detected"
        return

    st.session_state.validation_passed = True
    st.session_state.validation_message = "Apple Leaf Detected"
    try:
        with st.spinner("ALeaf-STENet Active Inference Parallel Pipeline running..."):
            bundle = cached_model(str(MODEL_PATH))
            prediction = predict_leaf(bundle, image)
            try:
                gradcam_image, gradcam_hotspots = generate_gradcam_plus_plus(
                    bundle,
                    image,
                    prediction.get("class_index"),
                )
                gradcam_buffer = io.BytesIO()
                gradcam_image.save(gradcam_buffer, format="PNG")
                st.session_state.gradcam_bytes = gradcam_buffer.getvalue()
                st.session_state.gradcam_uri = to_data_uri(st.session_state.gradcam_bytes, "image/png")
                prediction["gradcam_hotspots"] = gradcam_hotspots
            except Exception as gradcam_exc:  # noqa: BLE001
                st.session_state.gradcam_error = f"Grad-CAM++ generation failed: {gradcam_exc}"
            st.session_state.diagnosis = build_diagnosis(prediction, image)
    except MissingModelDependency as exc:
        st.session_state.model_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        st.session_state.model_error = f"Model inference failed: {exc}"


def process_arch_upload(uploaded_file) -> None:
    if uploaded_file is not None:
        st.session_state.arch_image_uri = to_data_uri(
            uploaded_file.getvalue(),
            uploaded_file.type or "image/png",
        )


def set_active_tab(tab_name: str) -> None:
    if tab_name != st.session_state.active_tab:
        st.session_state.previous_tab = st.session_state.active_tab
        st.session_state.active_tab = tab_name


def render_back_control() -> None:
    if st.session_state.active_tab == "Research":
        return

    target = st.session_state.previous_tab or "Research"
    if target == st.session_state.active_tab:
        target = "Research"

    back_col, _ = st.columns([1.15, 5], gap="small")
    with back_col:
        if st.button(f"Back to {target}", key=f"back_to_{target}", use_container_width=True):
            st.session_state.previous_tab = st.session_state.active_tab
            st.session_state.active_tab = target
            st.rerun()


def render_topbar() -> None:
    device = st.session_state.get("diagnosis", {}).get("device")
    model_state = "CUDA Multi-GPU Node" if device == "cuda" else "Streamlit Ready"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="logo-mark">{leaf_svg(26)}</div>
            <div>
              <div class="brand-name">ALeaf-STENet <span class="version">v2.4.1</span></div>
              <div class="subtitle">Hybrid Neural Network Diagnosis Center</div>
            </div>
          </div>
          <div class="status-row">
            <span class="status-pill"><span class="pulse-dot"></span> STENet Engine Live</span>
            <span class="status-pill">{html.escape(model_state)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 14px 4px 18px;">
              <div style="display:flex;align-items:center;gap:10px;color:#34d399;">
                <div class="logo-mark" style="width:38px;height:38px;">{leaf_svg(23)}</div>
                <div>
                  <div style="font-weight:900;color:white;font-size:16px;">ALeaf-STENet</div>
                  <div style="font-size:9px;color:#2da06e;letter-spacing:.15em;">STABILITY LIVE</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        options = ["Research", "Diagnosis", "Swin-T"]
        choice = st.radio(
            "Navigation",
            options,
            index=options.index(st.session_state.active_tab),
            label_visibility="collapsed",
        )
        set_active_tab(choice)
        st.markdown(
            """
            <div style="margin-top:28px;padding:12px;border:1px solid #1A3A32;border-radius:12px;background:#020605;">
              <div style="color:#64748b;font-size:9px;">MODEL CHECKPOINT</div>
              <div style="color:#34d399;font-size:10px;font-weight:900;word-break:break-all;">aleaf_best.pth</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_error() -> None:
    if not st.session_state.model_error:
        return
    st.error(st.session_state.model_error)
    st.info(
        "For deployment, keep aleaf_best.pth beside app.py and install packages "
        "from requirements.txt. On Streamlit Community Cloud, large model files should "
        "be committed with Git LFS or provided through ALEAF_MODEL_PATH."
    )


def render_validation_status() -> None:
    if st.session_state.validation_passed is None:
        return

    if st.session_state.validation_passed:
        st.markdown(
            """
            <div class="validation-card">
              <div class="validation-title">🟢 Apple Leaf Detected</div>
              Image Validation → Apple Leaf Detected → ALeaf-STENet Ready
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="validation-card invalid">
          <div class="validation-title">🔴 Apple Leaf Not Detected</div>
          <strong>❌ Invalid Image Detected</strong><br><br>
          Please upload a clear Apple Leaf image for disease diagnosis.<br><br>
          <strong>Accepted:</strong>
          <ul>
            <li>✓ Apple Leaf</li>
            <li>✓ Healthy Leaf</li>
            <li>✓ Diseased Leaf</li>
          </ul>
          <br>
          <strong>Not Accepted:</strong>
          <ul>
            <li>✗ Human</li>
            <li>✗ Animal</li>
            <li>✗ Car</li>
            <li>✗ Building</li>
            <li>✗ Document</li>
            <li>✗ Screenshot</li>
            <li>✗ Random Object</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_leaf_demo_visual() -> None:
    st.markdown(
        """
        <div class="glass-panel" style="padding:20px;position:relative;overflow:hidden;">
          <div class="specimen-frame" style="min-height:330px;">
            <svg viewBox="0 0 100 100" style="width:220px;height:220px;color:#10b981;animation:pulse 2s infinite;">
              <path d="M50,15 C65,30 85,45 50,85 C15,45 35,30 50,15 Z" fill="none" stroke="currentColor" stroke-width="1.6" opacity=".85"/>
              <path d="M50,15 C50,15 50,85 50,85" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="3,3"/>
              <line x1="10" y1="35" x2="90" y2="35" stroke="#10b981" stroke-width=".5" stroke-dasharray="2,2"/>
              <line x1="10" y1="55" x2="90" y2="55" stroke="#10b981" stroke-width=".5" stroke-dasharray="2,2"/>
              <circle cx="38" cy="44" r="7" fill="#dc2626" fill-opacity=".45" stroke="#f87171" stroke-width=".5"/>
              <circle cx="65" cy="55" r="5" fill="#ea580c" fill-opacity=".45" stroke="#fb923c" stroke-width=".5"/>
              <circle cx="50" cy="30" r="4" fill="#84cc16" fill-opacity=".45" stroke="#a3e635" stroke-width=".5"/>
              <rect x="15" y="47" width="70" height="2" fill="#10b981" opacity=".7"/>
            </svg>
            <div class="map-caption">Swin-T Localizer | 96.4% Acc</div>
          </div>
          <div style="margin-top:16px;display:grid;gap:8px;font-size:12px;color:#cbd5e1;">
            <div style="display:flex;justify-content:space-between;"><span>Classification:</span><strong style="color:#34d399;font-family:JetBrains Mono;">Apple Scab (Venturia)</strong></div>
            <div style="display:flex;justify-content:space-between;"><span>Confidence Level:</span><strong style="color:#2dd4bf;font-family:JetBrains Mono;">96.4% Accuracy</strong></div>
            <div style="display:flex;justify-content:space-between;"><span>Leaf Tissue Severity:</span><strong style="color:#fb7185;font-family:JetBrains Mono;">42.0% Infected Area</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero">
              <div class="eyebrow">IEEE Academic Research Project</div>
              <div class="hero-title">ALeaf-STENet</div>
              <div class="hero-subtitle">Hybrid Deep Learning Framework for Apple Leaf Disease Detection and Severity Assessment</div>
              <p class="body-copy">
                Integrating <strong style="color:#a7f3d0;">EfficientNetV2-S</strong> texture extractors,
                <strong style="color:#a7f3d0;"> CBAM Channel & Spatial Attention</strong> modules, and
                <strong style="color:#a7f3d0;"> Swin Transformer</strong> shift-window networks to assess apple foliage diseases
                and spatial infection densities. The Streamlit backend now runs the trained hybrid checkpoint directly.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        uploaded = st.file_uploader(
            "Diagnose Leaf Specimen",
            type=["png", "jpg", "jpeg", "webp"],
            key="home_leaf_uploader",
            help="Upload an apple leaf image for hybrid model classification and severity regression.",
        )
        st.markdown(
            """
            <div class="camera-capture-card">
              <div class="camera-capture-content">
                <div class="camera-lens"></div>
                <div>
                  <div class="camera-title">Camera Capture</div>
                  <div class="camera-copy">Take a fresh apple leaf photo from phone camera. Validation and diagnosis will run the same way as uploaded images.</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        captured = st.camera_input(
            "Capture Apple Leaf Photo",
            key="home_camera_input",
            help="Use your phone camera to capture a clear apple leaf photo for validation and diagnosis.",
        )
        if captured is not None:
            run_inference(captured)
            set_active_tab("Diagnosis")
            st.rerun()
        elif uploaded is not None:
            run_inference(uploaded)
            set_active_tab("Diagnosis")
            st.rerun()
    with right:
        render_leaf_demo_visual()

    st.write("")
    feature_data = [
        (
            "Hybrid DL Framework",
            "By combining localized Swin Transformer patch partitions with multi-scale deep CNN channels, ALeaf-STENet captures irregular lesion textures and global foliage context.",
        ),
        (
            "Disease Severity Head",
            "A parallel regression head maps infected tissue ratios, enabling direct leaf damage percentage assessment from the uploaded specimen.",
        ),
        (
            "Pathological Treatment",
            "The app returns disease-specific preventive, organic, and chemical guidance after model classification.",
        ),
    ]
    for col, (title, text) in zip(st.columns(3, gap="large"), feature_data):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                  <div class="feature-icon">{leaf_svg(25)}</div>
                  <h3 style="color:white;font-size:18px;margin:0 0 10px;">{html.escape(title)}</h3>
                  <p class="body-copy" style="font-size:12px;margin:0;">{html.escape(text)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    btn1, btn2, _ = st.columns([1, 1.2, 2])
    with btn1:
        if st.button("Load Live Demo Playground", use_container_width=True):
            set_active_tab("Diagnosis")
            st.rerun()
    with btn2:
        if st.button("View Model Architecture Framework", use_container_width=True):
            set_active_tab("Swin-T")
            st.rerun()


def specimen_panel() -> None:
    leaf_uri = st.session_state.selected_leaf_uri
    file_name = st.session_state.selected_leaf_name or "Malus_Dome_Scab_99"
    if leaf_uri:
        visual = f"""
        <div class="specimen-frame">
          <img src="{leaf_uri}" alt="Original Leaf Specimen"/>
          <div class="scan-line"></div>
        </div>
        """
    else:
        visual = """
        <div class="specimen-frame">
          <div class="placeholder">
            <div style="font-size:13px;font-weight:900;text-transform:uppercase;">No Specimen Uploaded</div>
            <div style="font-size:10px;margin-top:8px;">Please drag or load a leaf image to initialize visual feature segmentation maps.</div>
          </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="glass-panel" style="padding:20px;">
          <div class="section-label">Specimen Focus Target</div>
          {visual}
          <div style="height:12px;"></div>
          <div class="meta-grid">
            <div><span>SPECIMEN ID:</span><strong>{html.escape(file_name)}</strong></div>
            <div><span>ENGINE RES:</span><strong>224 x 224 px</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_color(value: float) -> str:
    if value > 60:
        return "#ef4444"
    if value > 30:
        return "#fb923c"
    return "#34d399"


def render_result_header(diagnosis: dict) -> None:
    severity = float(diagnosis["severity"])
    color = severity_color(severity)
    st.markdown(
        f"""
        <div class="result-header">
          <div>
            <span class="small-badge">Pathology Classifier Output</span>
            <div class="result-title">{html.escape(str(diagnosis["diseaseName"]))}</div>
            <div style="font-family:JetBrains Mono;color:#34d399;font-size:12px;font-style:italic;">
              {html.escape(str(diagnosis["scientificName"]))}
            </div>
          </div>
          <div class="metric-card" style="text-align:center;">
            <div style="font-family:JetBrains Mono;color:#94a3b8;font-size:10px;text-transform:uppercase;">Confidence</div>
            <div class="metric-value">{diagnosis["confidence"]}%</div>
            <div style="font-family:JetBrains Mono;color:#2da06e;font-size:9px;font-weight:900;text-transform:uppercase;">
              Active STENet Class Match
            </div>
          </div>
          <div class="metric-card" style="text-align:center;">
            <div style="font-family:JetBrains Mono;color:#94a3b8;font-size:10px;text-transform:uppercase;">Tissue Damage Area</div>
            <div class="severity-value" style="color:{color};">{severity}%</div>
            <div class="bar"><div class="bar-fill" style="width:{severity}%;background:{color};"></div></div>
            <div style="font-family:JetBrains Mono;color:#94a3b8;font-size:9px;font-weight:900;text-transform:uppercase;margin-top:5px;">
              {html.escape(str(diagnosis["severityLevel"]))} State
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_heatmap(diagnosis: dict) -> None:
    if st.session_state.gradcam_uri:
        visual = f'<img src="{st.session_state.gradcam_uri}" alt="Grad-CAM++ attention map"/>'
        caption = f"Attention Map Focus High-Density hotspots: {len(diagnosis['gradCamMetadata']['hotspots'])} Nodes"
        status = "Auto Map"
    elif st.session_state.selected_leaf_uri:
        visual = f'<img src="{st.session_state.selected_leaf_uri}" alt="Uploaded leaf specimen"/>'
        caption = st.session_state.gradcam_error or "Original uploaded image shown while Grad-CAM++ is unavailable."
        status = "Original"
    else:
        visual = '<div class="placeholder">Awaiting Uploaded Specimen</div>'
        caption = "Awaiting uploaded specimen"
        status = "Waiting"

    warning = ""
    if st.session_state.gradcam_error:
        warning = f'<div class="warning-note">{html.escape(str(st.session_state.gradcam_error))}</div>'

    st.markdown(
        f"""
        <div class="glass-panel diagnosis-card">
          <div class="diagnosis-card-head">
            <div>
              <div class="section-label no-margin">Grad-CAM++ Explainability Map</div>
              <div class="card-subtext">Generated from the same uploaded leaf image.</div>
            </div>
            <span class="status-badge">{html.escape(status)}</span>
          </div>
          <div class="heatmap">
            {visual}
            <div class="map-caption">{html.escape(caption)}</div>
          </div>
          {warning}
          <div class="dark-card" style="font-family:JetBrains Mono;font-size:10px;color:#94a3b8;line-height:1.55;margin-top:12px;">
            <strong style="color:#a7f3d0;">Target Attributions:</strong>
            {html.escape(str(diagnosis["gradCamMetadata"]["description"]))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_predictions(diagnosis: dict) -> None:
    rows = ""
    for pred in diagnosis.get("top_predictions", []):
        rows += f"""
        <div style="display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid rgba(26,58,50,.55);padding:7px 0;">
          <span>{html.escape(str(pred["class_name"]))}</span>
          <strong style="color:#34d399;">{pred["confidence"]}%</strong>
        </div>
        """
    if not rows:
        rows = '<div style="color:#64748b;">No probability table available.</div>'
    st.markdown(
        f"""
        <div class="dark-card" style="font-family:JetBrains Mono;font-size:11px;color:#cbd5e1;margin-top:12px;">
          <div style="color:#2da06e;font-size:10px;text-transform:uppercase;font-weight:900;margin-bottom:6px;">Top Class Probabilities</div>
          {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnosis() -> None:
    uploaded = st.file_uploader(
        "Foliage Specimen Upload",
        type=["png", "jpg", "jpeg", "webp"],
        key="diagnosis_leaf_uploader",
        help="Supports PNG, JPG, JPEG, and WEBP. The trained hybrid model runs after upload.",
    )

    st.markdown(
        """
        <div class="camera-capture-card">
          <div class="camera-capture-content">
            <div class="camera-lens"></div>
            <div>
              <div class="camera-title">Camera Capture</div>
              <div class="camera-copy">Take a fresh apple leaf photo from phone camera. Validation and diagnosis will run the same way as uploaded images.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    captured = st.camera_input(
        "Capture Apple Leaf Photo",
        key="diagnosis_camera_input",
        help="Use your phone camera to capture a clear apple leaf photo for validation and diagnosis.",
    )
    if captured is not None:
        run_inference(captured)
    elif uploaded is not None:
        run_inference(uploaded)

    render_validation_status()
    render_model_error()
    if st.session_state.validation_passed is False:
        left, _right = st.columns([0.95, 2.05], gap="large")
        with left:
            specimen_panel()
        return

    diagnosis = st.session_state.diagnosis
    left, right = st.columns([0.95, 2.05], gap="large")

    with left:
        specimen_panel()

    with right:
        render_result_header(diagnosis)
        st.write("")
        h1, h2 = st.columns(2, gap="large")
        with h1:
            render_heatmap(diagnosis)
            render_top_predictions(diagnosis)

        with h2:
            st.markdown(
                f"""
                <div class="glass-panel diagnosis-card">
                  <div class="diagnosis-card-head">
                    <div>
                      <div class="section-label no-margin">Diagnosis Details</div>
                      <div class="card-subtext">Symptoms and notes matched to the predicted class.</div>
                    </div>
                    <span class="status-badge">{html.escape(str(diagnosis["severityLevel"]))}</span>
                  </div>
                  <div class="dark-card">
                    <div style="color:#34d399;font-family:JetBrains Mono;font-size:10px;font-weight:900;text-transform:uppercase;border-bottom:1px solid rgba(6,78,59,.55);padding-bottom:6px;margin-bottom:8px;">Visible Symptoms</div>
                    {html_list(diagnosis["symptoms"])}
                  </div>
                  <div style="height:14px;"></div>
                  <div class="dark-card">
                    <div style="color:#34d399;font-family:JetBrains Mono;font-size:10px;font-weight:900;text-transform:uppercase;border-bottom:1px solid rgba(6,78,59,.55);padding-bottom:6px;margin-bottom:8px;">Research Notes</div>
                    <p style="color:#cbd5e1;font-size:12px;line-height:1.6;margin:0;">{html.escape(str(diagnosis["researchNotes"]))}</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown(
            f"""
            <div class="glass-panel diagnosis-card">
              <div class="section-label" style="border-bottom:1px solid #1A3A32;padding-bottom:10px;">Treatment Plan</div>
              <div class="treatment-grid">
                <div>
                  <div class="treatment-label">Preventive Care</div>
                  {html_list(diagnosis["treatment"]["preventive"], ordered=True)}
                </div>
                <div>
                  <div class="treatment-label" style="color:#2dd4bf;border-color:rgba(45,212,191,.25);background:rgba(20,184,166,.12);">Organic Control</div>
                  {html_list(diagnosis["treatment"]["organic"], ordered=True)}
                </div>
                <div>
                  <div class="treatment-label" style="color:#fb7185;border-color:rgba(251,113,133,.25);background:rgba(159,18,57,.22);">Chemical Control</div>
                  {html_list(diagnosis["treatment"]["chemical"], ordered=True)}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        report_col, print_col, json_col = st.columns([2, 1, 1], gap="medium")
        with report_col:
            st.markdown(
                """
                <div class="glass-panel diagnosis-card">
                  <div style="font-family:JetBrains Mono;color:#34d399;font-weight:900;">Report Export</div>
                  <p class="body-copy" style="font-size:12px;margin:6px 0 0;">
                    Current diagnosis summary with class, confidence, severity, attention map, and treatment plan.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with print_col:
            st.download_button(
                "Generate PDF Report",
                data=build_pdf_report(
                    diagnosis,
                    st.session_state.selected_leaf_name,
                    st.session_state.selected_leaf_bytes,
                    st.session_state.gradcam_bytes,
                ),
                file_name=f"aleaf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with json_col:
            st.download_button(
                "Download JSON",
                data=json.dumps(diagnosis, indent=2),
                file_name=f"aleaf_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )


def render_architecture() -> None:
    st.markdown(
        """
        <div class="glass-panel" style="padding:24px;background:linear-gradient(90deg,#0A1F1B,#040D0B);">
          <div style="display:flex;justify-content:space-between;gap:22px;align-items:center;flex-wrap:wrap;">
            <div>
              <h2 style="margin:0;color:white;font-size:21px;">ALeaf-STENet Hybrid Deep Feature Pipeline</h2>
              <p class="body-copy" style="font-size:12px;max-width:760px;margin:8px 0 0;">
                High-resolution publication-grade model architecture flow mapping:
                <strong style="color:#34d399;"> EfficientNetV2-S</strong> primary spatial texture layers,
                <strong style="color:#34d399;"> CBAM Modules</strong> channel-wise spatial context scaling,
                <strong style="color:#34d399;"> Swin Transformer Base</strong> shifting window self-attention,
                and <strong style="color:#34d399;"> Pathology Class and Regression Severity Heads</strong>.
              </p>
            </div>
            <div style="font-family:JetBrains Mono;background:rgba(0,0,0,.62);border:1px solid #1A3A32;border-radius:10px;padding:12px;text-align:right;">
              <div style="color:#64748b;font-size:10px;">TOTAL ENSEMBLE PARAMS</div>
              <div style="color:#34d399;font-size:11px;font-weight:900;">42.8 Million Weights</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    controls, _ = st.columns([1.2, 2])
    with controls:
        zoom = st.slider("Interactive Architectural Schematic Zoom Vault", 50, 250, 100, step=25)
        arch_upload = st.file_uploader(
            "Upload PNG Diagram",
            type=["png", "jpg", "jpeg", "webp"],
            key="arch_upload",
        )
        process_arch_upload(arch_upload)

    if st.session_state.arch_image_uri:
        st.markdown(
            f"""
            <div class="pipeline">
              <img src="{st.session_state.arch_image_uri}" style="max-width:100%;max-height:620px;transform:scale({zoom / 100});transform-origin:center;" alt="User Uploaded Schematic"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="glass-panel" style="padding:16px;border-radius:24px;">
          <div class="pipeline">
            <div class="pipeline-box" style="transform:scale({zoom / 100});">
              <div style="position:absolute;top:10px;right:18px;color:#2da06e;font-size:9px;font-family:JetBrains Mono;font-weight:900;">ALEAF-STENET PIPELINE DIAGRAM v2.0</div>
              <div class="pipeline-grid">
                <div class="stage-card" style="background:#0C241E;">
                  <span class="stage-chip">STAGE 1: INPUT</span>
                  <span class="stage-title">Apple Foliage Specimen</span>
                  <div style="height:86px;background:#0f172a;border:1px solid #27272a;border-radius:8px;display:grid;place-items:center;color:#34d399;">{leaf_svg(42)}</div>
                  <div style="font-size:9px;color:#94a3b8;margin-top:13px;">224 x 224 RGB Pixels</div>
                </div>
                <div class="stage-card">
                  <span class="stage-chip">STAGE 2</span>
                  <span class="stage-title">EfficientNetV2-S Base</span>
                  <div class="mini-line">MBConv Texture</div>
                  <div class="mini-line">Fused-MBConv Layer</div>
                  <div class="mini-line">Spatial Reductions</div>
                  <div style="font-size:9px;color:#2dd4bf;font-weight:900;margin-top:13px;">Feature Extractor</div>
                </div>
                <div class="stage-card">
                  <span class="stage-chip">STAGE 3 (CBAM)</span>
                  <span class="stage-title">CBAM Attention Block</span>
                  <div class="mini-line" style="color:#5eead4;">Channel Attention Block</div>
                  <div class="mini-line" style="color:#bef264;">Spatial Attention Map</div>
                  <div style="font-size:9px;color:#34d399;font-weight:900;margin-top:13px;">Weights Calibration</div>
                </div>
                <div class="stage-card">
                  <span class="stage-chip">STAGE 4</span>
                  <span class="stage-title">Swin Transformer Base</span>
                  <div class="mini-line" style="background:rgba(0,0,0,.38);">Shifting Window MSA</div>
                  <div class="mini-line" style="background:rgba(0,0,0,.38);">Linear Embedding Modules</div>
                  <div style="font-size:9px;color:#64748b;text-transform:uppercase;margin-top:13px;">Continuous attention</div>
                </div>
                <div class="stage-card" style="background:#0C241E;">
                  <span class="stage-chip">STAGE 5: HEADS</span>
                  <span class="stage-title">Multi-Task Heads</span>
                  <div style="font-size:10px;color:#34d399;font-weight:900;">1. Classification Head</div>
                  <div style="font-size:10px;color:#2dd4bf;font-weight:900;margin-top:7px;">2. Severity Reg Head</div>
                  <div style="font-size:10px;color:#bef264;font-weight:900;margin-top:7px;">3. Attention Profiler</div>
                  <div style="font-size:9px;color:#94a3b8;margin-top:13px;">Softmax & MSE output</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-strip">
          <span>{html.escape(FOOTER_TEXT)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    inject_css()
    render_sidebar()
    render_topbar()
    render_back_control()
    st.write("")

    if st.session_state.active_tab == "Research":
        render_home()
    elif st.session_state.active_tab == "Diagnosis":
        render_diagnosis()
    else:
        render_architecture()

    render_footer()


if __name__ == "__main__":
    main()

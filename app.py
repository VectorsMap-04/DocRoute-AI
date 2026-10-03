import base64
import html
import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ============================================================
# LIGHT THEME (keeps everything in one file)
# Streamlit reads theme settings only at startup, so this writes
# .streamlit/config.toml once; it applies after the next restart.
# The CSS below also forces readable colors in dark mode.
# ============================================================

_THEME = """[theme]
base = "light"
primaryColor = "#2f5bea"
backgroundColor = "#f5f7fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#111827"
"""
_config_file = Path.cwd() / ".streamlit" / "config.toml"
if not _config_file.exists():
    try:
        _config_file.parent.mkdir(exist_ok=True)
        _config_file.write_text(_THEME)
    except OSError:
        pass


# ============================================================
# MODEL CONFIGURATION (unchanged)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "document_classifier_best.pth"
THRESHOLD = 0.90

CLASS_NAMES = ["Hand-Written", "Invoice-Doc", "Letter-Doc", "Resume-Doc"]

ROUTING = {
    "Hand-Written": "Manual Review",
    "Invoice-Doc": "Finance Department",
    "Letter-Doc": "Correspondence Department",
    "Resume-Doc": "Human Resources",
}

st.set_page_config(
    page_title="DocRoute AI — Intelligent Document Routing",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    """Load the trained ResNet18 checkpoint on CUDA when available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device


model, device = load_model()

transform = transforms.Compose(
    [
        transforms.Lambda(ImageOps.exif_transpose),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_document(image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    return (
        CLASS_NAMES[prediction.item()],
        confidence.item(),
        probabilities[0].cpu().numpy(),
    )


# ============================================================
# VISUAL SYSTEM
# Palette: ink #111827 · slate #64748b · line #e3e8ef · paper #f5f7fa
#          accent #2f5bea · ok #1b8a5a · hold #b7791f
# Type: Manrope (headings, numbers) + DM Sans (body)
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#111827;--slate:#64748b;--line:#e3e8ef;--paper:#f5f7fa;--accent:#2f5bea;--ok:#1b8a5a;--hold:#b7791f;--r:16px}
html{color-scheme:light}
html,body,[class*="css"]{font-family:'DM Sans',system-ui,sans-serif}
.stApp{background:var(--paper);color:var(--ink)}
[data-testid="stHeader"],footer,#MainMenu{visibility:hidden;height:0}
.block-container{max-width:1280px;padding:1.2rem clamp(1rem,4vw,3rem) 3rem}
h1,h2,h3,.display{font-family:Manrope,sans-serif}

/* top bar */
.topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:1rem}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:38px;height:38px;border-radius:11px;background:var(--accent);color:#fff;display:grid;place-items:center;font-size:18px;flex:none}
.brand-name{font:800 18px Manrope,sans-serif;letter-spacing:-.3px;line-height:1.1}
.brand-sub{color:var(--slate);font-size:12px;font-weight:500}
.status{display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border:1px solid var(--line);border-radius:99px;background:#fff;color:var(--ok);font-size:12px;font-weight:700;white-space:nowrap}
.status:before{content:'';width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px #1b8a5a22}

/* hero */
.hero{position:relative;overflow:hidden;background:#101a2e;border-radius:22px;padding:clamp(26px,5vw,46px);color:#fff}
.hero h1{color:#fff;font:800 clamp(30px,4.6vw,50px)/1.06 Manrope,sans-serif;letter-spacing:-.04em;margin:0 0 14px;max-width:11em}
.hero p{color:#b9c4d8;font-size:clamp(13px,1.5vw,15px);line-height:1.65;max-width:34em;margin:0}
.hero-doc{position:absolute;right:clamp(24px,8%,90px);top:50%;width:150px;transform:translateY(-50%) rotate(6deg);background:#fff;border-radius:12px;padding:18px 16px;box-shadow:0 24px 50px #0006}
.hero-doc i{display:block;height:6px;background:#e4e9f2;border-radius:4px;margin:9px 0}
.hero-doc i.a{background:#9fb5ff;width:70%;height:8px}.hero-doc i.s{width:55%}
.hero-doc b{display:block;margin-top:14px;font-size:11px;color:var(--accent)}

/* nav (st.radio) */
div[role="radiogroup"]{gap:8px;flex-wrap:wrap;margin:1.1rem 0 .4rem}
div[role="radiogroup"] label{background:#fff;border:1px solid var(--line);border-radius:10px;padding:7px 16px;transition:border-color .15s,background .15s}
div[role="radiogroup"] label:has(input:checked){border-color:var(--accent);background:#eef2ff;color:var(--accent);font-weight:700}
div[role="radiogroup"] label>div:first-child{display:none}
div[role="radiogroup"] label,div[role="radiogroup"] label *{color:var(--ink)!important}
div[role="radiogroup"] label:has(input:checked),div[role="radiogroup"] label:has(input:checked) *{color:var(--accent)!important}

/* section headings */
.sec{font:700 13px 'DM Sans',sans-serif;color:var(--slate);margin:26px 0 10px}
.page-heading{font:800 clamp(24px,3vw,30px) Manrope,sans-serif;letter-spacing:-.03em;margin:14px 0 4px}
.page-intro{color:var(--slate);font-size:14px;margin-bottom:6px;max-width:42em}

/* grids and cards */
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr))}
.grid.sm{grid-template-columns:repeat(auto-fit,minmax(min(100%,135px),1fr));gap:10px}
.card{background:#fff;border:1px solid var(--line);border-radius:var(--r);padding:18px 20px}
.metric-number{font:800 clamp(22px,2.6vw,27px) Manrope,sans-serif;letter-spacing:-.03em}
.metric-label{color:var(--slate);font-size:13px;margin-top:3px}
.callout{background:#eef2ff;border:1px solid #d9e2ff;border-radius:var(--r);padding:18px 22px;color:#334155}
.callout .big{font:800 30px Manrope,sans-serif;color:var(--accent);margin:2px 0}
.callout .sm{font-size:13px;color:var(--slate)}

/* upload */
.st-key-upload{background:#fff;border:1px solid var(--line)!important;border-radius:20px;padding:clamp(16px,3vw,26px)!important}
.upload-title{font:700 19px Manrope,sans-serif;margin-bottom:3px}
.upload-sub{color:var(--slate);font-size:13px;margin-bottom:12px}
[data-testid="stFileUploader"] section{border:1.5px dashed #c5d0e3;border-radius:14px;background:#f8faff;padding:20px}
[data-testid="stFileUploader"] section,[data-testid="stFileUploader"] section *{color:var(--ink)}
[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] small *{color:var(--slate)}
[data-testid="stFileUploader"] button{background:#fff!important;border:1px solid #c5d0e3!important;border-radius:10px}
[data-testid="stFileUploader"] button,[data-testid="stFileUploader"] button *{color:var(--ink)!important}
[data-testid="stFileUploader"] button:hover{border-color:var(--accent)!important}
[data-testid="stFileUploader"] section:hover{border-color:var(--accent);background:#f2f5ff}
.empty{text-align:center;padding:34px 18px;margin-top:16px}
.empty b{display:block;font:700 16px Manrope,sans-serif;margin:6px 0 4px}
.empty span{color:var(--slate);font-size:13px}

/* result */
.panel-title{font:700 15px Manrope,sans-serif;margin-bottom:12px}
.preview{background:#f1f4f9;border:1px solid var(--line);border-radius:12px;display:flex;align-items:center;justify-content:center;padding:10px;min-height:240px}
.preview img{max-width:100%;max-height:460px;object-fit:contain;border-radius:6px}
.fname{color:var(--slate);font-size:12px;margin-top:10px;overflow-wrap:anywhere}
.result-class{font:800 clamp(24px,3vw,36px)/1.1 Manrope,sans-serif;letter-spacing:-.03em;overflow-wrap:anywhere}
.conf{font:700 17px Manrope,sans-serif;color:var(--accent);margin:6px 0 16px}
.route{border-radius:14px;padding:16px 18px;margin-bottom:18px;border:1px solid}
.route.approved{background:#edf9f3;border-color:#c5ebd8}
.route.review{background:#fff8ec;border-color:#f0dcb4}
.route-label{font-size:12px;font-weight:700;margin-bottom:5px}
.approved .route-label{color:var(--ok)}.review .route-label{color:var(--hold)}
.route-dest{font:800 clamp(20px,2.4vw,26px) Manrope,sans-serif;letter-spacing:-.02em;overflow-wrap:anywhere}
.route-note{font-size:13px;color:var(--slate);line-height:1.5;margin-top:4px}
.meter-head{display:flex;justify-content:space-between;font-size:13px;color:var(--slate);font-weight:500;margin-bottom:8px}
.meter{position:relative;height:9px;background:#e8edf5;border-radius:9px;margin-bottom:6px}
.meter-fill{height:100%;border-radius:9px;background:var(--accent)}
.meter-mark{position:absolute;left:90%;top:-4px;width:2px;height:17px;background:var(--ink)}
.meter-note{font-size:12px;color:var(--slate);text-align:right;margin-bottom:18px}
.note{font-size:12px;color:var(--slate);margin:-6px 0 18px;line-height:1.5}
.prob{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:#fff}
.prob.sel{border-color:#9db2ff;background:#f5f7ff}
.prob-name{font-size:12px;font-weight:700;color:var(--slate);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.prob-val{font:700 16px Manrope,sans-serif;margin:3px 0 8px}
.track{height:4px;background:#e7ebf2;border-radius:4px;overflow:hidden}
.fill{height:100%;background:#a3b3e8}.sel .fill{background:var(--accent)}

/* workflow */
.flow{display:flex;gap:8px;align-items:stretch}
.step{flex:1;min-width:0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.step-l{font-size:12px;color:var(--slate);margin-bottom:5px}
.step-v{font-size:13px;font-weight:700;overflow-wrap:anywhere}
.arrow{align-self:center;color:#a3aec0}

/* tables */
.tbl-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:#fff}
table.tbl{border-collapse:collapse;width:100%;font-size:14px;min-width:420px}
.tbl th{text-align:left;font-weight:700;color:var(--slate);font-size:12px;padding:11px 16px;background:#f8fafc;border-bottom:1px solid var(--line)}
.tbl td{padding:12px 16px;border-bottom:1px solid #eef1f6}
.tbl tr:last-child td{border-bottom:0}
.tbl tr.hl td{background:#f2f5ff;font-weight:700;color:var(--accent)}

.footnote{color:var(--slate);font-size:12px;padding-top:16px;border-top:1px solid var(--line);margin-top:30px}

/* responsive */
@media(max-width:900px){
  div[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:1rem!important}
  div[data-testid="stColumn"],div[data-testid="column"]{flex:1 1 100%!important;min-width:100%!important;width:100%!important}
  .hero-doc{opacity:.18;right:-20px}
}
@media(max-width:640px){
  .brand-sub{display:none}
  .hero-doc{display:none}
  .flow{flex-direction:column}
  .arrow{transform:rotate(90deg);align-self:flex-start;margin-left:14px}
  .card{padding:15px 16px}
}
</style>
""",
    unsafe_allow_html=True,
)


def md(markup):
    st.markdown(markup, unsafe_allow_html=True)


def metric_card(value, label):
    return f'<div class="card"><div class="metric-number">{value}</div><div class="metric-label">{label}</div></div>'


def table_html(rows, highlight=None):
    head = "".join(f"<th>{k}</th>" for k in rows[0])
    body = ""
    for i, row in enumerate(rows):
        cls = ' class="hl"' if i == highlight else ""
        body += f"<tr{cls}>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def image_to_data_uri(image, max_side=900):
    preview = ImageOps.exif_transpose(image).convert("RGB")
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


# ============================================================
# HEADER
# ============================================================

md(
    '<div class="topbar"><div class="brand"><div class="logo">▤</div>'
    '<div><div class="brand-name">DocRoute AI</div><div class="brand-sub">Intelligent document operations</div></div></div>'
    '<div class="status">System online</div></div>'
    '<section class="hero"><div>'
    "<h1>Turn documents into decisions.</h1>"
    "<p>Classify incoming documents in seconds. High-confidence submissions go straight to the right team; uncertain ones are held for a person to review.</p>"
    '</div><div class="hero-doc"><i class="a"></i><i></i><i class="s"></i><i></i><b>AI verified ✓</b></div></section>'
)

page = st.radio(
    "Workspace",
    ["Dashboard", "Analytics", "Safety"],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# PAGES
# ============================================================

def render_dashboard():
    md('<div class="sec">Model at a glance</div>')
    cards = [
        ("92.83%", "Test accuracy"),
        ("92.86%", "Macro-F1"),
        ("90%", "Confidence threshold"),
        ("80%", "Out-of-scope rejection (8 of 10)"),
    ]
    md('<div class="grid">' + "".join(metric_card(v, l) for v, l in cards) + "</div>")

    md('<div class="sec">Document intake</div>')
    with st.container(border=True, key="upload"):
        md(
            '<div class="upload-title">Upload a document</div>'
            '<div class="upload-sub">Drop an invoice, letter, resume, or handwritten page. JPG, JPEG, PNG or WEBP.</div>'
        )
        uploaded_file = st.file_uploader(
            "Choose a document image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="document_upload",
        )

    if uploaded_file is None:
        md(
            '<div class="card empty"><div style="font-size:26px;color:#7990d8">＋</div>'
            "<b>Upload a document to get started</b>"
            "<span>You'll see its class, confidence, and where it gets routed.</span></div>"
        )
        return

    try:
        image = Image.open(uploaded_file)
        predicted_class, confidence, probabilities = predict_document(image)
        preview_uri = image_to_data_uri(image)
    except Exception as exc:
        st.error(
            f"This image could not be processed. Please try another JPG, PNG, or WEBP image. ({exc})"
        )
        return

    approved = confidence >= THRESHOLD
    destination = ROUTING[predicted_class] if approved else "Human Review"

    md('<div class="sec">Classification result</div>')
    preview_col, result_col = st.columns([0.9, 1.1], gap="medium")

    with preview_col:
        md(
            '<div class="card"><div class="panel-title">Document preview</div>'
            f'<div class="preview"><img src="{preview_uri}" alt="Uploaded document"></div>'
            f'<div class="fname">{html.escape(uploaded_file.name)}</div></div>'
        )

    with result_col:
        if approved:
            title = html.escape(predicted_class)
            conf_text = f"{confidence:.2%} confidence"
            style, heading = "approved", "✓ Automatic routing approved"
            note = "Confidence meets or exceeds the 90% routing threshold."
        else:
            title = "Possibly out-of-scope"
            conf_text = f"{confidence:.2%} top-class confidence"
            style, heading = "review", "! Manual review required"
            note = "Top confidence is below the 90% routing threshold, so automatic routing is blocked."

        extra = (
            f'<div class="note">Top model prediction: {html.escape(predicted_class)} ({confidence:.2%}). '
            "Low confidence can mean an unfamiliar, unreadable, or out-of-scope document.</div>"
            if not approved
            else ""
        )

        probs = "".join(
            f'<div class="prob {"sel" if name == predicted_class else ""}">'
            f'<div class="prob-name">{name}</div><div class="prob-val">{float(probabilities[i]):.2%}</div>'
            f'<div class="track"><div class="fill" style="width:{float(probabilities[i]) * 100:.2f}%"></div></div></div>'
            for i, name in enumerate(CLASS_NAMES)
        )

        md(
            '<div class="card">'
            '<div class="metric-label" style="margin:0 0 6px">AI classification</div>'
            f'<div class="result-class">{title}</div><div class="conf">{conf_text}</div>'
            f'<div class="route {style}"><div class="route-label">{heading}</div>'
            f'<div class="route-dest">{html.escape(destination)}</div><div class="route-note">{note}</div></div>'
            f"{extra}"
            f'<div class="meter-head"><span>Model confidence</span><span>{confidence:.2%}</span></div>'
            f'<div class="meter"><div class="meter-fill" style="width:{min(confidence, 1.0) * 100:.2f}%"></div><div class="meter-mark"></div></div>'
            '<div class="meter-note">90% routing threshold</div>'
            '<div class="panel-title">Class probabilities</div>'
            f'<div class="grid sm">{probs}</div></div>'
        )

    md('<div class="sec">Workflow trace</div>')
    steps = [
        ("Document", "Image received"),
        ("AI classification", predicted_class if approved else "Possibly out-of-scope"),
        ("Confidence gate", f"{confidence:.2%} vs 90%"),
        ("Routing decision", "Approved" if approved else "Review required"),
        ("Destination", destination),
    ]
    flow = '<div class="flow">'
    for i, (label, value) in enumerate(steps):
        flow += f'<div class="step"><div class="step-l">{label}</div><div class="step-v">{html.escape(value)}</div></div>'
        if i < len(steps) - 1:
            flow += '<div class="arrow">›</div>'
    md(flow + "</div>")


def render_analytics():
    md(
        '<div class="page-heading">Model analytics</div>'
        '<div class="page-intro">Transfer learning improves document recognition across all supported workflows.</div>'
    )
    md(
        '<div class="grid" style="margin-top:14px">'
        + metric_card("80.17%", "HOG baseline accuracy")
        + metric_card("92.83%", "ResNet18 test accuracy")
        + '<div class="callout"><div class="sm">Accuracy improvement</div><div class="big">+12.66 pp</div>'
        '<div class="sm">HOG + Logistic Regression → ResNet18</div></div></div>'
    )

    md('<div class="sec">Test set results</div>')
    md(
        table_html(
            [
                {"Model": "HOG + Logistic Regression", "Accuracy": "80.17%", "Macro-F1": "80.19%"},
                {"Model": "ResNet18 Transfer Learning", "Accuracy": "92.83%", "Macro-F1": "92.86%"},
            ],
            highlight=1,
        )
    )

    md('<div class="sec">Supported document workflows</div>')
    md(
        '<div class="grid sm">'
        + "".join(
            f'<div class="prob"><div class="prob-name">{name}</div>'
            f'<div style="font-size:13px;color:#64748b;margin-top:6px">Routed to <b style="color:#111827">{ROUTING[name]}</b></div></div>'
            for name in CLASS_NAMES
        )
        + "</div>"
    )


def render_safety():
    md(
        '<div class="page-heading">Safety and confidence gating</div>'
        '<div class="page-intro">A 90% confidence gate holds uncertain predictions for human review before anything is routed.</div>'
    )
    md(
        '<div class="callout" style="margin-top:14px"><div class="sm">Selected routing threshold</div>'
        '<div class="big">90%</div>'
        '<div class="sm">Validation accepted accuracy 98.23% · Manual review 5.83% · Misrouting 1.67%</div></div>'
    )

    md('<div class="sec">Threshold experiment</div>')
    md(
        table_html(
            [
                {"Threshold": "0.50", "Manual Review": "0.33%", "Misrouting": "3.50%", "Accepted Accuracy": "96.49%"},
                {"Threshold": "0.60", "Manual Review": "1.33%", "Misrouting": "2.83%", "Accepted Accuracy": "97.13%"},
                {"Threshold": "0.70", "Manual Review": "2.33%", "Misrouting": "2.67%", "Accepted Accuracy": "97.27%"},
                {"Threshold": "0.80", "Manual Review": "3.67%", "Misrouting": "2.50%", "Accepted Accuracy": "97.40%"},
                {"Threshold": "0.90 ✓", "Manual Review": "5.83%", "Misrouting": "1.67%", "Accepted Accuracy": "98.23%"},
            ],
            highlight=4,
        )
    )

    md('<div class="sec">Out-of-scope demonstration</div>')
    md(
        '<div class="grid">'
        + metric_card("10", "Images in demonstration")
        + metric_card("8 / 10", "Rejected at 0.90")
        + metric_card("80%", "Rejection rate")
        + "</div><div style='height:14px'></div>"
    )

    st.warning(
        "The out-of-scope set contains only 10 images. This is a small demonstration set, not a statistically representative benchmark."
    )
    st.info(
        "Confidence-based rejection is not guaranteed open-set recognition. Some unfamiliar inputs can still receive high confidence from the classifier."
    )


if page == "Dashboard":
    render_dashboard()
elif page == "Analytics":
    render_analytics()
else:
    render_safety()

md(
    '<div class="footnote">DocRoute AI · ResNet18 document classification · Inference device: '
    + str(device).upper()
    + "</div>"
)
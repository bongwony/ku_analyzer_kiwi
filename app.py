import io
import datetime
import pandas as pd
import streamlit as st
from kiwipiepy import Kiwi

# ── 페이지 설정 ───────────────────────────────────────
st.set_page_config(
    page_title="한국어 발화 분석기",
    page_icon="🗣️",
    layout="wide",
)

# ── Kiwi 초기화 (캐시: 앱 재시작 시에만 로드) ─────────
@st.cache_resource
def load_kiwi():
    return Kiwi()

kiwi = load_kiwi()

# ── 태그 집합 ─────────────────────────────────────────
PUNCT_TAGS = {'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB', 'UNKNOWN'}
CONTENT_TAGS = {
    'NNG', 'NNP', 'NNB', 'NP', 'NR',
    'VV',  'VA',
    'MM',  'MAG', 'MAJ',
}

# ── 핵심 분석 함수 ────────────────────────────────────
def run_analysis(text_content: str):
    sents = kiwi.split_into_sents(text_content)
    utterances = [s.text.strip() for s in sents if s.text.strip()]
    if not utterances:
        return None, None

    results        = []
    all_tokens_sf  = []
    all_cont_forms = []

    for i, utt in enumerate(utterances):
        tokens         = kiwi.tokenize(utt)
        content_tokens = [t for t in tokens if t.tag not in PUNCT_TAGS]
        cont_tok       = [t for t in content_tokens if t.tag in CONTENT_TAGS]
        func_tok       = [t for t in content_tokens if t.tag not in CONTENT_TAGS]

        sf_all     = [t.form for t in content_tokens]
        cont_forms = [t.form for t in cont_tok]

        all_tokens_sf.extend(sf_all)
        all_cont_forms.extend(cont_forms)

        morph_str = " ".join(f"{t.form}/{t.tag}" for t in content_tokens)

        results.append({
            "No":       i + 1,
            "발화":     utt,
            "어절수":   len(utt.split()),
            "형태소수": len(content_tokens),
            "내용어수": len(cont_tok),
            "기능어수": len(func_tok),
            "형태소분석": morph_str,
        })

    df    = pd.DataFrame(results)
    n_utt = len(df)

    token_n = len(all_tokens_sf)
    type_n  = len(set(all_tokens_sf))
    cont_n  = len(all_cont_forms)
    ndw     = len(set(all_cont_forms))
    ndw_50  = len(set(all_cont_forms[:50]))

    summary = {
        '발화수':    n_utt,
        'Token':     token_n,
        'Type':      type_n,
        'MLU':       round(token_n / n_utt, 2) if n_utt else 0,
        'MLU_w':     round(df['어절수'].mean(), 2),
        'TTR_전체':  round(type_n / token_n, 4)                    if token_n else 0,
        'TTR_내용어':round(len(set(all_cont_forms)) / cont_n, 4)   if cont_n  else 0,
        'NDW':       ndw,
        'NDW_50':    ndw_50,
        '내용어수':  cont_n,
        '기능어수':  token_n - cont_n,
    }
    return df, summary

# ── Excel 생성 (BytesIO) ──────────────────────────────
def make_excel(df: pd.DataFrame, summary: dict) -> bytes:
    summary_df = pd.DataFrame([
        {"지표": "발화수",      "값": summary["발화수"],     "설명": "총 발화 수"},
        {"지표": "Token",       "값": summary["Token"],      "설명": "총 형태소 수 (문장부호 제외)"},
        {"지표": "Type",        "값": summary["Type"],       "설명": "형태소 유형 수"},
        {"지표": "MLU",         "값": summary["MLU"],        "설명": "평균발화길이 (형태소/발화)"},
        {"지표": "MLU-w",       "값": summary["MLU_w"],      "설명": "평균발화길이 (어절/발화)"},
        {"지표": "TTR_전체",    "값": summary["TTR_전체"],   "설명": "Type / Token"},
        {"지표": "TTR_내용어",  "값": summary["TTR_내용어"], "설명": "내용어 Type/Token"},
        {"지표": "NDW",         "값": summary["NDW"],        "설명": "다른 단어 수 (내용어)"},
        {"지표": "NDW-50",      "값": summary["NDW_50"],     "설명": "첫 50 내용어 기준 NDW"},
        {"지표": "내용어수",    "값": summary["내용어수"],   "설명": "내용어 토큰 수"},
        {"지표": "기능어수",    "값": summary["기능어수"],   "설명": "기능어 토큰 수"},
    ])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="발화별 상세", index=False)
        summary_df.to_excel(writer, sheet_name="요약 통계", index=False)
    return buf.getvalue()

# ── 요약 통계 렌더링 ──────────────────────────────────
def render_summary(summary: dict):
    st.markdown("#### 📊 전체 요약 통계")
    st.caption("TTR_내용어·NDW는 내용어 형태소 표면형 기준")

    row1 = st.columns(5)
    row2 = st.columns(6)

    metrics1 = [
        ("발화수",  summary["발화수"],   "utterances"),
        ("Token",   summary["Token"],    "총 형태소 수"),
        ("Type",    summary["Type"],     "형태소 유형 수"),
        ("MLU",     summary["MLU"],      "형태소/발화"),
        ("MLU-w",   summary["MLU_w"],    "어절/발화"),
    ]
    metrics2 = [
        ("TTR 전체",   summary["TTR_전체"],   "Type/Token"),
        ("TTR 내용어", summary["TTR_내용어"], "내용어 Type/Token"),
        ("NDW",        summary["NDW"],        "다른 단어 수"),
        ("NDW-50",     summary["NDW_50"],     "첫 50 내용어"),
        ("내용어수",   summary["내용어수"],   "content words"),
        ("기능어수",   summary["기능어수"],   "function words"),
    ]

    for col, (label, val, delta) in zip(row1, metrics1):
        col.metric(label=label, value=val, delta=delta, delta_color="off")
    for col, (label, val, delta) in zip(row2, metrics2):
        col.metric(label=label, value=val, delta=delta, delta_color="off")

# ── UI ───────────────────────────────────────────────
st.markdown("## 🗣️ 한국어 발화 분석기 &nbsp; <span style='font-size:14px;color:#64748b;font-weight:400;'>Kiwi 버전 v09</span>", unsafe_allow_html=True)
st.caption("발화 분리 → 형태소 분석 → MLU · TTR · NDW 자동 산출")
st.divider()

tab_text, tab_file = st.tabs(["📝 직접 입력", "📁 파일 업로드"])

# ── 탭1: 직접 입력 ────────────────────────────────────
with tab_text:
    text_input = st.text_area(
        "발화 텍스트 입력",
        placeholder="발화를 입력하세요. 문장 단위로 자동 분리됩니다.\n\n예시:\n할머니가 손녀한테 선물을 줬어요.\n어린 여자 아이가 상자를 열어봤어요.\n상자 안에 예쁜 인형이 들어있었어요.",
        height=200,
        label_visibility="collapsed",
    )
    col_btn, col_clear = st.columns([1, 5])
    run_text = col_btn.button("분석 시작", type="primary", use_container_width=True, key="btn_text")

    if run_text and text_input.strip():
        with st.spinner("Kiwi 형태소 분석 중..."):
            df, summary = run_analysis(text_input)
        if df is not None:
            render_summary(summary)
            st.divider()
            st.markdown("#### 📋 발화별 상세 지표")
            st.dataframe(df, use_container_width=True, hide_index=True)

            date_str  = datetime.datetime.now().strftime("%Y%m%d")
            fname     = f"발화분석_직접입력_{date_str}.xlsx"
            excel_bytes = make_excel(df, summary)
            st.download_button(
                label="⬇️ Excel 저장",
                data=excel_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("발화를 인식하지 못했습니다. 텍스트를 확인해 주세요.")
    elif run_text:
        st.warning("텍스트를 입력해 주세요.")

# ── 탭2: 파일 업로드 ──────────────────────────────────
with tab_file:
    uploaded = st.file_uploader(
        ".txt 파일을 선택하세요 (UTF-8 인코딩)",
        type=["txt"],
        label_visibility="visible",
    )
    run_file = st.button("분석 시작", type="primary", key="btn_file")

    if run_file and uploaded is not None:
        text = uploaded.read().decode("utf-8")
        src_name = uploaded.name.rsplit(".", 1)[0]
        with st.spinner("Kiwi 형태소 분석 중..."):
            df, summary = run_analysis(text)
        if df is not None:
            render_summary(summary)
            st.divider()
            st.markdown("#### 📋 발화별 상세 지표")
            st.dataframe(df, use_container_width=True, hide_index=True)

            date_str    = datetime.datetime.now().strftime("%Y%m%d")
            fname       = f"발화분석_{src_name}_{date_str}.xlsx"
            excel_bytes = make_excel(df, summary)
            st.download_button(
                label="⬇️ Excel 저장",
                data=excel_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("발화를 인식하지 못했습니다. 파일 내용을 확인해 주세요.")
    elif run_file:
        st.warning("파일을 먼저 업로드해 주세요.")

# ── 사이드바: 지표 안내 ───────────────────────────────
with st.sidebar:
    st.markdown("### 📖 지표 설명")
    st.markdown("""
| 지표 | 설명 |
|---|---|
| **발화수** | 총 발화 수 |
| **Token** | 총 형태소 수 |
| **Type** | 형태소 유형 수 |
| **MLU** | 평균발화길이 (형태소) |
| **MLU-w** | 평균발화길이 (어절) |
| **TTR 전체** | Type / Token |
| **TTR 내용어** | 내용어 Type/Token |
| **NDW** | 다른 단어 수 |
| **NDW-50** | 첫 50 내용어 기준 NDW |
| **내용어수** | 명사·동사·형용사·부사 |
| **기능어수** | 조사·어미 등 |
""")
    st.divider()
    st.markdown("### ⚙️ 내용어 태그")
    st.code("NNG NNP NNB NP NR\nVV VA\nMM MAG MAJ", language=None)
    st.caption("분석 엔진: Kiwi (kiwipiepy)")

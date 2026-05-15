"""
한국어 발화 분석기 v13 (Streamlit 버전)
======================================
언어치료 및 발화 분석 연구를 위한 도구.

산출 지표:
  - 발화수 / Token / Type
  - MLU-e (어절) / MLU-w (단어, 학교문법) / MLU-m (형태소)
  - TTR_전체 / TTR_내용어
  - NDW / NDW-50
  - 내용어수 / 기능어수

옵션:
  - 파생접사(XSV·XSA·XSN·XSM·XPN) 분석 ON/OFF

실행:
  pip install streamlit kiwipiepy pandas openpyxl
  streamlit run app.py
"""

import io
import datetime
from collections import Counter
import pandas as pd
import streamlit as st
from kiwipiepy import Kiwi


# ──────────────────────────────────────────────────────
# Kiwi 인스턴스 (캐시: 매 rerun마다 재생성 방지)
# ──────────────────────────────────────────────────────
@st.cache_resource
def get_kiwi():
    return Kiwi()


kiwi = get_kiwi()


# ──────────────────────────────────────────────────────
# 태그 집합 정의
# ──────────────────────────────────────────────────────

# 문장부호·기호 (분석에서 완전 제외)
PUNCT_TAGS = {'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB', 'UNKNOWN'}

# 내용어 태그 (Kiwi 기준)
#   명사류: NNG NNP NNB NP NR
#   용언류: VV VA  (VX 보조용언은 기능어로 분류)
#   수식언: MM MAG MAJ
CONTENT_TAGS = {'NNG', 'NNP', 'NNB', 'NP', 'NR',
                'VV',  'VA',
                'MM',  'MAG', 'MAJ'}

# 학교문법 기반 '단어' 단위 태그 (MLU-w 산정용)
#  ─ 단어로 카운트:
#     · 체언  NNG NNP NNB NP NR
#     · 용언  VV VA VX VCP VCN  (어간 1개 = 단어 1개. 어미는 제외)
#     · 수식언 MM MAG MAJ
#     · 독립언 IC
#     · 관계언(조사) JKS JKC JKG JKO JKB JKV JKQ JX JC
#     · 기타 자립형식 SL(외국어) SH(한자) SN(숫자)
WORD_TAGS = {
    'NNG', 'NNP', 'NNB', 'NP', 'NR',
    'VV', 'VA', 'VX', 'VCP', 'VCN',
    'MM', 'MAG', 'MAJ',
    'IC',
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
    'SL', 'SH', 'SN',
}


# ──────────────────────────────────────────────────────
# 태그 정규화 (Kiwi는 불규칙 활용을 'VV-I', 'XSA-I' 등으로 태그함)
# ──────────────────────────────────────────────────────
def base_tag(tag):
    """'VV-I' → 'VV', 'XSA-I' → 'XSA' 처럼 접미사를 떼어낸 기본 태그를 반환."""
    return tag.split('-')[0]


# ──────────────────────────────────────────────────────
# 파생접사 결합 (옵션 OFF용)
# ──────────────────────────────────────────────────────
DERIV_SUFFIX_TAGS = {'XSV', 'XSA', 'XSN', 'XSM'}
SUFFIX_TO_TAG = {
    'XSV': 'VV',   # 동사파생  (예: 공부+하 → 공부하/VV)
    'XSA': 'VA',   # 형용사파생 (예: 행복+하 → 행복하/VA)
    'XSN': 'NNG',  # 명사파생  (예: 가능+성 → 가능성/NNG)
    'XSM': 'MAG',  # 부사파생
}
DERIV_PREFIX_TAGS = {'XPN'}


class _Tok:
    """결합 후 토큰을 표현하기 위한 가벼운 클래스."""
    __slots__ = ('form', 'tag')
    def __init__(self, form, tag):
        self.form = form
        self.tag = tag


def merge_derivational_affixes(tokens):
    """파생접사를 인접 어근에 결합한 토큰 리스트를 반환.

    - XSV/XSA/XSN/XSM: 앞 토큰과 결합, 태그는 SUFFIX_TO_TAG 매핑
    - XPN: 뒤 토큰과 결합, 태그는 뒤 토큰 태그 유지
    """
    out = []
    toks = list(tokens)
    i = 0
    n = len(toks)
    while i < n:
        cur = toks[i]
        cur_base = base_tag(cur.tag)

        # 접두사 + 다음 어근 결합
        if cur_base in DERIV_PREFIX_TAGS and i + 1 < n:
            nxt = toks[i + 1]
            out.append(_Tok(cur.form + nxt.form, nxt.tag))
            i += 2
            continue

        # 일반 토큰 + 다음 토큰이 파생접미사면 결합
        if i + 1 < n:
            nxt_base = base_tag(toks[i + 1].tag)
            if nxt_base in DERIV_SUFFIX_TAGS:
                new_tag = SUFFIX_TO_TAG.get(nxt_base, cur.tag)
                out.append(_Tok(cur.form + toks[i + 1].form, new_tag))
                i += 2
                continue

        out.append(_Tok(cur.form, cur.tag))
        i += 1
    return out


# ──────────────────────────────────────────────────────
# 발화 분리 (줄바꿈 강제 경계 + Kiwi 자동 분리)
# ──────────────────────────────────────────────────────
def split_utterances(text):
    """입력 텍스트를 발화 단위로 분리.

    - 줄바꿈은 강제 발화 경계로 존중 (사용자가 명시한 경계는 절대 합쳐지지 않음).
    - 한 줄 안에 여러 발화가 있으면 Kiwi의 split_into_sents()로 추가 분리.
    - 빈 줄과 앞뒤 공백은 무시.
    """
    utterances = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        sents = kiwi.split_into_sents(line)
        for s in sents:
            t = s.text.strip()
            if t:
                utterances.append(t)
    return utterances


# ──────────────────────────────────────────────────────
# 핵심 분석 함수
# ──────────────────────────────────────────────────────
def run_analysis(text_content, analyze_affixes=False):
    """발화 분석 메인 함수.

    Args:
        text_content (str): 분석할 텍스트
        analyze_affixes (bool): False면 파생접사를 어근에 결합해 단일 토큰으로 처리 (기본값).
                                True면 파생접사를 별개 형태소로 분석.

    Returns:
        (DataFrame, dict) 또는 (None, None) - 발화가 없을 때
    """
    utterances = split_utterances(text_content)
    if not utterances:
        return None, None

    results = []
    all_tokens_sf = []   # 전체 형태소 표면형 (문장부호 제외)
    all_cont_forms = []  # 내용어 형태소 표면형
    all_cont_pairs = []  # 내용어 (표면형, 품사) 페어 — NDW 단어 리스트용

    for i, utt in enumerate(utterances):
        tokens = kiwi.tokenize(utt)

        # 옵션 적용: 접사 분석 OFF면 파생접사를 어근에 결합
        if not analyze_affixes:
            tokens = merge_derivational_affixes(tokens)

        # 문장부호 제외 (= 형태소 토큰)
        morph_tokens = [t for t in tokens if base_tag(t.tag) not in PUNCT_TAGS]

        # 내용어 / 기능어 분류
        cont_tok = [t for t in morph_tokens if base_tag(t.tag) in CONTENT_TAGS]
        func_tok = [t for t in morph_tokens if base_tag(t.tag) not in CONTENT_TAGS]

        # 학교문법 '단어' 토큰: 어미·접사·문장부호 제외, 조사는 포함
        word_tokens = [t for t in morph_tokens if base_tag(t.tag) in WORD_TAGS]

        # 표면형
        sf_all = [t.form for t in morph_tokens]
        cont_forms = [t.form for t in cont_tok]
        cont_pairs = [(t.form, base_tag(t.tag)) for t in cont_tok]

        all_tokens_sf.extend(sf_all)
        all_cont_forms.extend(cont_forms)
        all_cont_pairs.extend(cont_pairs)

        morph_str = " ".join(f"{t.form}/{t.tag}" for t in morph_tokens)

        results.append({
            "No":          i + 1,
            "발화":        utt,
            "어절수":      len(utt.split()),       # MLU-e 단위
            "단어수":      len(word_tokens),       # MLU-w 단위 (학교문법)
            "형태소수":    len(morph_tokens),      # MLU-m 단위
            "내용어수":    len(cont_tok),
            "기능어수":    len(func_tok),
            "형태소분석":  morph_str,
        })

    df = pd.DataFrame(results)
    n_utt = len(df)

    # 전체 요약 지표
    token_n = len(all_tokens_sf)
    type_n  = len(set(all_tokens_sf))
    cont_n  = len(all_cont_forms)
    func_n  = token_n - cont_n
    ndw     = len(set(all_cont_forms))                # NDW: 표면형 기준 (기존 호환)
    ndw_50  = len(set(all_cont_forms[:50]))           # NDW-50: 첫 50 내용어 토큰 기준

    # NDW 단어 리스트 (표면형, 품사) 페어 기준 + 빈도
    # 전체 NDW 어휘
    ndw_counter = Counter(all_cont_pairs)
    ndw_words_df = pd.DataFrame(
        [{"단어": f, "품사": t, "빈도": c} for (f, t), c in ndw_counter.most_common()]
    )
    # NDW-50 어휘 (첫 50개 내용어 토큰에서 추출)
    ndw50_counter = Counter(all_cont_pairs[:50])
    ndw50_words_df = pd.DataFrame(
        [{"단어": f, "품사": t, "빈도": c} for (f, t), c in ndw50_counter.most_common()]
    )

    summary = {
        '발화수':       n_utt,
        'Token':        token_n,
        'Type':         type_n,
        'MLU_e':        round(df['어절수'].mean(),   2),
        'MLU_w':        round(df['단어수'].mean(),   2),
        'MLU_m':        round(df['형태소수'].mean(), 2),
        'TTR_전체':     round(type_n / token_n, 4)               if token_n else 0,
        'TTR_내용어':   round(len(set(all_cont_forms)) / cont_n, 4) if cont_n else 0,
        'NDW':          ndw,
        'NDW_50':       ndw_50,
        '내용어수':      cont_n,
        '기능어수':      func_n,
        '_접사분석':     analyze_affixes,
        '_NDW_words':    ndw_words_df,    # 전체 NDW 단어 리스트 (단어/품사/빈도)
        '_NDW50_words':  ndw50_words_df,  # NDW-50 단어 리스트
    }
    return df, summary


# ──────────────────────────────────────────────────────
# Excel 다운로드용 바이트 생성
# ──────────────────────────────────────────────────────
def build_excel_bytes(df, summary):
    """발화별 상세 + 요약 통계 + NDW 어휘 시트로 구성된 Excel 바이트를 반환."""
    summary_df = pd.DataFrame([
        {"지표": "발화수",     "값": summary["발화수"],    "설명": "총 발화 수"},
        {"지표": "Token",      "값": summary["Token"],     "설명": "총 형태소 수 (문장부호 제외)"},
        {"지표": "Type",       "값": summary["Type"],      "설명": "형태소 유형 수"},
        {"지표": "MLU-e",      "값": summary["MLU_e"],     "설명": "평균발화길이 (어절/발화)"},
        {"지표": "MLU-w",      "값": summary["MLU_w"],     "설명": "평균발화길이 (단어/발화, 학교문법: 조사 별개 단어)"},
        {"지표": "MLU-m",      "값": summary["MLU_m"],     "설명": "평균발화길이 (형태소/발화)"},
        {"지표": "TTR_전체",   "값": summary["TTR_전체"],  "설명": "Type / Token"},
        {"지표": "TTR_내용어", "값": summary["TTR_내용어"],"설명": "내용어 Type/Token"},
        {"지표": "NDW",        "값": summary["NDW"],       "설명": "다른 단어 수 (내용어)"},
        {"지표": "NDW-50",     "값": summary["NDW_50"],    "설명": "첫 50 내용어 기준 NDW"},
        {"지표": "내용어수",   "값": summary["내용어수"],  "설명": "내용어 토큰 수"},
        {"지표": "기능어수",   "값": summary["기능어수"],  "설명": "기능어 토큰 수"},
        {"지표": "접사 분석",  "값": "ON" if summary.get('_접사분석', True) else "OFF",
                              "설명": "파생접사(XSV·XSA·XSN·XSM·XPN)를 별개 형태소로 분석할지 여부"},
    ])

    ndw_words_df   = summary.get('_NDW_words',   pd.DataFrame())
    ndw50_words_df = summary.get('_NDW50_words', pd.DataFrame())

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="발화별 상세", index=False)
        summary_df.to_excel(writer, sheet_name="요약 통계", index=False)
        if not ndw_words_df.empty:
            ndw_words_df.to_excel(writer, sheet_name="NDW 어휘", index=False)
        if not ndw50_words_df.empty:
            ndw50_words_df.to_excel(writer, sheet_name="NDW-50 어휘", index=False)
    buf.seek(0)
    return buf.getvalue()


# ──────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="한국어 발화 분석기",
    page_icon="🗣️",
    layout="wide",
)

st.title("🗣️ 한국어 발화 분석기")
st.caption("Kiwi 형태소 분석 기반 · MLU-e / MLU-w / MLU-m 산출 · 언어치료 및 발화 분석 연구 도구")

# ── 사이드바: 옵션 + 안내 ─────────────────────────────
with st.sidebar:
    st.header("⚙️ 분석 옵션")
    analyze_affixes = st.checkbox(
        "파생접사 분석 (ON/OFF)",
        value=False,
        help=(
            "**ON**: 파생접사(XSV·XSA·XSN·XSM·XPN)를 별개 형태소로 분석\n\n"
            "**OFF**: 접사를 어근에 결합 (예: 공부+하/XSV → 공부하/VV) — 기본값"
        ),
    )
    st.caption(
        "예: '공부하다'\n"
        "- OFF → 공부하/VV · 다/EF (2형태소) — 기본값\n"
        "- ON  → 공부/NNG · 하/XSV · 다/EF (3형태소)"
    )

    st.divider()
    with st.expander("📖 산출 지표 설명"):
        st.markdown("""
| 지표 | 기준 |
|------|------|
| **MLU-e** | 어절/발화 (공백 분리) |
| **MLU-w** | 단어/발화 (학교문법: 조사 별개) |
| **MLU-m** | 형태소/발화 |
| Token | 총 형태소 수 |
| Type | 형태소 유형 수 |
| TTR | Type / Token |
| NDW | 내용어 종류 수 |
| NDW-50 | 첫 50 내용어 기준 NDW |
""")
    with st.expander("📐 MLU 비교 (예: '시간이 있다')"):
        st.markdown("""
| 방식 | 분석 | 길이 |
|------|------|:----:|
| MLU-e | `시간이` / `있다` | **2** |
| MLU-w | `시간` / `이` / `있다` | **3** |
| MLU-m | `시간` / `이` / `있` / `다` | **4** |
""")
    with st.expander("✂️ 발화 분리 규칙"):
        st.markdown("""
- **줄바꿈은 강제 발화 경계** (사용자가 엔터로 구분한 줄은 별개 발화)
- 한 줄 안에 여러 발화가 있으면 Kiwi가 구두점·문법 단서로 추가 분리
- 빈 줄·앞뒤 공백은 무시
- 구두점 없는 구어체는 엔터로 구분해 주세요
""")

# ── 입력 영역: 탭 ─────────────────────────────────────
tab_text, tab_file = st.tabs(["📝 직접 입력", "📁 파일 업로드"])

text_content = None
source_title = "분석 결과"

with tab_text:
    user_text = st.text_area(
        "발화 입력",
        height=220,
        placeholder="발화를 입력하세요. 구두점이 없는 구어체는 엔터로 구분해 주세요.",
        key="text_input",
    )
    analyze_text_clicked = st.button("📝 입력 텍스트 분석", type="primary", key="btn_text")
    if analyze_text_clicked:
        text_content = user_text
        source_title = "직접 입력 텍스트 분석 결과"

with tab_file:
    uploaded = st.file_uploader(
        ".txt 파일을 선택해 주세요",
        type=["txt"],
        accept_multiple_files=False,
        key="file_input",
    )
    analyze_file_clicked = st.button("📁 업로드 파일 분석", key="btn_file")
    if analyze_file_clicked and uploaded is not None:
        try:
            text_content = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("UTF-8 인코딩이 아닙니다. 파일 인코딩을 UTF-8로 저장해 주세요.")
            text_content = None
        source_title = f"파일 '{uploaded.name}' 분석 결과"
    elif analyze_file_clicked and uploaded is None:
        st.warning("파일을 먼저 업로드해 주세요.")

# ── 분석 실행 ──────────────────────────────────────────
if text_content is not None and text_content.strip():
    df, summary = run_analysis(text_content, analyze_affixes=analyze_affixes)

    if df is None:
        st.warning("분석할 발화가 없습니다. 텍스트를 확인해 주세요.")
    else:
        # 결과 헤더 + 옵션 상태 배지
        affix_on = summary.get('_접사분석', False)
        affix_label = "접사분석 ON" if affix_on else "접사분석 OFF (접사 결합)"

        st.divider()
        col_title, col_badge = st.columns([4, 1])
        with col_title:
            st.subheader(f"📊 {source_title}")
        with col_badge:
            # 두 상태 모두 정보 표시(info)로 — OFF가 기본값이므로 경고 색은 부적절
            st.info(affix_label)

        # ── 요약 통계: st.metric 그리드 ─────────────────
        st.markdown("##### ▸ 발화 / 형태소 기반")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("발화수",  summary['발화수'])
        m2.metric("Token",   summary['Token'],   help="총 형태소 수 (문장부호 제외)")
        m3.metric("Type",    summary['Type'],    help="형태소 유형 수")
        m4.metric("MLU-e",   summary['MLU_e'],   help="어절/발화")
        m5.metric("MLU-w",   summary['MLU_w'],   help="단어/발화 (학교문법)")
        m6.metric("MLU-m",   summary['MLU_m'],   help="형태소/발화")

        st.markdown("##### ▸ 어휘 다양도 / 내용어·기능어")
        n1, n2, n3, n4, n5, n6 = st.columns(6)
        n1.metric("TTR 전체",  summary['TTR_전체'],   help="Type / Token")
        n2.metric("TTR 내용어", summary['TTR_내용어'], help="내용어 Type / Token")
        n3.metric("NDW",       summary['NDW'],        help="다른 단어 수 (내용어)")
        n4.metric("NDW-50",    summary['NDW_50'],     help="첫 50 내용어 기준")
        n5.metric("내용어수",  summary['내용어수'])
        n6.metric("기능어수",  summary['기능어수'])

        # ── 결과 탭: 발화별 상세 / NDW 어휘 / NDW-50 어휘 ──
        ndw_words_df   = summary['_NDW_words']
        ndw50_words_df = summary['_NDW50_words']

        tab_detail, tab_ndw, tab_ndw50 = st.tabs([
            "📋 발화별 상세 지표",
            f"📚 NDW 어휘 ({summary['NDW']}개)",
            f"📚 NDW-50 어휘 ({summary['NDW_50']}개)",
        ])

        with tab_detail:
            display_cols = ["No", "발화", "어절수", "단어수", "형태소수",
                            "내용어수", "기능어수", "형태소분석"]
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "No":         st.column_config.NumberColumn(width="small"),
                    "발화":       st.column_config.TextColumn(width="medium"),
                    "형태소분석": st.column_config.TextColumn(width="large"),
                },
            )

        with tab_ndw:
            st.caption(
                f"전체 발화에 사용된 서로 다른 내용어 {summary['NDW']}개 "
                f"(내용어 토큰 총 {summary['내용어수']}개 중). "
                "같은 표면형이라도 품사가 다르면 별개 단어로 카운트됩니다."
            )
            if ndw_words_df.empty:
                st.info("내용어가 추출되지 않았습니다.")
            else:
                st.dataframe(
                    ndw_words_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "단어": st.column_config.TextColumn(width="medium"),
                        "품사": st.column_config.TextColumn(width="small"),
                        "빈도": st.column_config.NumberColumn(width="small"),
                    },
                )

        with tab_ndw50:
            st.caption(
                f"첫 50개 내용어 토큰에서 추출된 서로 다른 단어 {summary['NDW_50']}개. "
                "표본 크기를 50으로 통제한 NDW로, 발화량 차이에 따른 편향을 줄이는 데 쓰입니다."
            )
            if ndw50_words_df.empty:
                st.info("내용어가 50개 미만이거나 추출되지 않았습니다.")
            else:
                st.dataframe(
                    ndw50_words_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "단어": st.column_config.TextColumn(width="medium"),
                        "품사": st.column_config.TextColumn(width="small"),
                        "빈도": st.column_config.NumberColumn(width="small"),
                    },
                )

        # ── Excel 다운로드 ─────────────────────────────
        src_label = '직접입력' if '직접' in source_title else (
            source_title.split("'")[1].rsplit('.', 1)[0]
            if "'" in source_title else 'unknown'
        )
        affix_tag = '접사ON' if affix_on else '접사OFF'
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        fname = f"발화분석_{src_label}_{affix_tag}_{date_str}.xlsx"

        excel_bytes = build_excel_bytes(df, summary)
        st.download_button(
            label="⬇️ Excel 저장 (발화별 상세 + 요약 통계 + NDW 어휘)",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

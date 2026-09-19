# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="외국인 Smart Money 분석기",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>

.block-container {
    padding-top: 0.7rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1 {
    margin-bottom: 0.1rem;
}

[data-testid="stMetric"] {
    background: #f7f9fc;
    border: 1px solid #e5eaf2;
    padding: 12px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    font-size: 1.45rem;
    font-weight: 700;
}

div[data-baseweb="tab-list"] {
    gap: 6px;
}

@media (max-width: 700px) {

    .block-container {
        padding-left: 0.55rem;
        padding-right: 0.55rem;
        padding-top: 0.4rem;
    }

    h1 {
        font-size: 1.55rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.15rem;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 점수 함수
# =========================================================

def accumulation_score(rate):

    if rate <= 0:
        return 0
    elif rate < 1:
        return 5
    elif rate < 2:
        return 10
    elif rate < 3:
        return 15
    elif rate < 5:
        return 20
    elif rate < 7:
        return 25
    else:
        return 30


def ratio_score(ratio, max_score):

    if ratio < 0.40:
        return 0
    elif ratio < 0.50:
        return round(max_score / 6)
    elif ratio < 0.60:
        return round(max_score * 2 / 6)
    elif ratio < 0.70:
        return round(max_score * 3 / 6)
    elif ratio < 0.80:
        return round(max_score * 4 / 6)
    elif ratio < 0.90:
        return round(max_score * 5 / 6)
    else:
        return max_score


# =========================================================
# Smart Money Score
# =========================================================

def smart_money_scores(data, floating):

    d = (
        data
        .dropna(subset=["date", "foreign"])
        .sort_values("date")
        .copy()
    )

    net = float(d["foreign"].sum())

    # 1. 누적 매집 규모 /30
    if floating > 0:

        rate = net / floating * 100
        s1 = accumulation_score(rate)

    else:

        rate = 0
        s1 = 0


    # 2. 매집 지속성 /30
    # 장기 추세이므로 월 단위

    monthly_flow = (
        d
        .set_index("date")["foreign"]
        .resample("ME")
        .sum()
    )

    positive_ratio = (
        float((monthly_flow > 0).mean())
        if len(monthly_flow)
        else 0
    )

    s2 = ratio_score(
        positive_ratio,
        30
    )


    # 3. 하락·횡보 구간 매수 /20

    monthly_price = (
        d
        .dropna(subset=["price"])
        .set_index("date")["price"]
        .resample("ME")
        .agg(["first", "last"])
    )

    mf = (
        d
        .set_index("date")["foreign"]
        .resample("ME")
        .sum()
    )

    monthly = (
        monthly_price
        .join(
            mf.rename("foreign"),
            how="inner"
        )
        .dropna()
    )

    if len(monthly):

        monthly["return"] = (
            monthly["last"]
            / monthly["first"]
            - 1
        )

        weak = monthly[
            monthly["return"] <= 0.03
        ]

        weak_buy_ratio = (
            float(
                (weak["foreign"] > 0).mean()
            )
            if len(weak)
            else 0
        )

    else:

        weak = pd.DataFrame()
        weak_buy_ratio = 0


    s3 = ratio_score(
        weak_buy_ratio,
        20
    )


    # 4. 최근 매집 가속도 /20

    flow = d["foreign"].to_numpy(float)

    recent = (
        float(np.mean(flow[-60:]))
        if len(flow)
        else 0
    )

    if len(flow) >= 120:

        prior = float(
            np.mean(flow[-120:-60])
        )

    elif len(flow) > 60:

        prior = float(
            np.mean(flow[:-60])
        )

    else:

        prior = 0


    if recent <= 0:

        s4 = 0
        acceleration_text = "최근 60거래일 평균 순매수 ≤ 0"

    elif prior <= 0:

        s4 = 20
        acceleration_text = "직전 구간 순매도/중립 → 최근 순매수"

    else:

        ratio = recent / prior

        if ratio >= 2:
            s4 = 20

        elif ratio >= 1.5:
            s4 = 16

        elif ratio >= 1:
            s4 = 12

        elif ratio >= 0.5:
            s4 = 8

        else:
            s4 = 4

        acceleration_text = (
            f"최근/직전 60거래일 평균 {ratio:.2f}배"
        )


    return {

        "net": net,
        "rate": rate,

        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,

        "total": s1 + s2 + s3 + s4,

        "positive_ratio": positive_ratio,
        "weak_buy_ratio": weak_buy_ratio,

        "recent": recent,
        "prior": prior,

        "acceleration_text": acceleration_text
    }


# =========================================================
# Score 변화
# =========================================================

def build_score_history(data, floating):

    d = (
        data
        .dropna(subset=["date", "foreign"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    rows = []

    for i in range(len(d)):

        part = d.iloc[:i + 1]

        score = smart_money_scores(
            part,
            floating
        )

        rows.append({

            "date": d.loc[i, "date"],

            "price": d.loc[i, "price"],

            "total": score["total"],

            "s1": score["s1"],
            "s2": score["s2"],
            "s3": score["s3"],
            "s4": score["s4"]
        })

    return pd.DataFrame(rows)


# =========================================================
# 가격대별 외국인 매물대
# =========================================================

def build_price_profile(data, bins=18):

    d = (
        data
        .dropna(subset=["price", "foreign"])
        .copy()
    )

    if d.empty:
        return pd.DataFrame()


    low = float(d["price"].min())
    high = float(d["price"].max())


    if low == high:
        return pd.DataFrame()


    edges = np.linspace(
        low,
        high,
        bins + 1
    )


    d["price_bucket"] = pd.cut(
        d["price"],
        bins=edges,
        include_lowest=True,
        duplicates="drop"
    )


    profile = (
        d
        .groupby(
            "price_bucket",
            observed=True
        )
        .agg(

            net_foreign=(
                "foreign",
                "sum"
            ),

            buy_foreign=(
                "foreign",
                lambda x:
                x[x > 0].sum()
            ),

            sell_foreign=(
                "foreign",
                lambda x:
                -x[x < 0].sum()
            )
        )
        .reset_index()
    )


    profile["price_mid"] = (
        profile["price_bucket"]
        .apply(
            lambda x:
            (x.left + x.right) / 2
        )
        .astype(float)
    )


    return profile


# =========================================================
# 날짜 파싱
# 실제 파일 A열 형식: '20/03/20
# =========================================================

def parse_dates(series):

    text = (
        series
        .astype(str)
        .str.strip()
        .str.replace(
            "'",
            "",
            regex=False
        )
    )


    result = pd.to_datetime(
        text,
        format="%y/%m/%d",
        errors="coerce"
    )


    missing = result.isna()

    if missing.any():

        result.loc[missing] = (
            pd.to_datetime(
                text.loc[missing],
                errors="coerce"
            )
        )


    return result


# =========================================================
# 제목
# =========================================================

st.title(
    "📈 외국인 Smart Money 분석기"
)

st.caption(
    "외국인의 장기 누적 매집과 주가 흐름을 한눈에 분석합니다."
)


# =========================================================
# 엑셀 업로드
# =========================================================

uploaded = st.file_uploader(
    "엑셀 불러오기",
    type=["xlsx", "xls"]
)


if uploaded is None:

    st.info(
        "분석할 엑셀 파일을 선택해주세요."
    )

    st.stop()


# =========================================================
# Sheet1 읽기
# =========================================================

try:

    excel = pd.ExcelFile(uploaded)

    if "Sheet1" in excel.sheet_names:

        raw = pd.read_excel(
            uploaded,
            sheet_name="Sheet1"
        )

    else:

        raw = pd.read_excel(
            uploaded,
            sheet_name=excel.sheet_names[-1]
        )

except Exception as e:

    st.error(
        f"엑셀을 읽지 못했습니다: {e}"
    )

    st.stop()


# =========================================================
# 실제 파일 구조 확인
#
# A열 = 일자
# B열 = 종가
# H열 = 외국인 일별 순매수
# =========================================================

if raw.shape[1] < 8:

    st.error(
        "필요한 A열, B열, H열을 찾을 수 없습니다."
    )

    st.stop()


df = pd.DataFrame({

    "date":
        parse_dates(
            raw.iloc[:, 0]
        ),

    "price":
        pd.to_numeric(
            raw.iloc[:, 1],
            errors="coerce"
        ),

    "foreign":
        pd.to_numeric(
            raw.iloc[:, 7],
            errors="coerce"
        )
})


df = (
    df
    .dropna(subset=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)


if df.empty:

    st.error(
        "정상적인 날짜 데이터를 찾지 못했습니다."
    )

    st.stop()


# =========================================================
# 비정상 값 제거
# =========================================================

df = df[
    (df["date"] >= pd.Timestamp("2000-01-01"))
    &
    (
        df["date"]
        <=
        pd.Timestamp.today()
        + pd.Timedelta(days=10)
    )
].copy()


df.loc[
    df["price"] <= 0,
    "price"
] = np.nan


if df.empty:

    st.error(
        "분석할 데이터가 없습니다."
    )

    st.stop()


min_date = df["date"].min().date()
max_date = df["date"].max().date()


# =========================================================
# 분석 설정
# =========================================================

with st.expander(
    "⚙️ 분석 설정",
    expanded=True
):

    c1, c2 = st.columns(2)


    with c1:

        stock_name = st.text_input(
            "종목명",
            value=
            uploaded.name
            .rsplit(".", 1)[0]
        )


        listed = st.number_input(
            "상장주식수",
            min_value=0,
            value=0,
            step=1000000,
            format="%d"
        )


        floating = st.number_input(
            "유동주식수",
            min_value=0,
            value=0,
            step=1000000,
            format="%d"
        )


    with c2:

        analysis_period = st.date_input(

            "분석기간",

            value=(
                min_date,
                max_date
            ),

            min_value=min_date,

            max_value=max_date
        )


        if listed > 0 and floating > 0:

            floating_ratio = (
                floating
                / listed
                * 100
            )

            st.metric(
                "유동주식 비율",
                f"{floating_ratio:.2f}%"
            )

        else:

            floating_ratio = 0

            st.metric(
                "유동주식 비율",
                "-"
            )


# =========================================================
# 기간 선택
# =========================================================

if (
    isinstance(
        analysis_period,
        (tuple, list)
    )
    and
    len(analysis_period) == 2
):

    start_date = analysis_period[0]
    end_date = analysis_period[1]

else:

    st.info(
        "분석기간 시작일과 종료일을 선택해주세요."
    )

    st.stop()


# =========================================================
# 분석기간 필터
# =========================================================

x = df[
    (
        df["date"]
        >=
        pd.Timestamp(start_date)
    )
    &
    (
        df["date"]
        <=
        pd.Timestamp(end_date)
    )
].copy()


x = (
    x
    .sort_values("date")
    .reset_index(drop=True)
)


if x.empty:

    st.error(
        "선택한 기간에 데이터가 없습니다."
    )

    st.stop()


# =========================================================
# 입력값 확인
# =========================================================

if listed > 0 and floating > listed:

    st.warning(
        "유동주식수가 상장주식수보다 많습니다. 입력값을 확인해주세요."
    )


if floating <= 0:

    st.warning(
        "유동주식수를 입력하면 유동주식 대비 매집률과 Smart Money Score가 계산됩니다."
    )


# =========================================================
# Score
# =========================================================

score = smart_money_scores(
    x,
    float(floating)
)


# =========================================================
# 핵심 결과 카드
# =========================================================

m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "분석기간",
    f"{start_date.strftime('%y.%m.%d')} ~ {end_date.strftime('%y.%m.%d')}"
)


m2.metric(
    "외국인 누적 순매수",
    f"{score['net']:,.0f}주"
)


m3.metric(
    "유동주식 대비 매집률",
    (
        f"{score['rate']:+.2f}%"
        if floating > 0
        else "-"
    )
)


m4.metric(
    "SMART MONEY SCORE",
    f"{score['total']:.0f} / 100"
)


# =========================================================
# 탭
# =========================================================

tabs = st.tabs([

    "📊 요약",

    "📈 주가 · 외국인 수급",

    "📉 누적 순매수",

    "🎯 Score 변화"
])


# =========================================================
# 요약
# =========================================================

with tabs[0]:

    st.subheader(
        f"{stock_name} · Smart Money 요약"
    )


    s1, s2, s3, s4 = st.columns(4)


    s1.metric(
        "누적 매집 규모",
        f"{score['s1']} / 30"
    )


    s2.metric(
        "매집 지속성",
        f"{score['s2']} / 30"
    )


    s3.metric(
        "하락·횡보 매수",
        f"{score['s3']} / 20"
    )


    s4.metric(
        "최근 매집 가속도",
        f"{score['s4']} / 20"
    )


    st.progress(
        min(
            max(
                score["total"] / 100,
                0
            ),
            1
        )
    )


    summary = pd.DataFrame({

        "평가항목": [
            "누적 매집 규모",
            "매집 지속성",
            "하락·횡보 구간 매수",
            "최근 매집 가속도"
        ],

        "점수": [
            score["s1"],
            score["s2"],
            score["s3"],
            score["s4"]
        ],

        "배점": [
            30,
            30,
            20,
            20
        ]
    })


    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )


    st.caption(
        f"매집 지속성 "
        f"{score['positive_ratio'] * 100:.1f}%"
        f" · "
        f"하락·횡보 구간 매수 "
        f"{score['weak_buy_ratio'] * 100:.1f}%"
    )


    st.caption(
        score["acceleration_text"]
    )


# =========================================================
# 주가 · 외국인 수급
# PC 프로그램과 비슷한 구조
# =========================================================

with tabs[1]:

    st.subheader(
        "주가 · 외국인 수급 · 가격대별 외국인 매물대"
    )


    # ---------------------------------------------
    # 가격대별 매물대
    # ---------------------------------------------

    profile = build_price_profile(
        x,
        bins=18
    )


    # ---------------------------------------------
    # 위쪽: 주가 + 오른쪽 매물대
    # 아래쪽: 일별 외국인 순매수
    # ---------------------------------------------

    fig = make_subplots(

        rows=2,
        cols=2,

        column_widths=[
            0.80,
            0.20
        ],

        row_heights=[
            0.72,
            0.28
        ],

        shared_xaxes=False,

        vertical_spacing=0.05,

        horizontal_spacing=0.025,

        specs=[

            [
                {},
                {}
            ],

            [
                {
                    "colspan": 2
                },
                None
            ]
        ]
    )


    # ---------------------------------------------
    # 주가
    # ---------------------------------------------

    price_data = (
        x
        .dropna(subset=["price"])
    )


    fig.add_trace(

        go.Scatter(

            x=price_data["date"],

            y=price_data["price"],

            mode="lines",

            name="주가",

            line=dict(
                width=1.7,
                color="#1683ff"
            ),

            hovertemplate=
                "%{x|%Y-%m-%d}<br>"
                "종가 %{y:,.0f}원"
                "<extra></extra>"
        ),

        row=1,
        col=1
    )


    # ---------------------------------------------
    # 오른쪽 가격대별 매물대
    # 매수 = +
    # 매도 = -
    # ---------------------------------------------

    if not profile.empty:

        fig.add_trace(

            go.Bar(

                x=
                    profile[
                        "buy_foreign"
                    ],

                y=
                    profile[
                        "price_mid"
                    ],

                orientation="h",

                name="외국인 매수(+)",

                marker_color=
                    "rgba(58, 180, 110, 0.55)",

                hovertemplate=
                    "가격대 %{y:,.0f}원<br>"
                    "매수 %{x:,.0f}주"
                    "<extra></extra>"
            ),

            row=1,
            col=2
        )


        fig.add_trace(

            go.Bar(

                x=
                    -profile[
                        "sell_foreign"
                    ],

                y=
                    profile[
                        "price_mid"
                    ],

                orientation="h",

                name="외국인 매도(-)",

                marker_color=
                    "rgba(255, 90, 90, 0.48)",

                hovertemplate=
                    "가격대 %{y:,.0f}원<br>"
                    "매도 %{x:,.0f}주"
                    "<extra></extra>"
            ),

            row=1,
            col=2
        )


    # ---------------------------------------------
    # 아래 외국인 일별 순매수
    # ---------------------------------------------

    buy = x["foreign"].clip(
        lower=0
    )

    sell = x["foreign"].clip(
        upper=0
    )


    fig.add_trace(

        go.Bar(

            x=x["date"],

            y=buy,

            name="일별 순매수(+)",

            marker_color=
                "rgba(40, 130, 255, 0.75)",

            hovertemplate=
                "%{x|%Y-%m-%d}<br>"
                "순매수 +%{y:,.0f}주"
                "<extra></extra>"
        ),

        row=2,
        col=1
    )


    fig.add_trace(

        go.Bar(

            x=x["date"],

            y=sell,

            name="일별 순매도(-)",

            marker_color=
                "rgba(255, 80, 60, 0.75)",

            hovertemplate=
                "%{x|%Y-%m-%d}<br>"
                "순매도 %{y:,.0f}주"
                "<extra></extra>"
        ),

        row=2,
        col=1
    )


    # ---------------------------------------------
    # 레이아웃
    # ---------------------------------------------

    fig.update_layout(

        height=720,

        barmode="relative",

        hovermode="x unified",

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="left",

            x=0
        ),

        margin=dict(
            l=45,
            r=20,
            t=40,
            b=35
        )
    )


    # 주가축

    fig.update_yaxes(

        title_text="주가(원)",

        tickformat=",",

        row=1,
        col=1
    )


    # 매물대 가격축은 주가축과 동일

    if not profile.empty:

        fig.update_yaxes(

            range=[
                float(price_data["price"].min()) * 0.98,
                float(price_data["price"].max()) * 1.02
            ],

            showticklabels=False,

            row=1,
            col=2
        )


        fig.update_xaxes(

            title_text="외국인 누적 수급(주)",

            tickformat=".2s",

            zeroline=True,

            zerolinewidth=1,

            row=1,
            col=2
        )


    # 아래 수급축

    fig.update_yaxes(

        title_text="외국인 일별 순매수(주)",

        tickformat=".2s",

        zeroline=True,

        zerolinewidth=1,

        row=2,
        col=1
    )


    # 위 날짜축은 숨김

    fig.update_xaxes(

        showticklabels=False,

        row=1,
        col=1
    )


    # 아래 날짜축

    fig.update_xaxes(

        type="date",

        tickformat="%y.%m",

        hoverformat="%Y-%m-%d",

        row=2,
        col=1
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True
        }
    )


    # ---------------------------------------------
    # 최근 데이터 카드
    # ---------------------------------------------

    latest_price = (
        price_data["price"].iloc[-1]
        if len(price_data)
        else np.nan
    )


    avg20_price = (
        price_data["price"]
        .tail(20)
        .mean()
        if len(price_data)
        else np.nan
    )


    foreign20 = (
        x["foreign"]
        .tail(20)
        .sum()
    )


    foreign60 = (
        x["foreign"]
        .tail(60)
        .sum()
    )


    r1, r2, r3, r4 = (
        st.columns(4)
    )


    r1.metric(

        "최근 주가",

        (
            f"{latest_price:,.0f}원"
            if pd.notna(latest_price)
            else "-"
        )
    )


    r2.metric(

        "최근 20거래일 평균 주가",

        (
            f"{avg20_price:,.0f}원"
            if pd.notna(avg20_price)
            else "-"
        )
    )


    r3.metric(

        "최근 20거래일 외국인 순매수",

        f"{foreign20:+,.0f}주"
    )


    r4.metric(

        "최근 60거래일 외국인 순매수",

        f"{foreign60:+,.0f}주"
    )


    st.info(
        "오른쪽 가격대별 매물대는 선택한 분석기간 동안 "
        "각 주가 구간에서 발생한 외국인 매수·매도를 집계한 값입니다. "
        "차트는 손가락 또는 마우스로 확대·축소할 수 있습니다."
    )


# =========================================================
# 누적 순매수
# =========================================================

with tabs[2]:

    st.subheader(
        "외국인 누적 순매수 · 주가"
    )


    cumulative = x.copy()


    cumulative["cum_foreign"] = (
        cumulative["foreign"]
        .fillna(0)
        .cumsum()
    )


    if floating > 0:

        cumulative["cum_rate"] = (
            cumulative["cum_foreign"]
            / float(floating)
            * 100
        )

    else:

        cumulative["cum_rate"] = np.nan


    c1, c2 = st.columns(2)


    c1.metric(

        "외국인 누적 순매수",

        f"{cumulative['cum_foreign'].iloc[-1]:+,.0f}주"
    )


    c2.metric(

        "유동주식 대비 누적 매집률",

        (
            f"{cumulative['cum_rate'].iloc[-1]:+.2f}%"
            if floating > 0
            else "-"
        )
    )


    fig2 = make_subplots(

        specs=[
            [{
                "secondary_y":
                    True
            }]
        ]
    )


    if floating > 0:

        fig2.add_trace(

            go.Scatter(

                x=
                    cumulative["date"],

                y=
                    cumulative["cum_rate"],

                mode="lines",

                name=
                    "유동주식 대비 누적 매집률",

                line=dict(
                    color="#e63946",
                    width=2.5
                ),

                hovertemplate=
                    "%{x|%Y-%m-%d}<br>"
                    "누적 매집률 %{y:.2f}%"
                    "<extra></extra>"
            ),

            secondary_y=False
        )

    else:

        fig2.add_trace(

            go.Scatter(

                x=
                    cumulative["date"],

                y=
                    cumulative["cum_foreign"]
                    / 10000,

                mode="lines",

                name=
                    "외국인 누적 순매수",

                line=dict(
                    color="#e63946",
                    width=2.5
                )
            ),

            secondary_y=False
        )


    cp = cumulative.dropna(
        subset=["price"]
    )


    fig2.add_trace(

        go.Scatter(

            x=cp["date"],

            y=cp["price"],

            mode="lines",

            name="주가",

            line=dict(
                color="#1683ff",
                width=1.5
            ),

            opacity=0.65
        ),

        secondary_y=True
    )


    fig2.update_layout(

        height=560,

        hovermode="x unified",

        legend=dict(
            orientation="h"
        ),

        margin=dict(
            l=40,
            r=40,
            t=30,
            b=30
        )
    )


    if floating > 0:

        fig2.update_yaxes(

            title_text=
                "유동주식 대비 누적 매집률(%)",

            ticksuffix="%",

            secondary_y=False
        )

    else:

        fig2.update_yaxes(

            title_text=
                "외국인 누적 순매수(만주)",

            secondary_y=False
        )


    fig2.update_yaxes(

        title_text="주가(원)",

        tickformat=",",

        secondary_y=True
    )


    fig2.update_xaxes(

        type="date",

        tickformat="%y.%m",

        hoverformat="%Y-%m-%d"
    )


    st.plotly_chart(

        fig2,

        use_container_width=True,

        config={
            "displaylogo": False,
            "scrollZoom": True
        }
    )


# =========================================================
# Score 변화
# =========================================================

with tabs[3]:

    st.subheader(
        "Smart Money Score 변화"
    )


    if floating <= 0:

        st.info(
            "유동주식수를 입력하면 Score 변화 추이를 확인할 수 있습니다."
        )

    else:

        with st.spinner(
            "일별 Smart Money Score를 계산하고 있습니다..."
        ):

            history = build_score_history(
                x,
                float(floating)
            )


        if history.empty:

            st.warning(
                "Score 변화 데이터를 만들 수 없습니다."
            )

        else:

            current = float(
                history.iloc[-1]["total"]
            )


            if len(history) > 20:

                previous = float(
                    history.iloc[-21]["total"]
                )

                delta = (
                    current
                    -
                    previous
                )

            else:

                previous = np.nan
                delta = np.nan


            h1, h2, h3 = st.columns(3)


            h1.metric(
                "현재 Score",
                f"{current:.0f}점"
            )


            h2.metric(
                "20거래일 전",
                (
                    f"{previous:.0f}점"
                    if pd.notna(previous)
                    else "-"
                )
            )


            h3.metric(
                "20거래일 변화",
                (
                    f"{delta:+.0f}점"
                    if pd.notna(delta)
                    else "-"
                )
            )


            score_fig = make_subplots(

                specs=[
                    [{
                        "secondary_y":
                            True
                    }]
                ]
            )


            score_fig.add_trace(

                go.Scatter(

                    x=
                        history["date"],

                    y=
                        history["total"],

                    mode="lines",

                    name=
                        "Smart Money Score",

                    line=dict(
                        color="#e63946",
                        width=2.3
                    )
                ),

                secondary_y=False
            )


            score_fig.add_hline(

                y=70,

                line_dash="dash",

                line_color="gray",

                annotation_text="70"
            )


            hp = history.dropna(
                subset=["price"]
            )


            score_fig.add_trace(

                go.Scatter(

                    x=hp["date"],

                    y=hp["price"],

                    mode="lines",

                    name="주가",

                    line=dict(
                        color="#1683ff",
                        width=1.4
                    ),

                    opacity=0.55
                ),

                secondary_y=True
            )


            score_fig.update_yaxes(

                range=[0, 100],

                title_text="Score",

                secondary_y=False
            )


            score_fig.update_yaxes(

                title_text="주가(원)",

                tickformat=",",

                secondary_y=True
            )


            score_fig.update_xaxes(

                type="date",

                tickformat="%y.%m",

                hoverformat="%Y-%m-%d"
            )


            score_fig.update_layout(

                height=560,

                hovermode="x unified",

                legend=dict(
                    orientation="h"
                ),

                margin=dict(
                    l=40,
                    r=40,
                    t=30,
                    b=30
                )
            )


            st.plotly_chart(

                score_fig,

                use_container_width=True,

                config={
                    "displaylogo": False,
                    "scrollZoom": True
                }
            )


            # ---------------------------------------------
            # 구성항목 변화
            # ---------------------------------------------

            component_fig = go.Figure()


            component_fig.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s1"],

                    name="누적 매집 규모 /30",

                    mode="lines"
                )
            )


            component_fig.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s2"],

                    name="매집 지속성 /30",

                    mode="lines"
                )
            )


            component_fig.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s3"],

                    name="하락·횡보 매수 /20",

                    mode="lines"
                )
            )


            component_fig.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s4"],

                    name="최근 매집 가속도 /20",

                    mode="lines"
                )
            )


            component_fig.update_layout(

                title=
                    "Score 구성항목 변화",

                height=440,

                hovermode="x unified",

                legend=dict(
                    orientation="h"
                ),

                margin=dict(
                    l=40,
                    r=20,
                    t=50,
                    b=30
                )
            )


            component_fig.update_xaxes(

                type="date",

                tickformat="%y.%m",

                hoverformat="%Y-%m-%d"
            )


            st.plotly_chart(

                component_fig,

                use_container_width=True,

                config={
                    "displaylogo": False,
                    "scrollZoom": True
                }
            )


# =========================================================
# 하단 안내
# =========================================================

st.caption(
    "※ 업로드 엑셀 Sheet1 기준: "
    "A열=일자 / B열=종가 / H열=외국인 일별 순매수 · "
    "누적 순매수는 선택한 분석기간 시작일부터 다시 계산합니다."
)

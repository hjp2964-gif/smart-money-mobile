# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="Smart Money Mobile",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

[data-testid="stMetricValue"] {
    font-size: 1.35rem;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: .65rem;
        padding-right: .65rem;
    }

    h1 {
        font-size: 1.45rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 누적 매집 규모 점수 / 30
# 유동주식 대비 누적 순매수율 기준
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


# =========================================================
# 비율 점수
# =========================================================

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

    x = data.dropna(subset=["foreign"]).copy()

    net = float(x["foreign"].sum())

    # 1. 누적 매집 규모 /30
    if floating > 0:

        rate = net / floating * 100
        s1 = accumulation_score(rate)

    else:

        rate = 0
        s1 = 0


    # 2. 매집 지속성 /30
    # 장기 매집 여부 판단은 월 단위 유지

    monthly_foreign = (
        x
        .set_index("date")["foreign"]
        .resample("ME")
        .sum()
    )

    positive_ratio = (
        float((monthly_foreign > 0).mean())
        if len(monthly_foreign)
        else 0
    )

    s2 = ratio_score(
        positive_ratio,
        30
    )


    # 3. 하락·횡보 구간 매수 /20
    # 월간 주가 + 외국인 월간 수급 기준

    monthly_price = (
        data
        .dropna(subset=["price"])
        .set_index("date")["price"]
        .resample("ME")
        .agg(["first", "last"])
    )

    monthly_flow = (
        data
        .set_index("date")["foreign"]
        .resample("ME")
        .sum()
    )

    monthly = (
        monthly_price
        .join(
            monthly_flow.rename("foreign"),
            how="inner"
        )
        .dropna()
    )

    monthly["ret"] = (
        monthly["last"]
        / monthly["first"]
        - 1
    )

    weak = monthly[
        monthly["ret"] <= 0.03
    ]

    weak_buy_ratio = (
        float((weak["foreign"] > 0).mean())
        if len(weak)
        else 0
    )

    s3 = ratio_score(
        weak_buy_ratio,
        20
    )


    # 4. 최근 매집 가속도 /20

    flow = x["foreign"].to_numpy(float)

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

        acceleration_ratio = recent / prior

        if acceleration_ratio >= 2:
            s4 = 20
        elif acceleration_ratio >= 1.5:
            s4 = 16
        elif acceleration_ratio >= 1:
            s4 = 12
        elif acceleration_ratio >= 0.5:
            s4 = 8
        else:
            s4 = 4

        acceleration_text = (
            f"최근/직전 60거래일 평균 "
            f"{acceleration_ratio:.2f}배"
        )


    total = s1 + s2 + s3 + s4


    return {

        "net": net,
        "rate": rate,

        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,

        "total": total,

        "positive_ratio": positive_ratio,
        "weak_buy_ratio": weak_buy_ratio,

        "recent": recent,
        "prior": prior,

        "acceleration_text": acceleration_text
    }


# =========================================================
# 일별 Score 변화
# =========================================================

def build_score_history(data, floating):

    d = (
        data
        .dropna(subset=["date", "foreign"])
        .copy()
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
# 공통 일별 X축 설정
# =========================================================

def daily_xaxis(fig, rangeslider=False):

    fig.update_xaxes(

        type="date",

        tickformat="%y.%m.%d",

        hoverformat="%Y-%m-%d",

        showgrid=True,

        rangeslider_visible=rangeslider
    )

    return fig


# =========================================================
# 제목
# =========================================================

st.title(
    "📈 외국인 Smart Money 분석기"
)

st.caption(
    "외국인의 장기 누적 매집과 Smart Money Score를 일별로 확인합니다."
)


# =========================================================
# 파일 업로드
# =========================================================

uploaded = st.file_uploader(
    "분석할 엑셀 파일",
    type=["xlsx", "xls"]
)


if uploaded is None:

    st.info(
        "엑셀 파일을 선택하면 분석을 시작합니다."
    )

    st.stop()


# =========================================================
# 엑셀 읽기
# =========================================================

try:

    raw = pd.read_excel(uploaded)

except Exception as e:

    st.error(
        f"엑셀을 읽지 못했습니다: {e}"
    )

    st.stop()


# =========================================================
# A / V / X 확인
# =========================================================

if raw.shape[1] < 24:

    st.error(
        "엑셀에 필요한 A열, V열, X열이 없습니다."
    )

    st.stop()


# =========================================================
# 분석 데이터
#
# A = 날짜
# V = 주가
# X = 외국인 순매수
# =========================================================

df = pd.DataFrame({

    "date": pd.to_datetime(
        raw.iloc[:, 0],
        errors="coerce"
    ),

    "price": pd.to_numeric(
        raw.iloc[:, 21],
        errors="coerce"
    ),

    "foreign": pd.to_numeric(
        raw.iloc[:, 23],
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
        "A열에서 날짜 데이터를 찾지 못했습니다."
    )

    st.stop()


# =========================================================
# 비정상 날짜 제거
# =========================================================

today_limit = (
    pd.Timestamp.today()
    + pd.Timedelta(days=7)
)


df = df[

    (df["date"] >= pd.Timestamp("2000-01-01"))

    &

    (df["date"] <= today_limit)

].copy()


if df.empty:

    st.error(
        "정상적인 날짜 데이터를 찾지 못했습니다."
    )

    st.stop()


min_d = df["date"].min().date()
max_d = df["date"].max().date()


# =========================================================
# 분석 설정
# =========================================================

with st.expander(
    "⚙️ 분석 설정",
    expanded=True
):

    col1, col2 = st.columns(2)


    with col1:

        stock_name = st.text_input(
            "종목명",
            value=uploaded.name.rsplit(".", 1)[0]
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


    with col2:

        analysis_period = st.date_input(

            "분석기간",

            value=(
                min_d,
                max_d
            ),

            min_value=min_d,
            max_value=max_d
        )


        if listed > 0 and floating > 0:

            float_ratio = (
                floating
                / listed
                * 100
            )

            st.metric(
                "유동주식 비율",
                f"{float_ratio:.2f}%"
            )

        else:

            float_ratio = 0

            st.metric(
                "유동주식 비율",
                "-"
            )


# =========================================================
# 분석기간
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
        "분석기간의 시작일과 종료일을 선택해주세요."
    )

    st.stop()


# =========================================================
# 기간 필터
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
    .dropna(subset=["foreign"])
    .reset_index(drop=True)
)


if x.empty:

    st.error(
        "선택한 분석기간에 외국인 수급 데이터가 없습니다."
    )

    st.stop()


# =========================================================
# 입력값 확인
# =========================================================

if listed > 0 and floating > listed:

    st.warning(
        "유동주식수가 상장주식수보다 많습니다."
    )


if floating <= 0:

    st.warning(
        "유동주식수를 입력하면 외국인 매집률과 Smart Money Score가 계산됩니다."
    )


# =========================================================
# Score 계산
# =========================================================

score = smart_money_scores(
    x,
    float(floating)
)


# =========================================================
# 탭
# =========================================================

tabs = st.tabs([

    "📊 요약",

    "📅 일별 수급",

    "📈 누적 순매수",

    "🎯 Score 변화"
])


# =========================================================
# 요약
# =========================================================

with tabs[0]:

    a, b, c, d = st.columns(4)


    a.metric(
        "Smart Money Score",
        f"{score['total']:.0f} / 100"
    )


    b.metric(
        "외국인 누적 순매수",
        f"{score['net'] / 10000:,.1f}만주"
    )


    c.metric(
        "유동주식 대비 매집률",
        (
            f"{score['rate']:.2f}%"
            if floating > 0
            else "-"
        )
    )


    d.metric(
        "최근 매집 가속도",
        f"{score['s4']:.0f} / 20"
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


    score_df = pd.DataFrame({

        "평가항목": [

            "누적 매집 규모",

            "매집 지속성",

            "하락·횡보 매수",

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
        score_df,
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
# 일별 수급
# =========================================================

with tabs[1]:

    st.subheader(
        "주가 · 외국인 일별 순매수"
    )


    fig = make_subplots(

        specs=[
            [{
                "secondary_y": True
            }]
        ]
    )


    price_data = x.dropna(
        subset=["price"]
    )


    fig.add_trace(

        go.Scatter(

            x=price_data["date"],

            y=price_data["price"],

            name="주가",

            mode="lines",

            line=dict(
                width=2
            )
        ),

        secondary_y=False
    )


    fig.add_trace(

        go.Bar(

            x=x["date"],

            y=x["foreign"] / 10000,

            name="외국인 일별 순매수",

            opacity=0.45
        ),

        secondary_y=True
    )


    fig.update_yaxes(

        title_text="주가(원)",

        secondary_y=False
    )


    fig.update_yaxes(

        title_text="외국인 순매수(만주)",

        secondary_y=True,

        zeroline=True
    )


    fig.update_layout(

        height=540,

        hovermode="x unified",

        legend=dict(
            orientation="h"
        ),

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )


    daily_xaxis(
        fig,
        rangeslider=True
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 누적 순매수
# =========================================================

with tabs[2]:

    cumulative = x.copy()


    cumulative[
        "cum_foreign"
    ] = (

        cumulative[
            "foreign"
        ]
        .cumsum()
    )


    # 유동주식 대비 누적 순매수율

    if floating > 0:

        cumulative[
            "cum_rate"
        ] = (

            cumulative[
                "cum_foreign"
            ]

            /

            float(floating)

            *

            100
        )

    else:

        cumulative[
            "cum_rate"
        ] = np.nan


    # 상단 핵심 숫자

    m1, m2 = st.columns(2)


    m1.metric(

        "외국인 누적 순매수",

        f"{cumulative['cum_foreign'].iloc[-1] / 10000:,.1f}만주"
    )


    m2.metric(

        "유동주식 대비 누적 매집률",

        (
            f"{cumulative['cum_rate'].iloc[-1]:.2f}%"
            if floating > 0
            else "-"
        )
    )


    if floating <= 0:

        st.info(
            "유동주식수를 입력하면 누적 순매수율 차트를 볼 수 있습니다."
        )

    else:

        fig = make_subplots(

            specs=[
                [{
                    "secondary_y": True
                }]
            ]
        )


        # 핵심 변경
        # 누적 주식 수 대신 유동주식 대비 % 사용

        fig.add_trace(

            go.Scatter(

                x=cumulative["date"],

                y=cumulative["cum_rate"],

                name="유동주식 대비 누적 매집률",

                mode="lines",

                line=dict(
                    color="red",
                    width=2.5
                ),

                hovertemplate=
                    "%{x|%Y-%m-%d}<br>"
                    "누적 매집률 %{y:.2f}%"
                    "<extra></extra>"
            ),

            secondary_y=False
        )


        price_data = cumulative.dropna(
            subset=["price"]
        )


        fig.add_trace(

            go.Scatter(

                x=price_data["date"],

                y=price_data["price"],

                name="주가",

                mode="lines",

                opacity=.55
            ),

            secondary_y=True
        )


        fig.update_yaxes(

            title_text=
                "유동주식 대비 누적 매집률(%)",

            ticksuffix="%",

            secondary_y=False,

            zeroline=True
        )


        fig.update_yaxes(

            title_text="주가(원)",

            secondary_y=True
        )


        fig.update_layout(

            height=540,

            hovermode="x unified",

            legend=dict(
                orientation="h"
            ),

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )
        )


        daily_xaxis(
            fig,
            rangeslider=True
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# Score 변화
# =========================================================

with tabs[3]:

    if floating <= 0:

        st.info(
            "유동주식수를 입력하면 Score 변화 추이를 볼 수 있습니다."
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

                previous_label = (
                    f"{previous:.0f}점"
                )

                delta_label = (
                    f"{delta:+.0f}점"
                )

            else:

                previous_label = "데이터 부족"
                delta_label = "-"


            s1, s2, s3 = st.columns(3)


            s1.metric(
                "현재 Score",
                f"{current:.0f}점"
            )


            s2.metric(
                "20거래일 전",
                previous_label
            )


            s3.metric(
                "20거래일 변화",
                delta_label
            )


            # ---------------------------------------------
            # 일별 Score + 주가
            # ---------------------------------------------

            fig = make_subplots(

                specs=[
                    [{
                        "secondary_y": True
                    }]
                ]
            )


            fig.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["total"],

                    name="Smart Money Score",

                    mode="lines",

                    line=dict(
                        width=2.5
                    )
                ),

                secondary_y=False
            )


            fig.add_hline(

                y=70,

                line_dash="dash",

                annotation_text="70점"
            )


            price_history = history.dropna(
                subset=["price"]
            )


            fig.add_trace(

                go.Scatter(

                    x=price_history["date"],

                    y=price_history["price"],

                    name="주가",

                    mode="lines",

                    opacity=.45
                ),

                secondary_y=True
            )


            fig.update_yaxes(

                range=[0, 100],

                title_text="Smart Money Score",

                secondary_y=False
            )


            fig.update_yaxes(

                title_text="주가(원)",

                secondary_y=True
            )


            fig.update_layout(

                title=
                    "일별 Smart Money Score · 주가",

                height=540,

                hovermode="x unified",

                legend=dict(
                    orientation="h"
                ),

                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                )
            )


            daily_xaxis(
                fig,
                rangeslider=True
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ---------------------------------------------
            # Score 세부 항목 변화
            # ---------------------------------------------

            fig2 = go.Figure()


            fig2.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s1"],

                    name="누적 매집 규모 /30",

                    mode="lines"
                )
            )


            fig2.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s2"],

                    name="매집 지속성 /30",

                    mode="lines"
                )
            )


            fig2.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s3"],

                    name="하락·횡보 매수 /20",

                    mode="lines"
                )
            )


            fig2.add_trace(

                go.Scatter(

                    x=history["date"],

                    y=history["s4"],

                    name="최근 매집 가속도 /20",

                    mode="lines"
                )
            )


            fig2.update_layout(

                title="Score 구성항목 일별 변화",

                height=460,

                hovermode="x unified",

                legend=dict(
                    orientation="h"
                ),

                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                )
            )


            daily_xaxis(
                fig2,
                rangeslider=True
            )


            st.plotly_chart(
                fig2,
                use_container_width=True
            )


# =========================================================
# 하단
# =========================================================

st.caption(
    "※ A열=날짜 / V열=주가 / X열=외국인 순매수 기준 · "
    "누적 매집률은 입력한 유동주식수를 기준으로 계산합니다."
)

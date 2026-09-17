# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Smart Money Mobile", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:2rem; max-width:1200px}
[data-testid="stMetricValue"] {font-size:1.45rem}
@media (max-width: 640px) {
  .block-container {padding-left:.75rem; padding-right:.75rem}
  h1 {font-size:1.55rem !important}
}
</style>
""", unsafe_allow_html=True)

def ratio_score(ratio,max_score):
    if ratio < .40:return 0
    if ratio < .50:return round(max_score/6)
    if ratio < .60:return round(max_score*2/6)
    if ratio < .70:return round(max_score*3/6)
    if ratio < .80:return round(max_score*4/6)
    if ratio < .90:return round(max_score*5/6)
    return max_score

def smart_money_scores(data,floating):
    x=data.dropna(subset=["foreign"]).copy()
    net=float(x.foreign.sum()); rate=net/floating*100; s1=accumulation_score(rate)
    m=x.set_index("date").foreign.resample("ME").sum()
    pr=float((m>0).mean()) if len(m) else 0; s2=ratio_score(pr,30)
    px=data.dropna(subset=["price"]).set_index("date").price.resample("ME").agg(["first","last"])
    mf=data.set_index("date").foreign.resample("ME").sum()
    mo=px.join(mf.rename("foreign"),how="inner").dropna()
    mo["ret"]=mo["last"]/mo["first"]-1
    weak=mo[mo.ret<=.03]
    wr=float((weak.foreign>0).mean()) if len(weak) else 0; s3=ratio_score(wr,20)
    f=x.foreign.to_numpy(float)
    recent=float(np.mean(f[-60:])) if len(f) else 0
    prior=float(np.mean(f[-120:-60])) if len(f)>=120 else (float(np.mean(f[:-60])) if len(f)>60 else 0)
    if recent<=0:s4=0; at="최근 순매수 평균 ≤ 0"
    elif prior<=0:s4=20; at="직전 구간 순매도/중립 → 최근 순매수"
    else:
        rr=recent/prior
        s4=20 if rr>=2 else 16 if rr>=1.5 else 12 if rr>=1 else 8 if rr>=.5 else 4
        at=f"최근/직전 60일 평균 = {rr:.2f}배"
    return dict(net=net,rate=rate,s1=s1,s2=s2,s3=s3,s4=s4,total=s1+s2+s3+s4,
                pr=pr,wr=wr,weakmonths=len(weak),recent=recent,prior=prior,at=at)

def nice_step(prices):
    span=float(prices.max()-prices.min())
    target=max(span/14,1)
    bases=[1,2,2.5,5,10]
    p=10**np.floor(np.log10(target))
    for b in bases:
        if b*p>=target:return b*p
    return 10*p

def price_profile(data):
    x=data.dropna(subset=["price","foreign"]).copy()
    step=nice_step(x.price)
    lo=np.floor(x.price.min()/step)*step
    hi=np.ceil(x.price.max()/step)*step+step
    bins=np.arange(lo,hi+step,step)
    x["bucket"]=pd.cut(x.price,bins=bins,right=False,include_lowest=True)
    g=x.groupby("bucket",observed=True).agg(
        buy=("foreign",lambda s:s[s>0].sum()),
        sell=("foreign",lambda s:-s[s<0].sum()),
        net=("foreign","sum"),
        days=("foreign","size")
    ).reset_index()
    g["mid"]=g.bucket.apply(lambda z:(z.left+z.right)/2).astype(float)
    return g,step

def build_score_history(x, floating):
    d=x.dropna(subset=["date","foreign"]).copy().reset_index(drop=True)
    rows=[]
    for i in range(len(d)):
        z=smart_money_scores(d.iloc[:i+1], floating)
        rows.append({"date":d.loc[i,"date"],"price":d.loc[i,"price"],"total":z["total"],
                     "s1":z["s1"],"s2":z["s2"],"s3":z["s3"],"s4":z["s4"]})
    return pd.DataFrame(rows)

def prepare(uploaded, start_date, end_date):
    df=pd.read_excel(uploaded)
    x=df[(df.date>=pd.Timestamp(start_date)) & (df.date<=pd.Timestamp(end_date))].copy()
    x=x.sort_values("date").dropna(subset=["foreign"])
    if x.empty:
        raise ValueError("선택한 기간에 외국인 수급 데이터가 없습니다.")
    return df,x

st.title("📈 외국인 Smart Money 분석기 · 모바일")
st.caption("PC v10의 핵심 분석을 스마트폰 브라우저에서 사용하도록 바꾼 웹앱 버전입니다.")

uploaded=st.file_uploader("분석할 엑셀 파일", type=["xlsx","xls"])
if uploaded is None:
    st.info("엑셀 파일을 선택하면 분석 화면이 열립니다.")
    st.stop()

try:
    df=pd.read_excel(uploaded)
except Exception as e:
    st.error(f"엑셀을 읽지 못했습니다: {e}")
    st.stop()

valid=df.dropna(subset=["date"]).sort_values("date")
min_d=valid.date.min().date()
max_d=valid.date.max().date()

with st.expander("분석 설정", expanded=True):
    c1,c2=st.columns(2)
    with c1:
        stock_name=st.text_input("종목명", value=uploaded.name.rsplit(".",1)[0])
        listed=st.number_input("상장주식수", min_value=0, value=0, step=1000000)
        float_ratio=st.number_input("유동주식 비율(%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    with c2:
        start_date=st.date_input("분석 시작일", value=min_d, min_value=min_d, max_value=max_d)
        end_date=st.date_input("분석 기준일", value=max_d, min_value=min_d, max_value=max_d)

floating=float(listed)*(float(float_ratio)/100) if listed else 0.0
if not floating:
    st.warning("상장주식수를 입력해야 유동주식 대비 매집률과 Smart Money Score가 정확히 계산됩니다.")

try:
    _,x=prepare(uploaded,start_date,end_date)
    s=smart_money_scores(x,floating)
except Exception as e:
    st.error(f"분석 중 오류가 발생했습니다: {e}")
    st.stop()

tabs=st.tabs(["요약","일별 수급","누적 순매수","Score 변화"])

with tabs[0]:
    cols=st.columns(4)
    cols[0].metric("Smart Money Score",f"{s['total']:.0f} / 100")
    cols[1].metric("외국인 누적 순매수",f"{s['net']:,.0f}주")
    cols[2].metric("유동주식 대비",f"{s['rate']*100:.2f}%" if floating else "-")
    cols[3].metric("최근 매집 가속도",f"{s['s4']:.0f} / 20")
    st.progress(min(max(float(s["total"])/100,0),1))
    score_df=pd.DataFrame({
        "평가항목":["누적 매집 규모","매집 지속성","하락·횡보 매수","최근 매집 가속도"],
        "점수":[s["s1"],s["s2"],s["s3"],s["s4"]],
        "배점":[30,30,20,20]
    })
    st.dataframe(score_df,hide_index=True,use_container_width=True)
    if s["total"]>=70:
        st.info("현재 Score가 70점 이상입니다. 점수 변화 방향과 주가 위치를 함께 확인해 보세요.")

with tabs[1]:
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    p=x.dropna(subset=["price"])
    fig.add_trace(go.Scatter(x=p.date,y=p.price,name="주가",mode="lines"),secondary_y=False)
    fig.add_trace(go.Bar(x=x.date,y=x.foreign,name="외국인 일별 순매수",opacity=.5),secondary_y=True)
    fig.update_layout(title=f"{stock_name} · 주가와 외국인 일별 순매수",height=520,hovermode="x unified",
                      legend=dict(orientation="h"))
    fig.update_yaxes(title_text="주가(원)",secondary_y=False)
    fig.update_yaxes(title_text="외국인 순매수(주)",secondary_y=True)
    st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    c=x.copy()
    c["cum_foreign"]=c.foreign.cumsum()
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=c.date,y=c.cum_foreign,name="외국인 누적 순매수",mode="lines",
                             line=dict(color="red",width=2.5)),secondary_y=False)
    p=c.dropna(subset=["price"])
    fig.add_trace(go.Scatter(x=p.date,y=p.price,name="주가",mode="lines",opacity=.45),secondary_y=True)
    fig.update_layout(title=f"{stock_name} · 외국인 누적 순매수와 주가",height=520,hovermode="x unified",
                      legend=dict(orientation="h"))
    fig.update_yaxes(title_text="누적 순매수(주)",secondary_y=False)
    fig.update_yaxes(title_text="주가(원)",secondary_y=True)
    st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    with st.spinner("날짜별 Smart Money Score를 다시 계산하고 있습니다..."):
        hist=build_score_history(x,floating)
    current=float(hist.iloc[-1].total)
    if len(hist)>20:
        prev=float(hist.iloc[-21].total)
        delta=current-prev
        prev_label=f"{prev:.0f}점"
        delta_label=f"{delta:+.0f}점"
    else:
        prev_label="데이터 부족"; delta_label="-"
    a,b,c=st.columns(3)
    a.metric("현재 Score",f"{current:.0f}점")
    b.metric("20거래일 전",prev_label)
    c.metric("20거래일 변화",delta_label)

    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=hist.date,y=hist.total,name="Smart Money Score",mode="lines"),secondary_y=False)
    fig.add_hline(y=70,line_dash="dash",annotation_text="70점 기준")
    pp=hist.dropna(subset=["price"])
    fig.add_trace(go.Scatter(x=pp.date,y=pp.price,name="주가",mode="lines",opacity=.4),secondary_y=True)
    fig.update_yaxes(range=[0,100],title_text="Score",secondary_y=False)
    fig.update_yaxes(title_text="주가(원)",secondary_y=True)
    fig.update_layout(title="Smart Money Score 변화 추이",height=520,hovermode="x unified",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig,use_container_width=True)

    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=hist.date,y=hist.s1,name="누적 매집 규모 /30",mode="lines"))
    fig2.add_trace(go.Scatter(x=hist.date,y=hist.s2,name="매집 지속성 /30",mode="lines"))
    fig2.add_trace(go.Scatter(x=hist.date,y=hist.s3,name="하락·횡보 매수 /20",mode="lines"))
    fig2.add_trace(go.Scatter(x=hist.date,y=hist.s4,name="최근 매집 가속도 /20",mode="lines"))
    fig2.update_layout(title="Score 구성항목 변화",height=430,hovermode="x unified",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig2,use_container_width=True)

st.caption("※ 이 버전은 엑셀 업로드 기반입니다. 스마트폰에서는 파일 앱/다운로드 폴더의 xlsx 파일을 선택해 분석할 수 있습니다.")

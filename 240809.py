import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

# CSV 파일에서 데이터를 불러오기 (오프라인 데이터)
@st.cache_data
def load_offline_data():
    try:
        df = pd.read_csv('240809off.csv')  # 오프라인 파일
        return df
    except FileNotFoundError:
        st.error("240809off.csv 파일을 찾을 수 없습니다.")
        return None
# 긍정 키워드 리스트
positive_keywords = [
    "만족", "훌륭", "배려", "기분", "세심", "프로페셔널", "도움", "최고", "좋다", "좋았", "청결",
    "쾌적", "깔끔", "정돈", "깨끗", "넓다", "안락", "편안", "환영", "유쾌", "행복", "기쁘다", "정성",
    "편리", "완벽", "신선", "즐겁다", "저렴", "추천", "구성", "효율적","다음에도","ㅎㅎ","감사"]

# 부정 키워드 리스트
negative_keywords = [
    "불만", "느리다", "대기", "오래", "혼잡", "지저분", "불편", "부족", "문제", "싫다", "나쁘다", "늦다", 
    "미흡", "아쉽다", "불친절", "지루", "비싸다", "고장", "어렵다", "복잡", "바쁘다", "모자라다", "거칠다","못함"
    "어수선", "좁다", "복잡하다", "힘들다", "불쾌", "답답", "엉망", "구성부족", "실망", "ㅠ", "귀찮", "불", "최악", "비싸요", "안됨", "아쉬","없음"
]

# 중립 키워드 리스트
neutral_keywords = [
    "보통", "무난", "괜찮다", "평범", "중립", "기대", "그럭저럭", "적당", "그저", "아무렇지", "괜찮", 
    "보통", "알맞다", "중간", "평균", "일반", "무난", "기본", "차별"
]

# 불용어 리스트 (감성 분석에 큰 의미 없는 단어들)
stopwords = [
    "너무", "정말", "진짜", "그냥", "매우", "아주", "조금", "약간", "좀", "다소", "또한", "대체로", 
    "때문에", "이", "저", "그", "그리고", "하지만", "그래서", "또", "보다", "더", "그런", "같은", 
    "사실", "이건", "그건", "저건", "정도", "한", "이런", "저런", "그런", "게다가", "결국", "결과적으로", "많", "합니다", "은ㅋ"
]

# 감성 분류 함수
def classify_sentiment(comment):
    comment = comment.lower()  # 소문자로 변환하여 처리
    # 불용어 제거
    for word in stopwords:
        comment = comment.replace(word, "")
    # 감성 분류
    if any(word in comment for word in positive_keywords):
        return "긍정"
    elif any(word in comment for word in negative_keywords):
        return "부정"
    else:
        return "중립"

# CSV 파일에서 데이터를 불러오기 (온라인 데이터)
@st.cache_data
def load_online_data():
    try:
        df = pd.read_csv('240809on.csv')  # 온라인 파일
        return df
    except FileNotFoundError:
        st.error("240809on.csv 파일을 찾을 수 없습니다.")
        return None

# 데이터 필터링 함수 (월별 데이터 선택)
def filter_data_by_month(df, month):
    return df[df["시작일시"].str.contains(month)]

# 오프라인 점포별 종합만족도 계산 함수
def calculate_scores_by_store(df):
    average_scores = df.groupby("점포")[["직원 서비스", "정보 제공", "상품 준비", "신속 결제", "매장 환경"]].mean()
    average_scores_100 = (average_scores * 100 / 7).round(0).astype(int)
    average_scores_100["종합만족도"] = average_scores_100.mean(axis=1).round(0).astype(int)
    return average_scores_100

# 온라인 종합만족도 계산 함수
def calculate_online_scores(df):
    average_scores = df[["로그인 접속", "상품 검색", "상품 준비", "상품 결제", "앱 사용성"]].mean()
    average_scores_100 = (average_scores * 100 / 7).round(0).astype(int)
    average_scores_100["종합만족도"] = average_scores_100.mean().round(0).astype(int)
    return average_scores_100

# 점포별 재이용의향률 계산 함수
def calculate_revisit_intention_rate(df):
    # '재이용의향률' 칼럼에서 "예" 응답 비율을 계산
    df_yes = df[df["재이용의향률"] == "예."]
    
    # 점포별 "예" 응답 비율 계산
    revisit_rate = df_yes.groupby("점포").size() / df.groupby("점포").size() * 100
    
    # Fill NaN values with 0 and convert to integers
    revisit_rate = revisit_rate.fillna(0).round(0).astype(int)
    
    return revisit_rate

    
# 오프라인 데이터 불러오기
df_offline = load_offline_data()

# 온라인 데이터 불러오기
df_online = load_online_data()

# Streamlit 앱
st.title("9월 고객만족도📊")

# 탭 생성
tab1, tab2, tab3 = st.tabs(['요약', '명동점', '인천공항점'])

with tab1:  
    if df_offline is not None and df_online is not None:
        # 8월과 9월 데이터 필터링 (오프라인)
        df_offline_august = filter_data_by_month(df_offline, "2024-08")
        df_offline_september = filter_data_by_month(df_offline, "2024-09")

        # 8월과 9월 데이터 필터링 (온라인)
        df_online_august = filter_data_by_month(df_online, "2024-08")
        df_online_september = filter_data_by_month(df_online, "2024-09")

        # 8월과 9월 오프라인 종합만족도 계산
        average_scores_august_offline = calculate_scores_by_store(df_offline_august)["종합만족도"]
        average_scores_september_offline = calculate_scores_by_store(df_offline_september)["종합만족도"]

        # 8월과 9월 온라인 종합만족도 계산
        online_scores_august = calculate_online_scores(df_online_august)["종합만족도"]
        online_scores_september = calculate_online_scores(df_online_september)["종합만족도"]

        # 가로 정렬로 모든 점포 및 온라인 정보를 한 줄에 표시
        st.text("")
        st.markdown("#### 종합만족도 점수")
        st.markdown("""종합만족도는 다섯 가지 서비스 요소; <span style="background-color: #ffffe0;">직원 서비스, 정보 제공, 상품 준비, 신속 결제, 매장 환경의 점수를 평균</span> 낸 값으로, 신세계면세점의 전반적인 만족도를 측정할 수 있습니다. 
""", unsafe_allow_html=True)

        # 모든 점포의 종합만족도 점수를 가로로 정렬하여 한 줄에 표시
        cols = st.columns(4)  # 4개의 칼럼 생성 (명동점, 인천공항점, 부산점, 온라인)

        cols[0].metric(label="명동점", value=f"{average_scores_september_offline.get('명동점', 'N/A')}점", 
                    delta=f"{average_scores_september_offline.get('명동점', 0) - average_scores_august_offline.get('명동점', 0)}점")
        
        cols[1].metric(label="인천공항점", value=f"{average_scores_september_offline.get('인천공항점', 'N/A')}점", 
                    delta=f"{average_scores_september_offline.get('인천공항점', 0) - average_scores_august_offline.get('인천공항점', 0)}점")
        
        cols[2].metric(label="부산점", value=f"{average_scores_september_offline.get('부산점', 'N/A')}점", 
                    delta=f"{average_scores_september_offline.get('부산점', 0) - average_scores_august_offline.get('부산점', 0)}점")
        
        cols[3].metric(label="온라인", value=f"{online_scores_september}점", 
                    delta=f"{online_scores_september - online_scores_august}점")

        st.write("")
        st.write("")
        st.divider()
        st.markdown("##### 1.온오프라인 채널별 점수 추이")
        st.markdown ("*24년 3월 고객만족도 문항이 변경되며 점수 변동")
        # 데이터프레임 생성
        data = data = {
                "기간": ["9월", "10월", "11월", "12월", "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "24년 9월"],
                "오프라인": [94, 94, 92, 94, 94, 94, 92, 86, 87, 87, 87, 88, 89],
                "온라인": [84, 86, 86, 86, 88, 86, 88, 86, 84, 84, 84, 85, 82]
            }

        # 데이터프레임 생성
        df_trend = pd.DataFrame(data)

        # 종합만족도 그래프 생성
        fig = go.Figure()

        # 오프라인 데이터 추가 (빨간색)
        fig.add_trace(go.Scatter(x=df_trend["기간"], y=df_trend["오프라인"], mode='lines+markers+text', name='오프라인',
                                line=dict(color='#EA385B'), marker=dict(color='#EA385B', size=6),
                                text=df_trend["오프라인"], textposition='top center'))

        # 온라인 데이터 추가 (파란색)
        fig.add_trace(go.Scatter(x=df_trend["기간"], y=df_trend["온라인"], mode='lines+markers+text', name='온라인',
                                line=dict(color='#6F80F2'), marker=dict(color='#6F80F2', size=6),
                                text=df_trend["온라인"], textposition='top center'))

        # 그래프 레이아웃 설정
        fig.update_layout(
            yaxis=dict(range=[80, 100]),  # Y축 최소값 80, 최대값 100
            legend=dict( 
                orientation="h",  # 범례를 수평으로 설정
                x=0.5,  # 범례의 x축 위치 (0이 왼쪽, 1이 오른쪽)
                y=1.15,  # 범례의 y축 위치 (그래프 위에 배치)
                xanchor='center',  # 중앙 정렬
                yanchor='bottom'   # 아래쪽 정렬
            ),
            hovermode="x unified",
            template="plotly_white"
        )

        # Streamlit에 Plotly 그래프 출력
        st.plotly_chart(fig)

    st.divider()

    # 오각형 그래프 (레이다 차트) 생성 함수
    def create_radar_chart(average_scores, title="항목별 종합만족도 (9월)"):
        categories = ["직원 서비스", "정보 제공", "상품 준비", "신속 결제", "매장 환경"]
        fig = go.Figure()

        # 명동점 데이터 추가 (빨간색 #EA385B)
        fig.add_trace(go.Scatterpolar(
            r=average_scores.loc['명동점', categories].values,
            theta=categories,
            fill='toself',
            name='명동점',
            line=dict(color='#EA385B')  # 빨간색
        ))

        # 인천공항점 데이터 추가 (파란색 #6F80F2)
        fig.add_trace(go.Scatterpolar(
            r=average_scores.loc['인천공항점', categories].values,
            theta=categories,
            fill='toself',
            name='인천공항점',
            line=dict(color='#6F80F2')  # 파란색
        ))

        # 부산점 데이터 추가 (초록색 #3BC14A)
        fig.add_trace(go.Scatterpolar(
            r=average_scores.loc['부산점', categories].values,
            theta=categories,
            fill='toself',
            name='부산점',
            line=dict(color='#3BC14A')  # 초록색
        ))

        # 그래프 레이아웃 설정
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[60, 100]  # Y축 범위 설정 (60 ~ 100)
                )),
            showlegend=True,
            title=title
        )

        return fig

    # 레이더 차트 함수 호출 및 데이터 유효성 확인
    with tab1:  
        if df_offline is not None:
            # 필터링된 9월 데이터로 평균 점수 계산
            df_offline_september = filter_data_by_month(df_offline, "2024-09")
            average_scores_september = calculate_scores_by_store(df_offline_september)

            # 9월 데이터가 비어있지 않으면 레이더 차트 생성
            if not average_scores_september.empty:
                st.markdown("##### 2.오프라인 항목별 점수")
                radar_chart = create_radar_chart(average_scores_september, title="9월 점포별 종합만족도")
                st.plotly_chart(radar_chart)
            else:
                st.error("9월 점포별 평균 점수 데이터가 비어 있습니다.")

        st.divider()

    # 데이터 필터링 함수 (월별 데이터 선택)
    def filter_data_by_month(df, month):
        return df[df["시작일시"].str.contains(month)]  # '시작일시' 컬럼에 month가 포함된 데이터 필터링

    # 100점 만점으로 점수 변환 함수
    def convert_to_100_scale(df, columns):
        # 각 항목의 점수를 7점 만점에서 100점 만점으로 변환
        return (df[columns].mean() * 100 / 7).round(0).astype(int)

   
        # 9월 온라인 항목별 종합만족도 점수 세로형 바 차트 생성 함수
    def create_vertical_bar_chart(average_scores, title="9월 항목별 종합만족도"):
        categories = ["로그인 접속", "상품 검색", "상품 준비", "상품 결제", "앱 사용성"]
        fig = go.Figure()

        # 9월 온라인 항목별 종합만족도 점수 세로형 바 차트 생성 함수
    def create_vertical_bar_chart(average_scores, title="9월 항목별 종합만족도"):
        categories = ["로그인 접속", "상품 검색", "상품 준비", "상품 결제", "앱 사용성"]
        fig = go.Figure()

        # 세로형 바 차트 추가
        fig.add_trace(go.Bar(
            x=categories,  # 항목 이름 (x축)
            y=average_scores.values,  # 점수 값 (y축)
            marker=dict(color='#EA385B'),  # 막대 색상 설정
            text=average_scores.values,  # 각 항목 점수를 텍스트로 표시
            textposition='outside',  # 텍스트 표시 위치
            width=0.4  # 막대 너비 설정
        ))

        # 그래프 레이아웃 설정
        fig.update_layout(
            title=title,
            xaxis_title='항목',
            yaxis_title='점수',
            yaxis=dict(range=[75, 95]),  # Y축 범위 75~100
            bargap=0.2,  # 막대 간격 설정
            template="plotly_white"  # 흰색 배경 템플릿
        )

        return fig

    # 세로형 바 차트를 Streamlit 앱에 표시하기
    if df_online is not None:
        # 9월 온라인 데이터 필터링 및 평균 점수 계산
        df_online_september = filter_data_by_month(df_online, "2024-09")

        # 100점 만점으로 평균 점수 변환
        columns = ["로그인 접속", "상품 검색", "상품 준비", "상품 결제", "앱 사용성"]
        average_scores_online_september = convert_to_100_scale(df_online_september, columns)

        # 9월 온라인 항목별 종합만족도 점수 세로형 바 차트 생성
        st.markdown("##### 3. 온라인 항목별 점수")
        vertical_bar_chart = create_vertical_bar_chart(average_scores_online_september, title="9월 항목별 종합만족도")
        st.plotly_chart(vertical_bar_chart)
    else:
        st.error("온라인 데이터를 불러올 수 없습니다.")

with tab2:
    # 8월과 9월 데이터 필터링 (명동점)
    store_august = filter_data_by_month(df_offline, "2024-08")  # 8월 데이터 필터링
    store_september = filter_data_by_month(df_offline, "2024-09")  # 9월 데이터 필터링

    # 명동점 데이터 필터링
    store_august_myeongdong = store_august[store_august["점포"] == "명동점"]
    store_september_myeongdong = store_september[store_september["점포"] == "명동점"]

    # 8월 종합만족도 및 재이용의향률 계산
    august_satisfaction = calculate_scores_by_store(store_august_myeongdong)["종합만족도"]["명동점"]
    august_revisit_rate = calculate_revisit_intention_rate(store_august_myeongdong)["명동점"]

    # 9월 종합만족도 및 재이용의향률 계산
    september_satisfaction = calculate_scores_by_store(store_september_myeongdong)["종합만족도"]["명동점"]
    september_revisit_rate = calculate_revisit_intention_rate(store_september_myeongdong)["명동점"]

    # Delta (차이 계산)
    satisfaction_delta = september_satisfaction - august_satisfaction
    revisit_rate_delta = september_revisit_rate - august_revisit_rate

    # 명동점 8월과 9월 종합만족도 및 재이용의향률을 요약탭처럼 표시
    st.write("")
    st.markdown("### 명동점 종합만족도 및 재이용의향률")
    
    cols = st.columns(2)  # 두 개의 칼럼 생성
    
    # 종합만족도 표시
    cols[0].metric(label="종합만족도", value=f"{september_satisfaction}점", 
                   delta=f"{satisfaction_delta}점")
    
    # 재이용의향률 표시
    cols[1].metric(label="재이용의향률", value=f"{september_revisit_rate}%", 
                   delta=f"{revisit_rate_delta}%")
    # 8월, 9월 데이터 필터링
    df_offline_august = filter_data_by_month(df_offline, "2024-08")
    df_offline_september = filter_data_by_month(df_offline, "2024-09")

    # 8월, 9월 종합만족도 계산
    august_scores = calculate_scores_by_store(df_offline_august)
    september_scores = calculate_scores_by_store(df_offline_september)

    # 비교할 항목들 선택
    categories = ["직원 서비스", "정보 제공", "상품 준비", "신속 결제", "매장 환경"]

    # Plotly 그래프 준비
    fig = go.Figure()

    # 8월 데이터 (투명도 50%)
    fig.add_trace(go.Bar(
        x=august_scores.loc[:, categories].mean(),
        y=categories,
        name="08월",
        marker_color='rgba(234, 56, 91, 0.5)',  # 투명도 50%
        orientation='h'
    ))

    # 9월 데이터 (레이블 추가)
    fig.add_trace(go.Bar(
        x=september_scores.loc[:, categories].mean(),
        y=categories,
        name="09월",
        marker_color='#EA385B',
        orientation='h',
        text=september_scores.loc[:, categories].mean().round(1),  # 레이블 텍스트 (소수점 1자리)
        textposition='auto'  # 레이블 자동 배치
    ))

    # 레이아웃 설정
    fig.update_layout(
        barmode='group',
        title='8월과 9월 종합만족도 항목 비교',
        xaxis_title='만족도 점수 (100점 만점)',
        yaxis_title='항목',
        yaxis=dict(categoryorder='total ascending')  # 항목 정렬
    )

    # 스트림릿으로 출력
    st.plotly_chart(fig)
    # 구분선
    st.divider()
    # 명동점의 9월 추가 의견 데이터 필터링
    comments_september = df_offline_september[df_offline_september["점포"] == "명동점"]["추가 의견"].dropna()

    if not comments_september.empty:
        # 감정 분석 수행 (9월 데이터만)
        comments_df_september = pd.DataFrame(comments_september)
        comments_df_september["감정 분류"] = comments_df_september["추가 의견"].apply(classify_sentiment)

        # 감정 분류 비율 계산
        sentiment_counts = comments_df_september["감정 분류"].value_counts()

        # ============================
        # 감정분석 비율 원형 차트
        # ============================
        # 원형 차트 생성 (긍정: 초록, 부정: 빨강, 중립: 회색)
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,  # 도넛 형태
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(colors=["#D3D3D3", "#EA385B", "#A9DFBF"])  # 긍정, 부정, 중립의 색상 지정
        )])
        # 원형 차트 레이아웃 설정
        st.markdown("##### 1.명동점 추가 의견 감정 분류")
        # 원형 차트 표시
        st.plotly_chart(fig_pie)

        # ============================
        # 감정분석 실제 내용
        # ============================
        st.dataframe(comments_df_september, use_container_width=True)
        
        # ============================
        # 워드클라우드
        # ============================
        # 불용어를 제거한 추가 의견 데이터를 하나의 문자열로 결합
        def remove_stopwords(text):
            for word in stopwords:
                text = text.replace(word, "")
            return text

        comments_september = comments_september.apply(remove_stopwords)
        comment_text = " ".join(comments_september)

        # 워드클라우드 생성 (한글 폰트 지정)
        font_path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 경로 (윈도우 맑은고딕)
        wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(comment_text)

        # 워드클라우드를 matplotlib으로 시각화
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")  # 축 숨기기
        st.write("")
        st.markdown("##### 2.명동점 추가 의견 워드클라우드")
        # 워드클라우드를 Streamlit에 표시
        st.pyplot(fig)

    # 구분선
    st.divider()


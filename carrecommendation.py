import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
df = pd.read_csv("co2.csv")  # 같은 폴더에 있는 co2.csv 파일 불러오기
df.columns = df.columns.str.strip()

# 타겟 컬럼 자동 탐색
target_col = [col for col in df.columns if "CO2" in col][0]

# -----------------------------
# 2. 데이터 전처리
# -----------------------------
X = df.drop(['Make', 'Model', target_col], axis=1)
y = df[target_col]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. 모델 학습
# -----------------------------
lr = LinearRegression().fit(X_train_scaled, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("🚗 차량 제원 기반 CO₂ 배출량 예측 인공지능")
st.write("차량 제원을 입력하면 CO₂ 배출량을 예측하고, 친환경 차량을 추천합니다.")

# 사용자 입력
engine_size = st.number_input("엔진 크기 (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
cylinders = st.number_input("실린더 수", min_value=3, max_value=16, value=4, step=1)
fuel_consumption = st.number_input("복합 연비 (L/100km)", min_value=2.0, max_value=30.0, value=7.5, step=0.1)
fuel_type = st.selectbox("Fuel Type (연료 종류)", sorted(df["Fuel Type"].unique()))
vehicle_class = st.selectbox("Vehicle Class (차량 클래스)", sorted(df["Vehicle Class"].unique()))

if st.button("예측하기"):
    # 입력값을 데이터프레임으로 변환
    input_df = pd.DataFrame([{
        "Engine Size (L)": engine_size,
        "Cylinders": cylinders,
        "Fuel Consumption Comb (L/100 km)": fuel_consumption,
        "Fuel Type": fuel_type,
        "Vehicle Class": vehicle_class
    }])
    input_df = pd.get_dummies(input_df, drop_first=True)

    # 학습 데이터와 동일한 컬럼 맞추기
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    # 예측
    lr_pred = lr.predict(scaler.transform(input_df))[0]
    rf_pred = rf.predict(input_df)[0]

    st.success(f"Linear Regression 예측: {lr_pred:.1f} g/km")
    st.success(f"Random Forest 예측: {rf_pred:.1f} g/km (추천 모델)")

    # 친환경 차량 추천 (자동차 이름 + CO₂ 배출량 표시)
    filtered = df[(df["Vehicle Class"] == vehicle_class) & (df["Fuel Type"] == fuel_type)]
    if not filtered.empty:
        best_car = filtered.loc[filtered[target_col].idxmin()]
        st.info(f"{vehicle_class} 클래스, {fuel_type} 차량 중 가장 친환경적인 모델은 "
                f"{best_car['Make']} {best_car['Model']}이며, "
                f"CO₂ 배출량은 {best_car[target_col]} g/km 입니다.")

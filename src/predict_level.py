import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 설정 상수
DB_CONFIG = "mysql+pymysql://musthave:tiger@10.125.121.184:3306/pms_db_dev_gs"
SAFETY_MIN = 2.0
SAFETY_MAX = 6.4  # 만수위(6.4m)를 고려한 상향 설정
PUMP_POWER_KW = 150  # 펌프 1대당 가상 소비전력 (현장 사정에 맞춰 수정 가능)

# 펌프 대수별 메인 관로 압력 (잔류 압력 9.07 포함)
PUMP_PRESSURE = {0: 9.07, 1: 9.50, 2: 9.94, 3: 10.07, 4: 10.10}

def get_load_info_winter(dt):
    """겨울철 전력 부하 시간대 및 단가 (원/kWh 가상 설정)"""
    hour = dt.hour
    if 23 <= hour or hour < 9: return "LOW", 70.0    # 경부하
    elif (10 <= hour < 12) or (17 <= hour < 20) or (22 <= hour < 23): return "HIGH", 200.0 # 최대부하
    else: return "MID", 130.0                        # 중간부하

def get_extended_dataset(days=3):
    engine = create_engine(DB_CONFIG)
    # 최근 데이터 중 3일치 추출
    query = f"""
    SELECT 
        r.collected_at AS timestamp, r.flow_out AS outflow_rate, 
        r.level AS water_level, t.press_pipe AS press_out
    FROM reservoir_minutely r
    LEFT JOIN treatment_minutely t ON r.collected_at = t.collected_at AND t.facility_id = 1
    WHERE r.collected_at BETWEEN '2023-01-10' AND '2023-01-12'
    AND r.facility_id = 4 ORDER BY r.collected_at ASC
    """
    df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['outflow_m3_min'] = df['outflow_rate'] / 60
    df['delta_h'] = df['water_level'].diff()
    return df.dropna()

def train_physics_model(df):
    feature_cols = ['press_out', 'outflow_m3_min']
    mask = (df['delta_h'].abs() > 0.0001)
    model = LinearRegression(fit_intercept=True)
    model.fit(df[mask][feature_cols], df[mask]['delta_h'])
    return model

def run_final_optimization_sim(df, model, start_level):
    results = []
    current_level = start_level
    active_pumps = 2

    for i in range(len(df)):
        row = df.iloc[i]
        load_type, price = get_load_info_winter(row['timestamp'])
        
        # --- 선제적 제어 로직 (10분 단위) ---
        if i % 10 == 0:
            if load_type == "LOW": # 경부하: 풀가동 채우기
                active_pumps = 4 if current_level < 4.0 else 2
            elif load_type == "HIGH": # 최대부하: 정지 버티기
                active_pumps = 0 if current_level > (SAFETY_MIN + 0.3) else 2
            else: # 중간부하: 밸런스 유지
                active_pumps = 2 if 3.5 <= current_level <= 5.0 else (3 if current_level < 3.5 else 1)

        # 긴급 안전 장치
        if current_level >= SAFETY_MAX: active_pumps = 0
        if current_level <= SAFETY_MIN: active_pumps = 4

        # 수위 예측
        target_press = PUMP_PRESSURE[active_pumps]
        input_data = pd.DataFrame([[target_press, row['outflow_m3_min']]], columns=['press_out', 'outflow_m3_min'])
        current_level += model.predict(input_data)[0]
        
        # 비용 계산 (kW * 시간(1/60h) * 단가)
        sim_cost = (active_pumps * PUMP_POWER_KW * (1/60)) * price
        
        # 실제 운영 비용 추정 (당시 관로 압력을 펌프 대수로 환산하여 계산)
        # PUMP_PRESSURE 기준: {0: 9.07, 1: 9.50, 2: 9.94, 3: 10.07, 4: 10.10}
        # 실제 압력 데이터를 기반으로 0~4대를 정확히 분류
        p_val = row['press_out']
        if p_val < 9.3: actual_pumps = 0      # 정지 구간 (약 9.07)
        elif p_val < 9.7: actual_pumps = 1    # 1대 구간 (약 9.50)
        elif p_val < 10.0: actual_pumps = 2   # 2대 구간 (약 9.94)
        elif p_val < 10.08: actual_pumps = 3  # 3대 구간 (약 10.07)
        else: actual_pumps = 4                # 4대 구간 (약 10.10 이상)
        
        actual_cost = (actual_pumps * PUMP_POWER_KW * (1/60)) * price

        results.append({
            'timestamp': row['timestamp'], 'load_type': load_type,
            'active_pumps': active_pumps, 'sim_level': current_level,
            'actual_level': row['water_level'], 'sim_cost': sim_cost, 'actual_cost': actual_cost
        })
    return pd.DataFrame(results)

# --- 실행 및 시각화 ---
if __name__ == "__main__":
    df_all = get_extended_dataset(days=3)
    model = train_physics_model(df_all)
    sim_res = run_final_optimization_sim(df_all, model, df_all.iloc[0]['water_level'])

    # 1. 시각화 (3일치)
    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax1.plot(sim_res['timestamp'], sim_res['sim_level'], label='Optimized Sim Level (6.0m)', color='#0072B2', linewidth=2)
    ax1.plot(sim_res['timestamp'], sim_res['actual_level'], label='Actual Level', color='gray', alpha=0.4, linestyle='--')
    ax1.set_ylabel('Water Level (m)', fontsize=12)
    ax1.set_ylim(1.5, 6.5)
    
    ax2 = ax1.twinx()
    ax2.step(sim_res['timestamp'], sim_res['active_pumps'], where='post', color='orange', alpha=0.6, label='Sim Pump Count')
    ax2.set_ylabel('Pump Count (N)', color='orange')

    # 배경색 처리
    for i, row in sim_res.iterrows():
        if row['load_type'] == "HIGH":
            ax1.axvspan(row['timestamp'], row['timestamp'] + pd.Timedelta(minutes=1), color='red', alpha=0.03)
        elif row['load_type'] == "LOW":
            ax1.axvspan(row['timestamp'], row['timestamp'] + pd.Timedelta(minutes=1), color='green', alpha=0.03)

    plt.title("3-Day Energy Optimization Simulation & Cost Analysis", fontsize=16)
    ax1.legend(loc='upper left'); plt.grid(True, alpha=0.2); plt.show()

    # 2. 요금 절감 결과 출력
    total_actual = sim_res['actual_cost'].sum()
    total_sim = sim_res['sim_cost'].sum()
    savings = total_actual - total_sim
    savings_pct = (savings / total_actual) * 100

    print(f"\n📊 [3일간의 에너지 절감 리포트]")
    print(f" - 기존 방식 추정 요금: {total_actual:,.0f} 원")
    print(f" - 최적화 방식 추정 요금: {total_sim:,.0f} 원")
    print(f" - 예상 절감액: {savings:,.0f} 원 (절감률: {savings_pct:.2f}%)")
    print(f" - 경부하 시간 평균 수위: {sim_res[sim_res['load_type']=='LOW']['sim_level'].mean():.2f} m")
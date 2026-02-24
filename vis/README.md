# 배수지 스마트 운영 관제 (시각화)

> 배수지 수요 예측 및 펌프 최적화 결과를 Gradio + Seaborn/Matplotlib로 시각화하는 웹 대시보드입니다.

## 설치해야 하는 도구

> gradio

```bash
pip install gradio
```

## 파일별 역할

- app.py: 메인 진입점, Gradio UI 구성, 수요 예측/펌프 최적화 탭, Seaborn/Matplotlib 차트, KPI 카드 생성
- demo_loader.py: 모델 및 설정 로드, `model/` 디렉터리의 pkl/pth로 수요 예측 추론, 샘플/실제용 데이터 생성, 배수지 목록 및 설정 JSON 읽기

## 사용법

```bash
# vis 디렉터리에서
cd vis
python app.py
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

## 시각화 항목

- 수요 예측
  - 배수지 선택: 예측할 배수지를 선택합니다.
  - 기준 시간: 예측 기준 시각을 입력합니다.
  - 샘플 생성: DB/모델 없이 합성 샘플로 차트와 JSON을 봅니다.
  - 실제 예측: `model/`의 pkl·pth와 데모 입력으로 추론한 결과를 차트와 JSON으로 봅니다.

- 펌프 최적화
  - 기준 시간: 시뮬레이션 기준 시각을 입력합니다.
  - 샘플 로드 / 실제 최적화: 펌프 가동 대수, 배수지별 수위, 비용, Spill량을 4분할 차트와 KPI로 확인합니다(현재는 DB 없이 샘플 데이터로 표시).

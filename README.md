# TeamFirst | MLOps 프로젝트

#### TMDB 영화 데이터를 기반으로 영화 평점을 예측하고, 모델 학습부터 웹 서비스 시연 및 자동화까지 포함한 MLOps 파이프라인을 구현한 프로젝트입니다.
---

## 1) 프로젝트 개요

- Task: TMDB 영화 메타데이터 기반 영화 평점 예측 (Regression)
- Model: Numpy 기반 MLP 회귀 모델 (MoviePredictor)
- 주요 구성 요소:
    - TMDB 영화 데이터를 활용한 평점 예측 모델
    - 학습 / 평가 / 추론 모듈 분리 구조
    - Streamlit 기반 웹 데모 서비스
    - Docker 기반 컨테이너 실행
    - GitHub Actions 기반 CI 자동화
---

## 2) 폴더 구조
```
repo/
 ├─ .github/                                                # GitHub 협업 및 자동화 설정
 │   ├─ ISSUE_TEMPLATE/                                     # 이슈 템플릿 모음
 │   │   ├─ bug_report.md                                   # 버그 제보용 이슈 템플릿
 │   │   ├─ feature_request.md                              # 기능 추가/개선 요청 템플릿
 │   │   ├─ experiment.md                                   # 모델 실험 기록 템플릿
 │   │   ├─ data_request.md                                 # 데이터 요청 템플릿
 │   │   ├─ refactor.md                                     # 코드 리팩토링 요청 템플릿
 │   │   └─ task.md                                         # 일반 작업 등록 템플릿
 │   ├─ workflows/                                          # GitHub Actions 자동화 파이프라인
 │   │   ├─ ci.yml                                          # 테스트 및 코드 검증 CI 파이프라인
 │   │   ├─ docker.yml                                      # Docker 이미지 빌드 및 레지스트리 업로드
 │   │   └─ run-container.yml                               # Docker 컨테이너 실행 테스트
 │   └─ pull_request_template.md                            # Pull Request 기본 템플릿
 │
 ├─ dataset/                                                # 모델 학습 및 데모 데이터
 │   ├─ movies.csv                                          # Streamlit 데모용 영화 데이터
 │   └─ tmdb_rating.csv                                     # TMDB 영화 평점 예측 학습 데이터
 │
 ├─ models/                                                 # 학습된 모델 저장 폴더
 │   └─ movie_predictor/
 │       └─ E10_T260304180143.pkl                           # 학습된 모델 체크포인트 파일
 │
 ├─ src/                                                    # 프로젝트 핵심 백엔드 코드
 │   ├─ main.py                                             # 모델 학습 및 추론 실행 진입점
 │   ├─ deploy.py                                           # Prefect 기반 모델 재학습 스케줄 배포
 │   ├─ webapp.py                                           # FastAPI 기반 모델 예측 API 서버
 │   ├─ requirements.txt                                    # 실행 환경
 │   │
 │   ├─ dataset/                                            # 데이터 로딩 및 전처리 모듈
 │   │   ├─ data_loader.py                                  # 간단한 배치 DataLoader 구현
 │   │   ├─ tmdb_dataset.py                                 # TMDB 데이터셋 로딩 및 전처리
 │   │   └─ watch_log.py                                    # 시청 로그 기반 데이터 로딩 (이전 실험 구조)
 │   │
 │   ├─ train/                                              # 모델 학습 로직
 │   │   └─ train.py                                        # 배치 단위 학습 및 손실 계산
 │   │
 │   ├─ evaluate/                                           # 모델 성능 평가
 │   │   └─ evaluate.py                                     # 검증 및 테스트 RMSE 계산
 │   │
 │   ├─ inference/                                          # 모델 추론 로직
 │   │   └─ inference.py                                    # 저장된 모델 로드 및 예측 수행
 │   │
 │   ├─ model/                                              # 모델 구조 정의
 │   │   ├─ __pycache__
 │   │   │   └─ movie_predictor.cpython-311.pyc             # movie_predictor.py의 컴파일 캐시
 │   │   └─ movie_predictor.py                              # 영화 평점 예측 모델 구현 (MLP 구조)
 │   │
 │   ├─ postprocess/                                        # 예측 결과 후처리 및 저장
 │   │   └─ postprocess.py                                  # 예측 결과 DB 저장 및 조회
 │   │
 │   └─ utils/                                              # 공통 유틸리티 함수
 │       ├─ factory.py                                      # 모델 객체 생성 팩토리
 │       ├─ utils.py                                        # 경로 설정 및 공통 함수
 │       └─ __pycache__
 │           └─ utils.cpython-311.pyc                       # utils.py의 컴파일 캐시
 │
 ├─ streamlit/                                              # Streamlit 기반 웹 데모 서비스
 │   ├─ streamlit_app.py                                    # 영화 평점 예측 UI 웹 애플리케이션
 │   └─ requirements.txt                                    # Streamlit 실행 환경
 │
 ├─ tests/                                                  # 테스트 코드
 │   └─ test_basic.py                                       # 기본 테스트 코드 (CI 검증용)
 │
 ├─ .gitignore                                              # Git 제외 규칙
 ├─ Dockerfile                                              # API 서버 실행용 Docker 이미지 설정
 ├─ README.md                                               # 프로젝트 설명 문서
 └─ start_api_server.sh                                     # FastAPI 서버 실행 스크립트


```


## 3) Setup

### Python

권장 환경
Python 3.11

### Install

pip install -r requirements.txt


---


## 4) Run

### Model Training

python src/main.py train --model_name movie_predictor


학습된 모델은 다음 경로에 저장됩니다.
models/movie_predictor/


### Model Inference

python src/main.py inference


### Streamlit Demo

streamlit run streamlit/streamlit_app.py


---


## 5) Demo

Streamlit 기반 웹 인터페이스를 통해

- 영화 메타데이터 입력
- 모델 기반 평점 예측
- 영화 카드 UI 확인

등의 기능을 확인할 수 있습니다.


---

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/x_ji_VNX)

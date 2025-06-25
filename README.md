# 딸기 인스턴스 세그멘테이션 마나 프로젝트

## 프로젝트 개요
본 프로젝트는 실내 수직농장 딸기 재배 시스템에서 수집된 이미지를 이용하여 딸기 과실을 인스턴스 세그멘테이션하는 YOLOv11-seg 모델을 개발하였습니다. 또한, 이 프로젝트에서는 이미지에 다양한 색상 보정를 적용하여 모델별 성능을 비교해았습니다.
이 프로젝트는 세종대학교 대학원 강의 '생명과학을 위한 머신러닝 응용 및 실습'의 미니 프로젝트로 수행되었습니다.

## 프로젝트 발표자료
https://drive.google.com/file/d/1wk7RftwzFULRCNfI4zrwCRAnKh7s7MBE/view?usp=drive_link

## 프로젝트 구조
```
code/
├── color_calibration.py     # 알고리즘 기반 색상 보정 코드
├── colorchecker.ipynb       #  ColorChecker 기반 색상 보정 코드
├── data_analysis.ipynb      # 데이터 분석 관련 코드
└── train.py                 # YOLOv11-seg 학습 코드
requirements.txt             # 모델 학습에 사용된 라이브러리 버전
```

## 프로젝트 구성

### 데이터셋
- 서울 남부터미널 실내 수직농장 시스템에서 재배되는 딸기를 드론으로 촬영
  -드론: DJI NEO
  - 해상도: 4000x2250 pixels
  - 일주일 간격으로 3주간 촬영: 이미지 수: 114/143/143장
- CVAT로 딸기 과실 어노테이션
  - 딸기 과실이 없는 이미지는 삭제 후 사용: 총 297장 사용

### 이미지 전처리
- 원본 이미지에 4가지 색상 보정 적용:
  1. Gray world: 
     - 이미지의 RGB 채널의 평균을 같게 맞춰 색 균형 조정
     - 각 채널을 동일한 평균값으로 맞춰 색감이 한쪽으로 치우진 이미지를 중성 회색톤에 가깝게 조정
     - 특정 색상이 지배적인 이미지의 색 왜곡을 줄임
  2. Simple white balance
     - OpenCV 내장 알고리즘으로 이미지의 색온도를 자동 보정
     - 이미지에서 '중립회색(회색 또는 흰색)'을 찾아 이를 기준으로 RGB 채널을 스케일링하여 색온도 보정
  3. HSV boosting
    - 이미지의 채도(S)와 명도(V) 채널에 히스토그램 평활화를 적용해 색감과 밝기 분포를 조절
    - 이미지가 더 선명하고 밝게 보이며, 색이 풍부해짐
  4. ColorChecker
    - 24패치를 가진 Classic mini 사용
    - 많은 연구에서 공식적으로 이미지 보정을 위해 사용하는 색상 보정 카드
    - 색상 카드를 대상과 같이 촬영하여 D65(자연광 환경) 기준 값과 비교 후 선형 보정
    - D65 공식 색상표 참조: https://xritephoto.com/documents/literature/en/ColorData-1p_EN.pdf

### 모델 개발
1. YOLOv11n-seg 모델
   - 기본 파라미터 적용
   - 원본 이미지 사용
2. YOLOv11n-seg 모델 튜닝
   - 하이퍼파라미터 튜닝
     - Optimizer: SGD, AdamW
     - Epoch: 100, 1000
     - Images size: 640, 768, 896, 1024
     - Learning rate(lr0): 0.01, 0.001
3. 색 보정 이미지 적용
   - 1주차 이미지만 사용
   - 모델: YOLOv11n-seg
   - Original/Gray world/Simple white balance/ColorChecker 비교

### 데이터 분석
1. 주차별 성능평가
   - 주차별 데이터가 모델에 미친 영향을 확인
   - 데이터를 1, 2, 3주차로 구분하여 성능평가
   - 각 주차별 RGB 평균 히스토그램 시각화로 분포 확인

2. 객체 크기 비교
   - 주차별 차이가 객체 수, 크기에 영향을 받았는지 확인
   - 각 주차별 이미지 수, 객체 수, 총 면적, 평균 면적 계산
  
 ### 결론

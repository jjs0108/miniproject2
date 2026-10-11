# 기계학습 미니프로젝트 발표 파일

제공한 `기계학습 미니프로젝트.pdf`에 필요한 데이터와 모델 단독 실행 파일만 정리했습니다. 기존 파일은 수정하지 않았습니다.

## 파일과 발표 순서

- `data/train.csv`, `data/test.csv`: PDF 5~6쪽 데이터
- `data/K-NCT_v1.4.json`: PDF 7쪽 데이터
- `note/00_data_overview.ipynb`: 세 데이터의 규모·원본 10행·태그 제거 예시
- `note/01_klue_bert.ipynb`: PDF 8~12쪽, BERT 구조·토큰화·빈칸 후보
- `note/02_kogrammar.ipynb`: PDF 13~20쪽, 교정 모델 구조·토큰화·생성·성공/실패 사례. 마지막에는 PDF 21쪽의 [MASK] 보존 문제 확인

노트북을 위 순서로 열면 됩니다. 실행 결과도 저장되어 있습니다. 현재 `/opt/miniconda3/bin/python`의 base 커널에서 검증했습니다. 새 환경은 `pip install -r requirements.txt`로 준비합니다.

원본 두 모델을 조사하는 자료입니다. 추가 학습 코드·가중치·실패한 학습 기록·PTB 영어 데이터는 포함하지 않았습니다. 교정 → 복원 통합 개선은 발표의 계획이며 이 폴더에 완성된 통합 모델이 있다고 설명하지 않습니다.

모델 가중치는 대용량이라 복사하지 않았습니다. 같은 workspace에서는 기존 캐시를 재사용하고, 이 폴더만 별도 복사하면 처음 모델 실행 시 Hugging Face에서 `note/.model_cache`로 다운로드합니다. 데이터 확인 노트북은 모델 캐시 없이 실행됩니다.

데이터 출처는 PDF에 기재된 아래 링크를 기록했습니다. 원격 파일과 로컬 파일의 동일성 및 이용 조건을 새로 검증한 것은 아닙니다.

- https://huggingface.co/datasets/gayom/KoreaSpellingCorrection/tree/main
- https://github.com/seonminkoo/K-NCT
- https://huggingface.co/klue/bert-base
- https://huggingface.co/theSOL1/kogrammar-base

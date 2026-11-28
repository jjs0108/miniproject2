# 빠진 단어 복원 (MASK 없이)

기존 오픈소스 `klue/bert-base`는 빠진 자리에 `[MASK]`가 있어야만 단어를 복원할 수 있다.
실제 문장에는 `[MASK]` 표시가 없으므로, 빠진 위치를 스스로 찾아서 복원하도록 개선한다.

## 방법

1. 단어 사이의 모든 자리에 `[MASK]`를 하나씩 넣어 본다
2. 두 글자 이상 한글 단어가 가장 자신 있게 들어가는 자리 5개를 고른다 (1단계)
3. 그 5개 자리에 단어를 채워 보고, 완성된 문장이 가장 자연스러운 자리를 고른다 (2단계)
   - 자연스러움 점수: 토큰을 하나씩 가리고 원래 토큰을 맞힐 확률의 합
4. 고른 자리에 `[MASK]` 1개 또는 2개를 넣어 단어를 채운다

따로 학습하지 않고 기존 모델만 사용한다.

## 테스트 데이터

KoreaSpellingCorrection `data/train.csv`의 정답 문장에서 일반명사로만 된 어절 하나를 지웠다 (100문장).

## 결과

| 항목 | 정확도 |
|---|---|
| 무작위로 위치 고르기 | 9% |
| 빠진 위치 찾기 (1단계만) | 30% |
| 빠진 위치 찾기 (2단계) | 49% |
| 위치 1칸 차이 이내 | 65% |
| 위치와 단어 모두 맞힘 | 16% |
| 참고: 위치를 알려 줬을 때 단어 맞히기 (기존 방식) | 22% |

## 실행

```bash
pip install kiwipiepy torch "transformers==4.46.3" pandas
cd word_restore
python restore_nomask.py ../data/train.csv results/restore_nomask_100.csv 100
```

## 파일

- `make_blank.py`: `[MASK]` 단어 복원 데이터 만들기
- `restore_nomask.py`: `[MASK]` 없이 위치 찾기와 복원 테스트
- `results/restore_nomask_100.csv`: 2단계 결과 100문장
- `results/restore_nomask_300_1단계.csv`: 1단계 결과 300문장
- `results/examples.txt`: 성공·실패 예시

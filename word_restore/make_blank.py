# 맞춤법 데이터의 정답 문장에서 일반명사 하나를 [MASK]로 바꿔 단어 복원 데이터를 만든다
import pandas as pd, random
from kiwipiepy import Kiwi

random.seed(42)
kiwi = Kiwi()
df = pd.read_csv("../data/train.csv").dropna()

def make_blank(sentence):
    # 일반명사(NNG)이고 두 글자 이상인 단어만 후보로
    cands = [t for t in kiwi.tokenize(sentence) if t.tag == "NNG" and len(t.form) >= 2]
    if not cands:
        return None, None
    t = random.choice(cands)
    return sentence[:t.start] + "[MASK]" + sentence[t.start + t.len:], t.form

df[["question", "answer"]] = df["correct"].apply(lambda s: pd.Series(make_blank(s)))
df = df.dropna(subset=["question"])
df[["question", "answer"]].to_csv("word_restore.csv", index=False, encoding="utf-8-sig")
print(len(df))

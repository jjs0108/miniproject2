# 단어가 빠진 문장(MASK 없음)에서 빠진 위치를 찾고 단어를 복원하는 테스트
import sys, random, pandas as pd, torch
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForMaskedLM

data_path, out_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
random.seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
kiwi = Kiwi()
tok = AutoTokenizer.from_pretrained("klue/bert-base")
m = AutoModelForMaskedLM.from_pretrained("klue/bert-base").to(dev).eval()

# 1. 테스트 문장 만들기: 일반명사로만 된 어절 하나를 통째로 지움
df = pd.read_csv(data_path).dropna().sample(3000, random_state=0)
samples = []
for s in df["correct"]:
    words = s.split()
    if not (5 <= len(words) <= 25):
        continue
    cands = []
    for i, w in enumerate(words):
        t = kiwi.tokenize(w)
        if len(t) == 1 and t[0].tag == "NNG" and len(w) >= 2:
            cands.append(i)
    if not cands:
        continue
    i = random.choice(cands)
    samples.append(dict(words=words[:i] + words[i+1:], pos=i, ans=words[i], orig=s))
    if len(samples) >= N:
        break

def mask_logits(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        lg = m(**enc).logits
    out = []
    for b in range(len(texts)):
        idx = (enc.input_ids[b] == tok.mask_token_id).nonzero()[:, 0]
        out.append(lg[b, idx].log_softmax(-1))
    return out

def fill(words, gap, n_mask):
    # gap 위치에 MASK n개를 넣고 확신 높은 자리부터 채움. (단어, 평균 log 확률) 반환
    text = " ".join(words[:gap] + [tok.mask_token * n_mask] + words[gap:])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
    ids = enc.input_ids.clone()
    pos = (ids[0] == tok.mask_token_id).nonzero()[:, 0].tolist()
    total = 0.0
    todo = list(pos)
    while todo:
        with torch.no_grad():
            lp = m(input_ids=ids, attention_mask=enc.attention_mask).logits[0, todo].log_softmax(-1)
        conf, arg = lp.max(-1)
        b = conf.argmax().item()
        ids[0, todo[b]] = arg[b]
        total += conf[b].item()
        todo.pop(b)
    word = tok.decode(ids[0, pos]).replace(" ", "")
    return word, total / n_mask

def pll(text):
    # 문장 자연스러움 점수: 토큰을 하나씩 가리고 원래 토큰을 맞힐 log 확률의 합
    ids = tok(text, return_tensors="pt", truncation=True, max_length=128).input_ids[0].to(dev)
    L = len(ids) - 2
    batch = ids.repeat(L, 1)
    ar = torch.arange(L, device=dev)
    batch[ar, ar + 1] = tok.mask_token_id
    with torch.no_grad():
        lp = m(input_ids=batch).logits.log_softmax(-1)
    return lp[ar, ar + 1, ids[1:-1]].sum().item()

res = []
for k, x in enumerate(samples):
    words = x["words"]
    gaps = list(range(len(words) + 1))
    texts = [" ".join(words[:g] + [tok.mask_token] + words[g:]) for g in gaps]
    lps = mask_logits(texts)
    # 위치 점수: 그 자리에 들어갈 '두 글자 이상 한글 단어' 후보의 최고 확률
    scores = []
    for lp in lps:
        top = lp[0].topk(20)
        best = -99.0
        for v, j in zip(top.values.tolist(), top.indices.tolist()):
            t = tok.convert_ids_to_tokens(j)
            if not t.startswith("##") and len(t) >= 2 and all("가" <= c <= "힣" for c in t):
                best = v
                break
        scores.append(best)
    first_gap = max(gaps, key=lambda g: scores[g])
    # 2단계: 점수 상위 5개 자리에 단어를 채워 보고, 완성된 문장 전체가 가장 자연스러운 자리를 고름
    top_gaps = sorted(gaps, key=lambda g: -scores[g])[:5]
    best = None
    for g in top_gaps:
        w = max([fill(words, g, n) for n in (1, 2)], key=lambda c: c[1])[0]
        sc = pll(" ".join(words[:g] + [w] + words[g:]))
        if best is None or sc > best[0]:
            best = (sc, g, w)
    _, pred_gap, pred_word = best
    # 위치를 알려 줬을 때(기존 방식) 단어 맞히기
    oracle_word = max([fill(words, x["pos"], n) for n in (1, 2)], key=lambda c: c[1])[0]
    res.append(dict(
        input=" ".join(words), answer=x["ans"], true_gap=x["pos"], pred_gap=pred_gap, n_gaps=len(gaps),
        pos_ok=pred_gap == x["pos"], first_ok=first_gap == x["pos"], pred_word=pred_word,
        full_ok=pred_gap == x["pos"] and pred_word == x["ans"],
        oracle_word=oracle_word, oracle_ok=oracle_word == x["ans"],
        restored=" ".join(words[:pred_gap] + [pred_word] + words[pred_gap:]), original=x["orig"]))
    if k % 5 == 0:
        print(k, len(samples), flush=True, file=sys.stderr)

r = pd.DataFrame(res)
r.to_csv(out_path, index=False, encoding="utf-8-sig")
print("문장 수", len(r))
print("무작위로 위치 고를 때 기대 정확도", round((1 / r.n_gaps).mean(), 3))
print("빠진 위치 찾기 정확도 (1단계만)", round(r.first_ok.mean(), 3))
print("빠진 위치 찾기 정확도 (2단계)", round(r.pos_ok.mean(), 3))
print("위치 1칸 차이 이내", round(((r.pred_gap - r.true_gap).abs() <= 1).mean(), 3))
print("위치+단어 모두 맞힘", round(r.full_ok.mean(), 3))
print("참고: 위치를 알려 줬을 때 단어 정확도", round(r.oracle_ok.mean(), 3))



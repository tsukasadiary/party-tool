
import random as rng

HANDS = ["✊", "✌", "✋"]
RESULTS = ["あいこ", "勝ち！", "負け..."]

def janken(my_hand):
    op_hand = rng.randint(0,2)
    mh, oh= HANDS[my_hand], HANDS[op_hand]
    res = RESULTS[(op_hand - my_hand + 3) % 3]
    display(f"自分：{mh} vs. PyScript君：{oh}", target="janken-result", append=False)
    display(f"結果：{res}", target="janken-result")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rng
from pyscript import Element
    

##### Utils #####
def load_text_box(tag, splitter=None):
    elem = Element(tag)
    if splitter is None: return elem.element.value.split()
    else: return elem.element.value.split(splitter)
    

def dump_error(message):
    elem = Element("result")
    elem.write(message)
    
    
##### output #####
def dump_assignment(member_names, group_names, assignment):
    idx = np.sort(np.where(assignment == -1)[0])
    G = [member_names[i] for i in idx]
    display("- 休憩: " + ", ".join(G), target="result", append=False)
    for k, gname in enumerate(group_names):
        idx = np.sort(np.where(assignment == k)[0])
        G = [member_names[i] for i in idx]
        display("- " + gname + ": " + ", ".join(G), target="result")


def dump_pair_matrix(A=None):
    if A is None: A = np.zeros((10,10), dtype=int)
    n = A.shape[0]
    plt.close()
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(
        A, annot=True, cbar=False,
        linewidth=0.5, square=True, ax=ax
    )
    display(fig, target="stats", append=False)


##### Status #####
def dump_status(count, member_names, group_names, group_sizes, latest_break, A, B):
    n = len(member_names)
    status = "count:" + str(count) + ";\n"
    status += "member-names:" + ",".join(member_names) + ";\n"
    status += "group-names:" + ",".join(group_names) + ";\n"
    status += "group-sizes:" + ",".join(map(str, group_sizes)) + ";\n"
    status += "latest-break:" + ",".join(map(str, latest_break)) + ";\n"
    pairs = [(i,j,A[i,j]) for i in range(n) for j in range(i+1,n)]
    status += "pairs:" + "|".join(map(str, pairs)) + ";\n"
    break_pairs = [(i,j,B[i,j]) for i in range(n) for j in range(i,n)]
    status += "break-pairs:" + "|".join(map(str, break_pairs)) + ";\n"
    elem = Element("status")
    elem.write(status)
    

def pairs_to_matrix(n, pairs):
    A = np.zeros((n, n), dtype=int)
    for p in pairs:
        val = list(map(int, p[1:-1].split(",")))
        i, j, a = val[0], val[1], val[2]
        A[i,j] = a
        A[j,i] = a
    return A


def load_status():
    status = load_text_box("status", splitter=";\n")
    status_dict = dict()
    for line in status:
        val = line.split(":")
        if len(val) > 1: status_dict[val[0]] = val[1]
        
    count = int(status_dict["count"])
    member_names = status_dict["member-names"].split(",")
    group_names = status_dict["group-names"].split(",")
    group_sizes = list(map(int, status_dict["group-sizes"].split(",")))
    latest_break = np.array(list(map(int, status_dict["latest-break"].split(","))))
    pairs = status_dict["pairs"].split("|")
    break_pairs = status_dict["break-pairs"].split("|")
    
    n = len(member_names)
    A = pairs_to_matrix(n, pairs)
    B = pairs_to_matrix(n, break_pairs)
        
    return count, member_names, group_names, group_sizes, latest_break, A, B


##### Main Engine #####
def next_break(n, group_sizes, latest_break, B):
    b = n - np.sum(group_sizes)
    idx_all = list(range(n))
    
    dmin, bmin = np.min(latest_break), np.min(B + np.diag([np.inf] * n))
    count = np.diag(B).copy()
    
    idx_best, best_val = None, None
    for _ in range(100):
        # 休憩回数の差が2以上になるペアが出てはいけない
        while True:
            idx = rng.sample(idx_all, b)
            count[idx] += 1
            vmin, vmax = np.min(count), np.max(count)
            count[idx] -= 1
            if vmax - vmin <= 1: break
            
        # 直近に休憩しているほどペナルティ
        val = np.sum((latest_break[idx] - dmin) ** 2)
        
        # 同時に休憩していることが多いほどペナルティ
        val += np.sum((B[np.ix_(idx,idx)] - bmin) ** 2)
        val -= np.sum((count[idx] - bmin) ** 2)
        
        if best_val is None:
            idx_best = idx.copy()
            best_val = val
        elif best_val > val:
            idx_best = idx.copy()
            best_val = val
            
    return idx_best


def next_assignment(n, group_sizes, idx_break):
    """割り当てをランダム生成"""
    idx = [i for i in range(n) if i not in idx_break]
    rng.shuffle(idx)    
    p = 0
    assignment = np.full(n, -1, dtype=int)
    for i, s in enumerate(group_sizes):
        for _ in range(s):
            assignment[idx[p]] = i
            p += 1
    return assignment


def eval_assignment(assignment, A):
    """割り当てを評価"""
    n = len(assignment)
    penalty = 0
        
    # 過去にペアになった回数が多いほどペナルティ
    vmin = np.min(A + np.diag([np.inf] * n))
    for i in range(n):
        if assignment[i] == -1: continue
        for j in range(i+1,n):
            if assignment[j] == -1: continue
            if assignment[i] == assignment[j]:
                penalty += 2 ** (A[i,j] - vmin)
                
    # 過去にペアになった回数が少ないほどペナルティ
    vmax = np.max(A)
    for i in range(n):
        if assignment[i] == -1: continue
        for j in range(i+1,n):
            if assignment[j] == -1: continue
            if assignment[i] != assignment[j]:
                penalty += 2 ** (vmax - A[i,j])
                
    return penalty


def main():
    status = load_text_box("status")
    
    if status[0] == "empty;":
        # 初期化時の動作
        count = 0
        member_names = load_text_box("member-names")
        group_names = load_text_box("group-names")
        
        # validation
        try:
            group_sizes = list(map(int, load_text_box("group-sizes")))
        except:
            dump_error("エラー：グループ人数の形式が間違っています")
            return
             
        if len(group_names) != len(group_sizes):
            dump_error("エラー：グループ情報の形式が間違っています")
            return
        
        if np.any([(s <= 0) for s in group_sizes]):
            dump_error("エラー：無効なグループ人数が入力されています")
            return
        
        if np.sum(group_sizes) > len(member_names):
            dump_error("エラー：グループサイズの合計がメンバー数を超えています")
            return
        
        # フォームのクリア  
        elem = Element("member-names")
        elem.clear()
        elem = Element("group-names")
        elem.clear()
        elem = Element("group-sizes")
        elem.clear()
        
        # ペアの初期化
        n = len(member_names)
        latest_break = np.zeros(n, dtype=int)
        A = np.zeros((n ,n), dtype=int)
        B = np.zeros((n ,n), dtype=int)
    else:
        # ステータスのロード
        count, member_names, group_names, group_sizes, latest_break, A, B = load_status()
        n = len(member_names)
        
    # カウントを加算
    count += 1
        
    # 割り当て作成
    idx_break = next_break(n, group_sizes, latest_break, B)
    assignment = next_assignment(n, group_sizes, idx_break)
    best_assignment, best_penalty = assignment.copy(), eval_assignment(assignment, A)
    
    m = len(group_names)
    for _ in range(1000):
        targets = [np.where(assignment == k)[0] for k in range(m)]
        tar = rng.sample(list(range(m)), 2)
        i = rng.choice(targets[tar[0]])
        j = rng.choice(targets[tar[1]])
        assignment[i], assignment[j] = assignment[j], assignment[i]
        penalty = eval_assignment(assignment, A)
        if best_penalty > penalty:
            best_assignment = assignment.copy()
            best_penalty = penalty
        else:
            assignment[i], assignment[j] = assignment[j], assignment[i]
            
    # 割り当てを反映
    idx = np.where(best_assignment == -1)[0]
    if idx.size > 0:
        latest_break[idx] = count
        B[np.ix_(idx, idx)] += 1
    
    for k in range(len(group_names)):
        idx = np.where(best_assignment == k)[0]
        A[np.ix_(idx, idx)] += 1
        for i in idx: A[i,i] -= 1
    
    # 出力
    dump_assignment(member_names, group_names, best_assignment)
    for i in range(n): A[i,i] = B[i,i] # 休憩回数を A にコピー
    dump_pair_matrix(A)
    dump_status(count, member_names, group_names, group_sizes, latest_break, A, B)
        

##### Options #####
def add_member():
    pass

def remove_member():
    pass
    
##### Initializer #####
def initialize_member_info():
    members = [
        "ちゃこレッド",
        "ちゃこグリーン",
        "ちゃこブルー",
        "ちゃこイエロー",
        "ちゃこピンク",
        "ちゃこホワイト",
        "ちゃこブラック",
        "ちゃこブロンズ",
        "ちゃこシルバー",
        "ちゃこゴールド"
    ]
    elem = Element("member-names")
    elem.write("\n".join(members))
    
    
def initialize_group_info():
    elem = Element("group-names")
    elem.write("アルファチーム\nベータチーム")
    elem = Element("group-sizes")
    elem.write("4\n4")

def initialize_status():
    elem = Element("result")
    elem.write("ここに結果が表示されます。")
    elem = Element("status")
    elem.write("empty;\n")
    
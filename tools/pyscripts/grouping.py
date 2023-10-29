import random as rng
from pyscript import Element


def fetch_member_list():
    """メンバーリスト取得"""
    try:
        member = Element("member").element.value.split()
        assert len(member) > 0
    except:
        display("エラー：メンバーリスト", target="output", append=False)
        return None, None
    
    member_id = {m:i for i, m in enumerate(member)}
    return member, member_id


def fetch_group_list():
    """グループリスト取得"""
    try:
        group = Element("group").element.value.split()
        assert len(group) % 2 == 0
        n_groups = len(group) // 2
        group_name = [group[2 * i] for i in range(n_groups)]
        group_size = [int(group[2 * i + 1]) for i in range(n_groups)]
    except:
        display("エラー：グループリスト", target="output", append=False)
        return None, None
    
    return group_name, group_size


def fetch_status():
    """ステータス取得"""
    try:
        status = Element("status").element.value.split(";")
        status = [sta.strip() for sta in status if len(sta) > 0]
    except:
        display("エラー：ステータス (1)", target="output", append=False)
        return None
    
    return status


def pair_matrix(member_id, group_name, group_size, status):
    """ペア行列の計算"""
    n = len(member_id)
    A = [[0 for _ in range(n)] for _ in range(n)]
    
    try:
        latest_rest = 0
        for line in status:
            if len(line) == 0: continue
            
            line = line.split(":")
            names = line[1].split()
            
            if line[0] == "$":
                for name in names:
                    i = member_id[name]
                    A[i][i] = latest_rest
                latest_rest += 1
            else:
                gi = group_name.index(line[0])
                assert group_size[gi] == len(names)
                  
                m = len(names)
                for i in range(m):
                    ii = member_id[names[i]]
                    for j in range(i+1,m):
                        jj = member_id[names[j]]
                        A[ii][jj] += 1
                        A[jj][ii] += 1
    except:
        display("エラー：ステータス (2)", target="output", append=False)
        return None
    
    return A


def next_assignment(n, group_size):
    """割り当てをランダム生成"""    
    idx = list(range(n))
    rng.shuffle(idx)
    
    p = 0
    assignment = [-1 for _ in range(n)]
    for i, s in enumerate(group_size):
        for _ in range(s):
            assignment[idx[p]] = i
            p += 1
            
    return assignment


def eval_assignment(assignment, A):
    """割り当てを評価"""
    n = len(assignment)
    penalty = 0
    
    dmin, vmin = 1e9, 1e9
    for i in range(n):
        dmin = min(dmin, A[i][i])
        for j in range(i+1,n):
            vmin = min(vmin, A[i][j])
    
    # 休憩はなるべく直近でしてない人から選ぶ
    for i in range(n):
        if assignment[i] == -1:
            penalty += (A[i][i] - dmin) ** 2
    
    # 過去にペアになった回数が多いほどペナルティ
    for i in range(n):
        for j in range(i+1,n):
            if assignment[i] == assignment[j]:
                penalty += (A[i][j] - vmin) ** 2
                
    return penalty


def decode_output(assignment, member, group_name):
    """割り当てを出力用文字列に変換"""
    n, m = len(assignment), len(group_name)
    group, rest = [[] for _ in range(m)], []
    for i in range(n):
        if assignment[i] == -1: rest.append(member[i])
        else: group[assignment[i]].append(member[i])
    group = [" ".join(g) for g in group]
    output = [f"{gname}: {gstr}" for gname, gstr in zip(group_name, group)]
    if len(rest) > 0:
        rest = " ".join(rest)
        output.append(f"$: {rest}")
    return output


def grouping():
    """メイン関数"""
    # load status
    member, member_id = fetch_member_list()
    if member is None: return
    
    group_name, group_size = fetch_group_list()
    if group_name is None: return
    
    status = fetch_status()
    if status is None: return
    
    A = pair_matrix(member_id, group_name, group_size, status)
    if A is None: return
    
    # assignment
    best_assignment, best_penalty = None, None
    for _ in range(10000):
        assignment = next_assignment(len(member), group_size)
        penalty = eval_assignment(assignment, A)
        
        if best_assignment is None:
            best_assignment = assignment
            best_penalty = penalty
        elif best_penalty > penalty:
            best_assignment = assignment
            best_penalty = penalty
    
    # output
    output = decode_output(best_assignment, member, group_name)
    
    # display(member, target="output", append=False)
    # display(group_name, target="output")
    # display(group_size, target="output")
    # display(status, target="output")
    # display(member_id, target="output")
    # display(A, target="output")
    # display(best_assignment, target="output")
    # display(best_penalty, target="output")
    display("次のグループ分け（$は休憩グループを表します）", target="output", append=False)
    for line in output: display(line, target="output")
    
    # ステータス更新
    log = [sta + ";" for sta in status] + [out + ";" for out in output]
    log_elem = Element("status")
    for line in output: log_elem.write("\n".join(log))
    
    
def clear_status():
    for name in ["member", "group", "status"]:
        elem = Element(name)
        elem.write("")
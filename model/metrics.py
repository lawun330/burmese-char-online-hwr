from __future__ import annotations

from typing import List, Sequence, Tuple


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> Tuple[int, int, int]:
    """
    Returns (S, D, I) minimal edit counts converting a -> b.
    """
    n = len(a)
    m = len(b)
    # dp cost
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    op[0][0] = "E"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "E"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                op[i][j] = "S" if best == sub else ("D" if best == dele else "I")

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "E":
            i -= 1
            j -= 1
        elif o == "S":
            S += 1
            i -= 1
            j -= 1
        elif o == "D":
            D += 1
            i -= 1
        elif o == "I":
            I += 1
            j -= 1
        else:
            break

    return S, D, I


def cer(ref: str, hyp: str) -> float:
    r = list(ref)
    h = list(hyp)
    S, D, I = _levenshtein(r, h)
    N = max(1, len(r))
    return (S + D + I) / N


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    S, D, I = _levenshtein(r, h)
    N = max(1, len(r))
    return (S + D + I) / N


def ctc_greedy_decode(
    log_probs_TBC, lengths, *, blank_id: int = 0
) -> List[List[int]]:
    """
    log_probs_TBC: (T,B,C) log-probs
    lengths: (B,) input lengths
    returns list of id sequences (after argmax + CTC collapse)
    """
    import torch

    with torch.no_grad():
        best = torch.argmax(log_probs_TBC, dim=-1)  # (T,B)
        best = best.transpose(0, 1)  # (B,T)
        out: List[List[int]] = []
        for b in range(best.shape[0]):
            T = int(lengths[b].item())
            seq = best[b, :T].tolist()
            collapsed: List[int] = []
            prev = None
            for s in seq:
                if s == prev:
                    continue
                prev = s
                if s != blank_id:
                    collapsed.append(int(s))
            out.append(collapsed)
        return out


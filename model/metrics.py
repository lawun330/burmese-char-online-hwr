from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def _logaddexp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


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


def ctc_beam_decode_topk(
    log_probs_TBC,
    lengths,
    *,
    blank_id: int = 0,
    beam_size: int = 10,
    top_k: int = 5,
) -> List[List[Tuple[List[int], float]]]:
    """
    CTC prefix beam search.

    log_probs_TBC: (T,B,C) log-probs
    lengths: (B,) input lengths
    returns per-batch list of (id_sequence, probability) up to top_k items
    """
    import torch

    with torch.no_grad():
        log_probs = log_probs_TBC.detach().cpu()
        batch_out: List[List[Tuple[List[int], float]]] = []

        for b in range(log_probs.shape[1]):
            t_len = int(lengths[b].item())
            frame_logp = log_probs[:t_len, b, :]
            num_classes = frame_logp.shape[1]

            # prefix -> (log_prob_blank_end, log_prob_non_blank_end)
            beam: dict[tuple[int, ...], Tuple[float, float]] = {(): (0.0, float("-inf"))}

            for t in range(t_len):
                next_beam: dict[tuple[int, ...], Tuple[float, float]] = {}
                for prefix, (lp_b, lp_nb) in beam.items():
                    lp_prefix = _logaddexp(lp_b, lp_nb)
                    for c in range(num_classes):
                        lp = float(frame_logp[t, c].item())
                        if c == blank_id:
                            nb_b, nb_nb = next_beam.get(prefix, (float("-inf"), float("-inf")))
                            next_beam[prefix] = (_logaddexp(nb_b, lp + lp_prefix), nb_nb)
                        elif prefix and c == prefix[-1]:
                            nb_b, nb_nb = next_beam.get(prefix, (float("-inf"), float("-inf")))
                            next_beam[prefix] = (nb_b, _logaddexp(nb_nb, lp + lp_b))
                            new_prefix = prefix + (c,)
                            nb_b2, nb_nb2 = next_beam.get(
                                new_prefix, (float("-inf"), float("-inf"))
                            )
                            next_beam[new_prefix] = (
                                nb_b2,
                                _logaddexp(nb_nb2, lp + lp_nb),
                            )
                        else:
                            new_prefix = prefix + (c,)
                            nb_b, nb_nb = next_beam.get(
                                new_prefix, (float("-inf"), float("-inf"))
                            )
                            next_beam[new_prefix] = (
                                nb_b,
                                _logaddexp(nb_nb, lp + lp_prefix),
                            )

                ranked = sorted(
                    (
                        (pfx, _logaddexp(lp_b, lp_nb))
                        for pfx, (lp_b, lp_nb) in next_beam.items()
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
                beam = {
                    pfx: next_beam[pfx]
                    for pfx, _ in ranked[:beam_size]
                }

            finals = sorted(
                (
                    (list(pfx), _logaddexp(lp_b, lp_nb))
                    for pfx, (lp_b, lp_nb) in beam.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]

            if not finals:
                batch_out.append([])
                continue

            max_log = finals[0][1]
            weights = [math.exp(score - max_log) for _, score in finals]
            total = sum(weights) or 1.0
            batch_out.append(
                [(ids, weight / total) for (ids, _), weight in zip(finals, weights)]
            )

        return batch_out


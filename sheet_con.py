import numpy as np
from typing import List, Tuple


def block_indices(g_pre: int, g_post: int, block_size: int = 3) -> Tuple[List[int], List[int]]:
    """Return flattened indices for a block_size x block_size sub-block
    connecting PEN subgroup g_pre to EPG subgroup g_post.

    Each subgroup contains `block_size` neurons laid out contiguously.
    """
    i_base = np.arange(g_pre * block_size, g_pre * block_size + block_size)
    j_base = np.arange(g_post * block_size, g_post * block_size + block_size)
    ii, jj = np.meshgrid(i_base, j_base, indexing='ij')
    return ii.ravel().tolist(), jj.ravel().tolist()


def build_pen_to_epg_indices() -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build PEN→EPG connectivity index lists.

    Returns:
      pre2, post2: indices for 2x w_PE connections
      pre1, post1: indices for 1x w_PE connections
    """
    pre2: List[int] = []
    post2: List[int] = []
    pre1: List[int] = []
    post1: List[int] = []

    # (A) right side: +1 (2x) and +2 (1x)
    #   PEN 0..6 -> EPG g+1  (2x)
    for g in range(0, 7):
        i, j = block_indices(g, g + 1)
        pre2 += i
        post2 += j
    #   PEN 0..5 -> EPG g+2  (1x), plus PEN6 -> EPG0 (1x)
    for g in range(0, 6):
        i, j = block_indices(g, g + 2)
        pre1 += i
        post1 += j
    i, j = block_indices(6, 0)
    pre1 += i
    post1 += j

    #   PEN 9..15 -> EPG (g-8)+1 = g-7  (2x)  => original method: k4+9 -> k4+1
    for k4 in range(0, 7):
        g = k4 + 9
        i, j = block_indices(g, k4 + 1)
        pre2 += i
        post2 += j
    #   PEN 9..14 -> EPG (g-8)+2 = g-6 (1x); plus PEN15 -> EPG0 (1x)
    for k4 in range(0, 6):
        g = k4 + 9
        i, j = block_indices(g, k4 + 2)
        pre1 += i
        post1 += j
    i, j = block_indices(15, 0)
    pre1 += i
    post1 += j

    # (B) special case on the middle seam: PEN7, PEN8 connected to 0,1,15,14 (2,1,2,1)
    for g in (7, 8):
        for gp, mul in zip([0, 1, 15, 14], [2, 1, 2, 1]):
            ii, jj = block_indices(g, gp)
            if mul == 2:
                pre2 += ii
                post2 += jj
            else:
                pre1 += ii
                post1 += jj

    # (C) left side: +8 (2x) and (+8 corresponding 1x)
    #   PEN 0..6 -> EPG g+8  (2x)
    for g in range(0, 7):
        i, j = block_indices(g, g + 8)
        pre2 += i
        post2 += j
    #   PEN 1..6 -> EPG g+7 (1x), plus PEN0 -> EPG15 (1x)
    for g in range(1, 7):
        i, j = block_indices(g, g + 7)
        pre1 += i
        post1 += j
    i, j = block_indices(0, 15)
    pre1 += i
    post1 += j

    #   PEN 9..15 -> EPG (g-8)+8 = g  (2x) => original method: k4+9 -> k4+8
    for k4 in range(0, 7):
        g = k4 + 9
        i, j = block_indices(g, k4 + 8)
        pre2 += i
        post2 += j
    #   PEN 10..15 -> EPG (g-8)+7 = g-1 (1x), plus PEN9 -> EPG15 (1x)
    for k4 in range(0, 6):
        g = k4 + 10
        i, j = block_indices(g, k4 + 8)
        pre1 += i
        post1 += j
    i, j = block_indices(9, 15)
    pre1 += i
    post1 += j

    return pre2, post2, pre1, post1

def build_pen_to_epg_array():
    pre2, post2, pre1, post1 = build_pen_to_epg_indices()
    return np.array(pre2), np.array(post2), np.array(pre1), np.array(post1)

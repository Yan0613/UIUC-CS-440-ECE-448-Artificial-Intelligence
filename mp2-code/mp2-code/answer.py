# -*- coding: utf-8 -*-
import numpy as np


# fetch the value inside Pentomino
def get_pent_idx_solve(pent):
    pent_idx = 0
    for i in range(pent.shape[0]):
        for j in range(pent.shape[1]):
            if pent[i][j] != 0:
                pent_idx = pent[i][j]
                break
        if pent_idx != 0:
            break
    if pent_idx == 0:
        return -1
    return pent_idx


def transformed_pent_list(this_p):
    list_p = []
    for flip_num in range(3):
        p = np.copy(this_p)
        if flip_num > 0:
            p = np.flip(this_p, flip_num - 1)
        for _ in range(4):
            exist = False
            for trans_p in list_p:
                if np.array_equal(trans_p, p):
                    exist = True
                    break
            if not exist:
                list_p.append(p)
            p = np.rot90(p)
    return list_p


def can_put(board, x, y, pent):
    m = board.shape[0]
    n = board.shape[1]
    pm = pent.shape[0]
    pn = pent.shape[1]
    if x + pm > m or y + pn > n:
        return False
    for row in range(pm):
        for col in range(pn):
            if pent[row][col] != 0:
                if board[x + row][y + col] != 0:  # Overlap
                    return False
    return True


def put(board, x, y, pent):
    number = get_pent_idx_solve(pent)
    pm = pent.shape[0]
    pn = pent.shape[1]
    for row in range(pm):
        for col in range(pn):
            if pent[row][col] != 0:
                board[x + row][y + col] = number
    return True


def recursive_dominos(board_flip, remaining_tiles, transformed_pents):
    if len(remaining_tiles) == 0:
        return ["end marker"]
    candidate_tiles = remaining_tiles.copy()
    m = board_flip.shape[0]
    n = board_flip.shape[1]

    var_idx = candidate_tiles[0]
    transformed_pent_lst = transformed_pents[var_idx]

    for i in range(m):
        for j in range(n):
            for do_pent in transformed_pent_lst:
                if can_put(board_flip, i, j, do_pent):
                    recur_board = board_flip.copy()
                    put(recur_board, i, j, do_pent)
                    recur_remain = candidate_tiles.copy()
                    recur_remain.remove(var_idx)
                    recur_result = recursive_dominos(recur_board, recur_remain, transformed_pents)
                    if len(recur_result) != 0:
                        recur_result.append((do_pent, (i, j)))
                        return recur_result
    return []


def recursive_triominos(board_flip, remaining_tiles, transformed_pents, pents):
    if len(remaining_tiles) == 0:
        return ["End marker"]

    visited_block = set()
    m = board_flip.shape[0]
    n = board_flip.shape[1]

    # check the size of remaining area
    def area(r, c):
        if not (0 <= r < m and 0 <= c < n and (r, c) not in visited_block and board_flip[r][c] == 0):
            return 0
        visited_block.add((r, c))
        return (1 + area(r + 1, c) + area(r - 1, c) +
                area(r, c - 1) + area(r, c + 1))

    for j in range(n):
        for i in range(m):
            if board_flip[i][j] == 0 and (i, j) not in visited_block:
                remaining_area_size = area(i, j)
                if remaining_area_size % 3 != 0:
                    return []

    # print(board)
    candidate_tiles = remaining_tiles.copy()
    # check whether any tile is in a line:
    line_tile = False
    for tile_i in candidate_tiles:
        tri_tile = pents[tile_i]
        if tri_tile.shape[0] == 1 or tri_tile.shape[1] == 1:
            line_tile = True

    # early stop for non-remaining tiles
    if not line_tile:
        for j in range(n - 1):
            if board_flip[0][j] == 0 and board_flip[0][j + 1] == 0 and \
                    board_flip[1][j] != 0 and board_flip[1][j + 1] != 0:
                return []

            if board_flip[m - 2][j] != 0 and board_flip[m - 2][j + 1] != 0 and \
                    board_flip[m - 1][j] == 0 and board_flip[m - 1][j + 1] == 0:
                return []

        for i in range(m - 1):
            if board_flip[i][0] == 0 and board_flip[i][1] != 0 and \
                    board_flip[i + 1][0] == 0 and board_flip[i + 1][1] != 0:
                return []

            if board_flip[i][n - 2] != 0 and board_flip[i][n - 1] == 0 and \
                    board_flip[i + 1][n - 2] != 0 and board_flip[i + 1][n - 1] == 0:
                return []

        for i in range(m - 2):
            for j in range(n - 2):
                if board_flip[i][j] != 0 and board_flip[i][j + 1] != 0 and \
                        board_flip[i + 1][j] == 0 and board_flip[i + 1][j + 1] == 0 and \
                        board_flip[i + 2][j] != 0 and board_flip[i + 2][j + 1] != 0:
                    return []

                if board_flip[i][j] != 0 and board_flip[i][j + 1] == 0 and board_flip[i][j + 2] != 0 and \
                        board_flip[i + 1][j] != 0 and board_flip[i + 1][j + 1] == 0 and board_flip[i + 1][j + 2] != 0:
                    return []

    var_idx = candidate_tiles[0]

    trans_list = transformed_pents[var_idx]
    for j in range(n):
        for i in range(m):
            for tri_pent in trans_list:
                if can_put(board_flip, i, j, tri_pent):
                    recur_board = board_flip.copy()
                    put(recur_board, i, j, tri_pent)
                    recur_remain = candidate_tiles.copy()
                    recur_remain.remove(var_idx)
                    recur_result = recursive_triominos(recur_board, recur_remain, transformed_pents, pents)
                    if len(recur_result) != 0:
                        recur_result.append((tri_pent, (i, j)))
                        return recur_result

    return []


def recursive_pentominos(board_flip, remaining_tiles, transformed_pents):
    if len(remaining_tiles) == 0:
        return ['End marker']
    visited_block = set()
    m = board_flip.shape[0]
    n = board_flip.shape[1]

    # check the size of remaining area
    def area(r, c):
        if not (0 <= r < m and 0 <= c < n and (r, c) not in visited_block and board_flip[r][c] == 0):
            return 0
        visited_block.add((r, c))
        return (1 + area(r + 1, c) + area(r - 1, c) +
                area(r, c - 1) + area(r, c + 1))

    for j in range(n):
        for i in range(m):
            if board_flip[i][j] == 0 and (i, j) not in visited_block:
                remaining_area_size = area(i, j)
                if remaining_area_size % 5 != 0:
                    return []

    candidate_tiles = remaining_tiles.copy()
    var_idx = candidate_tiles[0]
    tran_len = len(transformed_pents[var_idx])
    for index in candidate_tiles:
        transformed_pent_lst = transformed_pents[index]
        cur_len = len(transformed_pent_lst)
        if cur_len < tran_len:
            tran_len = cur_len
            var_idx = index
    transformed_pent_lst = transformed_pents[var_idx]

    for i in range(m):
        for j in range(n):
            for pent in transformed_pent_lst:
                if can_put(board_flip, i, j, pent):
                    recur_board = board_flip.copy()
                    put(recur_board, i, j, pent)
                    recur_remain = candidate_tiles.copy()
                    recur_remain.remove(var_idx)
                    recur_result = recursive_pentominos(recur_board, recur_remain, transformed_pents)
                    if len(recur_result) != 0:
                        recur_result.append((pent, (i, j)))
                        return recur_result

    return []


def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (maybe rotated or flipped), and (rowi, coli) is
    the coordinate of the upper left corner of pi in the board (lowest row and column index
    that the tile covers).

    -Use np.flip and np.rot90 to manipulate pentominos.

    -You may assume there will always be a solution.
    """
    board_flip = 1 - board
    remaining_origin = list(range(len(pents)))
    transformed_pents = {}
    for i, pentomino in enumerate(pents):
        transformed_pents[i] = transformed_pent_list(pentomino)
    pent_size = 0
    for i in range(pents[0].shape[0]):
        for j in range(pents[0].shape[1]):
            if pents[0][i][j] != 0:
                pent_size += 1
    pen_sol = None
    # fetch the size of pent, solve the problem by calling different recursive functions for different pents.
    if pent_size == 2:
        pen_sol = recursive_dominos(board_flip, remaining_origin, transformed_pents)
        pen_sol.pop(0)
    elif pent_size == 3:
        pen_sol = recursive_triominos(board_flip, remaining_origin, transformed_pents, pents)
        pen_sol.pop(0)
    elif pent_size == 5:
        pen_sol = recursive_pentominos(board_flip, remaining_origin, transformed_pents)
        pen_sol.pop(0)

    return pen_sol

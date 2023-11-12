# -*- coding: utf-8 -*-

# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from copy import deepcopy
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
from queue import Queue
from collections import deque
import queue
import  heapq
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start = maze.getStart()
    objectives = maze.getObjectives()
    rows, cols = maze.getDimensions()

    path = []

    visited = [[False] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]
    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True

    while queue:
        now_position = queue.popleft()
        row, col = now_position
        visited[row][col] = True
        if now_position in objectives:
            while now_position != start:
                path.append(now_position)
                now_position = parent[now_position[0]][now_position[1]]
            path.append(start)
            path.reverse()
            return path, sum(sum(visited, []))


        neighbors = maze.getNeighbors(row, col)

        for neighbor in neighbors:
            x, y = neighbor
            if not visited[x][y]:
                queue.append(neighbor)
                parent[x][y] = now_position

    return [], 0

    # # TODO: Write your code here
    # # return path, num_states_explored
    # queue = []
    # visited = set()
    # queue.append([maze.getStart()])
    # while queue:
    #     cur_path = queue.pop(0)
    #     cur_row, cur_col = cur_path[-1]
    #     if (cur_row, cur_col) in visited:
    #         continue
    #     visited.add((cur_row, cur_col))
    #     if maze.isObjective(cur_row, cur_col):
    #         return cur_path, len(visited)
    #     for item in maze.getNeighbors(cur_row, cur_col):
    #         if item not in visited:
    #             queue.append(cur_path + [item])
    # return [], 0


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    stack = []
    path = []
    start = maze.getStart()
    rows, cols = maze.getDimensions()
    stack.append(start)
    visited = [[False] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]

    while stack:
        now_position = stack.pop()
        row, col  = now_position
        visited[row][col] = True
        if maze.isObjective(row,col):
            while now_position != start:
                path.append(now_position)
                now_position = parent[now_position[0]][now_position[1]]
            path.append(start)
            path.reverse()

            return path, sum(sum(visited, []))

        neighbors = maze.getNeighbors(row, col)

        for neighbor in neighbors:
            x, y = neighbor
            if not visited[x][y]:
                stack.append(neighbor)
                parent[x][y] = now_position

    return [], 0

# def dfs(maze):
#     stack = []
#     visited = set()
#     stack.append([maze.getStart()])
#
#     while stack:
#         cur_path = stack.pop()
#         cur_row, cur_col = cur_path[-1]
#
#         if (cur_row, cur_col) in visited:
#             continue
#
#         visited.add((cur_row, cur_col))
#
#         if maze.isObjective(cur_row, cur_col):
#             return cur_path, len(visited)
#
#         for item in maze.getNeighbors(cur_row, cur_col):
#             if item not in visited:
#                 stack.append(cur_path + [item])
#
#     return [], 0



# def greedy(maze):
#     start = maze.getStart()
#     objectives = maze.getObjectives()
#     path = []
#     current_position = start
#
#     while current_position not in objectives:
#         neighbors = maze.getNeighbors(*current_position)
#         min_distance = float('inf')
#         next_position = None
#
#         for neighbor in neighbors:
#             if neighbor not in path:
#                 distance = calculate_distance(neighbor, objectives)
#                 if distance < min_distance:
#                     min_distance = distance
#                     next_position = neighbor
#
#         if next_position:
#             path.append(current_position)
#             current_position = next_position
#         else:
#             # 如果无法前进，回溯到上一个位置
#             if path:
#                 current_position = path.pop()
#             else:
#                 # 无解情况
#                 return [], 0
#
#     # 构建最终路径
#     path.append(current_position)
#     return path, len(path)
#
# def calculate_distance(position, objectives):
#     # 计算位置到目标的曼哈顿距离
#     x1, y1 = position
#     min_distance = float('inf')
#
#     for obj in objectives:
#         x2, y2 = obj
#         distance = abs(x1 - x2) + abs(y1 - y2)
#         min_distance = min(min_distance, distance)
#
#     return min_distance
#

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    pq = queue.PriorityQueue()#具有关联优先级的队列，通常和一个数值一起存储
    visited = set()
    result_row, result_col = maze.getObjectives()[0]
    start_row, start_col = maze.getStart()
    # pq item - tuple: (distance, path list)
    cost = abs(start_row-result_row) + abs(start_col - result_col)
    pq.put((cost, [maze.getStart()]))
    while not pq.empty():
        cur_path = pq.get()[1]
        cur_row, cur_col = cur_path[-1]

        for item in maze.getNeighbors(cur_row, cur_col):
            item_row, item_col = item[0], item[1]
            if maze.isObjective(item_row, item_col):
                visited.add(item)
                cur_path += [item]
                return cur_path, len(visited)

            if item not in visited:
                visited.add(item)
                cost = abs(item[0] - result_row) + abs(item[1] - result_col)
                pq.put((cost, cur_path + [item]))
    return [], 0


# ====================================== Astar PART 1 ===============================================
# astar for part 1&2

def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    pq = queue.PriorityQueue()#具有关联优先级的队列，通常和一个数值一起存储
    visited = set()
    result_row, result_col = maze.getObjectives()[0]
    start_row, start_col = maze.getStart()
    # pq item - tuple: (distance, path list)
    cost = abs(start_row-result_row) + abs(start_col - result_col)
    pq.put((cost, [maze.getStart()]))
    while not pq.empty():
        cur_path = pq.get()[1]
        cur_row, cur_col = cur_path[-1]

        for item in maze.getNeighbors(cur_row, cur_col):
            item_row, item_col = item[0], item[1]
            if maze.isObjective(item_row, item_col):
                visited.add(item)
                cur_path += [item]
                return cur_path, len(visited)

            if item not in visited:
                visited.add(item)
                cost = abs(item[0] - result_row) + abs(item[1] - result_col)+len(cur_path)
                pq.put((cost, cur_path + [item]))
    return [], 0
#====================================== Astar PART 2 ===============================================
#-------------------------------------- ASTAR-standard-answer -----------------------------------------------
class ctor:
    def __init__(self, row, col, cost, tcost):
        self.row = row
        self.col = col
        self.position = (row, col)
        self.sofarcost = 0
        self.cost = cost  # heuristic
        self.tcost = tcost  # f = g + h（total）
        self.prev = None
        self.not_visited = []
        self.objective_left = []

    def __lt__(self, other):
        return self.tcost < other.tcost


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    num_of_state = 0#TODO:未知
    #===========如果只有一个目标点=============
    if len(maze.getObjectives()) == 1:
        start_1 = maze.getStart()
        end_1 = maze.getObjectives()[0]
        return cost_sofar(maze, start_1, end_1)
    #========================================
    start = maze.getStart()#获取起点位置
    goals_left = maze.getObjectives()#获取路径点的位置
    goals_left.insert(0, start)#加入所有的点
    edge_list = {}#TODO:
    heuristic_list = {}#TODO:
    # building graph for mst
    # 遍历（traversal）地图上的点
    for i in goals_left:
        for j in goals_left:
            if i != j:#TODO:为什么非要i!=j?
                construct_path = cost_sofar(maze, i, j)[0]# 计算cost
                edge_list[(i, j)] = construct_path #存储某个点的cost，不过为什么不直接edge_list[(i,j)] = cost_sofar(maze, i, j))
                heuristic_list[(i, j)] = len(construct_path)
                num_of_state += 10#TODO:为什么要加10？
    not_visited_list = {}
    visited = {}
    cur_path = queue.PriorityQueue()
    mst_weights = get_MST(maze, goals_left, heuristic_list)
    start_r, start_c = maze.getStart()
    start_state = ctor(start_r, start_c, 0, mst_weights)
    start_state.not_visited = maze.getObjectives()

    cur_path.put(start_state)
    not_visited_list[(start_r, start_c)] = len(start_state.not_visited)

    while len(goals_left):
        cur_state = cur_path.get()
        if not cur_state.not_visited:
            break
        for n in cur_state.not_visited:
            n_row, n_col = n
            n_cost = cur_state.cost + \
                heuristic_list[(cur_state.position, n)] - 1
            next_state = ctor(n_row, n_col, n_cost, 0)
            next_state.prev = cur_state
            next_state.not_visited = deepcopy(cur_state.not_visited)
            if n in next_state.not_visited:
                next_state.not_visited.remove(n)
            visited[(n_row, n_col)] = 0
            not_visited_list[n] = len(next_state.not_visited)
            mst_weights = get_MST(maze, cur_state.not_visited, heuristic_list)
            next_state.tcost = n_cost + mst_weights
            a = len(goals_left) - 1
            if a:
                next_state.tcost += len(next_state.not_visited)
            cur_path.put(next_state)
    ret_path1 = print_path(maze, edge_list, cur_state, visited)
    return ret_path1, num_of_state


def print_path(maze, path, state, visited):
    ret_path = []
    goals_list = []
    while state:
        goals_list.append(state.position)
        state = state.prev
    total_dot = len(goals_list)-1
    for i in range(total_dot):
        ret_path += path[(goals_list[i], goals_list[i+1])][:-1]
    start = maze.getStart()
    ret_path.append(start)
    ret_path[::-1]
    return ret_path


def get_MST(maze, goals, heuristic_list):
    # Prim
    if not len(goals):
        return 0
    start = goals[0]
    visited = {}
    visited[start] = True
    MST_edges = []
    mst_weights = 0
    while len(visited) < len(goals):
        qe = queue.PriorityQueue()
        for v in visited:
            for n in goals:
                if visited.get(n) == True:
                    continue
                new_edge = (v, n)
                new_cost = heuristic_list[new_edge]-2
                qe.put((new_cost, new_edge))
        add_edge = qe.get()
        MST_edges.append(add_edge[1])
        mst_weights += add_edge[0]
        visited[add_edge[1][1]] = True
    return mst_weights


def cost_sofar(maze, start, end):
    pq = queue.PriorityQueue()
    visited = {}
    result_row, result_col = end
    start_row, start_col = start
    cost = abs(start_row-result_row) + abs(start_col - result_col)
    pq.put((cost, [(start_row, start_col)]))
    while not pq.empty():
        cur_path = pq.get()[1]
        cur_row, cur_col = cur_path[-1]
        if (cur_row, cur_col) in visited:
            continue
        cur_cost = abs(cur_row - result_row) + \
            abs(cur_col - result_col) + len(cur_path) - 1
        visited[(cur_row, cur_col)] = cur_cost
        if (cur_row, cur_col) == (result_row, result_col):
            return cur_path, len(visited)
        for item in maze.getNeighbors(cur_row, cur_col):
            new_cost = abs(item[0] - result_row) + \
                abs(item[1] - result_col) + len(cur_path) - 1
            if item not in visited:
                pq.put((new_cost, cur_path + [item]))
            else:
                # if a node that’s already in the explored set found, test to see if the new h(n)+g(n) is smaller than the old one.
                if visited[item] > new_cost:
                    visited[item] = new_cost
                    pq.put((new_cost, cur_path + [item]))
    return [], 0


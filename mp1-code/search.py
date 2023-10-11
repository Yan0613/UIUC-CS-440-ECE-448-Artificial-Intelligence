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


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)
from queue import Queue
from collections import deque
import queue

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
    pq = queue.PriorityQueue()
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


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0
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



def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0

def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    return [], 0
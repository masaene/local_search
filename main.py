#!/usr/bin/env python3

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import itertools
import random
import math
import numpy as np
import re

TARGET='./st70'
ANSWER=675

TARGET='./nrw1379'
ANSWER=56638

#don't use
#TARGET='./att48'
#ANSWER=10628

#TARGET='./a280'
#ANSWER=2579

#TARGET='./ts225'
#ANSWER=126643

class Solver:
    def __init__(self, tsp_route_list:np.ndarray):
        self.tsp_route_list = tsp_route_list
        self.city_num = tsp_route_list.shape[0]

    def idx2vec(self, index_list:list):
        return self.tsp_route_list[index_list]

    def get_city_num(self):
        return self.city_num

    def update(self, matrix):
        self.tsp_route_list = matrix

    def solve_nearest(self, start_idx:int):
        city_num = self.city_num
        visited = np.zeros(city_num, dtype=bool)
        target_idx = start_idx
        ret_index_list = []
        visited[target_idx] = True
        ret_index_list.append(target_idx)
        yield ret_index_list

        for _ in range(city_num-1):
            tsp_route_idx_list, = np.where(~visited)
            tsp_route_list = self.tsp_route_list[tsp_route_idx_list]
            baseline = self.tsp_route_list[ret_index_list[-1]]
            #diff = tsp_route_list[target_idx] - tsp_route_list
            diff = baseline - tsp_route_list
            norm = np.linalg.norm(diff, axis=1)
            min_idx = np.argmin(norm)

            ret_index_list.append(tsp_route_idx_list[min_idx])
            visited[ret_index_list[-1]] = True
            #time.sleep(0.2)

            yield ret_index_list

    def solve_2opt(self):
        city_num = self.city_num
        ret_index_list = list(range(city_num))

        diff = self.tsp_route_list[:, np.newaxis, :] - self.tsp_route_list[np.newaxis, :, :] 
        diff_matrix = np.linalg.norm(diff, axis=-1)
        ret=np.argmax(diff_matrix)
        i,j=np.unravel_index(ret,diff_matrix.shape)
        st.write(diff_matrix[i,j])

        T = 100
        alpha = 0.995

        while T > 0.01:
            improved = False
            for a,b in itertools.combinations(range(city_num),2):

                crt_diff1 = diff_matrix[ret_index_list[a-1],ret_index_list[a]]
                new_diff1 = diff_matrix[ret_index_list[a-1],ret_index_list[b]]

                crt_diff2 = diff_matrix[ret_index_list[b],ret_index_list[(b+1)%city_num]]
                new_diff2 = diff_matrix[ret_index_list[a],ret_index_list[(b+1)%city_num]]

                new_diff_total = new_diff1 + new_diff2
                crt_diff_total = crt_diff1 + crt_diff2

                delta = new_diff_total - crt_diff_total

                accept = False
                if delta < 0:
                    accept = True
                else:
                    probability = math.exp(-delta/T)
                    accept = random.random() < probability

                if accept == True:
                    improved = True
                    ret_index_list[a:b+1] = ret_index_list[a:b+1][::-1]
            yield (ret_index_list,T)
            T = T * alpha
            if improved == False:
                break
        yield (ret_index_list,T)


def count(matrix):
    diffs = np.diff(matrix, axis=0)
    return np.linalg.norm(diffs, axis=1).sum()

def draw(graph_area, coordinate_arr, name):
    x = coordinate_arr[:,0]
    y = coordinate_arr[:,1]
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', markersize=3, markerfacecolor='orange')
    norm_total = count(coordinate_arr)
    divergence_ratio = round((norm_total / ANSWER) * 100, 2)
    title = f'{name}, diff={divergence_ratio}%'
    fig.suptitle(title)

    graph_area.pyplot(fig)
    plt.close(fig)

if __name__ == '__main__':
    if 'main_data' not in st.session_state:
        l = list()
        with open(TARGET, 'r') as f:
            for v in f.read().splitlines():
                sp = re.split(r'\s+', v)
                x = int(sp[1])
                y = int(sp[2])
                l.append([x,y])
            vec = np.array(l)
            st.session_state['main_data'] = vec
    vec = st.session_state['main_data']

    solver = Solver(vec)
    city_num = solver.get_city_num()
    graph_area = st.empty()
    coordinate_arr = None
    for i, ret_index_list in enumerate(solver.solve_nearest(0)):
        coordinate_arr = solver.idx2vec(ret_index_list + [ret_index_list[0]])
    draw(graph_area, coordinate_arr, 'nearest')
    solver.update(coordinate_arr)

    graph_area = st.empty()
    for i, (ret_index_list, T) in enumerate(solver.solve_2opt()):
        coordinate_arr = solver.idx2vec(ret_index_list + [ret_index_list[0]])
        draw(graph_area, coordinate_arr, f'2opt({T})')
    solver.update(coordinate_arr)




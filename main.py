#!/usr/bin/env python3

import matplotlib.pyplot as plt
import streamlit as st
import itertools
import numpy as np
import time
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

    def count(self, matrix):
        diffs = np.diff(matrix, axis=0)
        return np.linalg.norm(diffs, axis=1).sum()

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
        for a,b in itertools.combinations(range(city_num),2):
            tmp = ret_index_list[:]
            tmp[a:b+1] = tmp[a:b+1][::-1]

            if a == 0:
                new_diff1 = self.tsp_route_list[tmp[a]] - self.tsp_route_list[tmp[-1]]
                crt_diff1 = self.tsp_route_list[ret_index_list[a]] - self.tsp_route_list[-1]
            else:
                new_diff1 = self.tsp_route_list[tmp[a-1]] - self.tsp_route_list[tmp[a]]
                crt_diff1 = self.tsp_route_list[ret_index_list[a-1]] - self.tsp_route_list[ret_index_list[a]]

            if b == (city_num-1):
                new_diff2 = self.tsp_route_list[tmp[b]] - self.tsp_route_list[tmp[0]]
                crt_diff2 = self.tsp_route_list[ret_index_list[b]] - self.tsp_route_list[ret_index_list[0]]
            else:
                new_diff2 = self.tsp_route_list[tmp[b+1]] - self.tsp_route_list[tmp[b]]
                crt_diff2 = self.tsp_route_list[ret_index_list[b+1]] - self.tsp_route_list[ret_index_list[b]]

            new_diff_total = np.linalg.norm(new_diff1) + np.linalg.norm(new_diff2)
            crt_diff_total = np.linalg.norm(crt_diff1) + np.linalg.norm(crt_diff2)
            if new_diff_total < crt_diff_total:
                ret_index_list = tmp

            yield ret_index_list



    '''
    def greedy_all(self, name:str):
        best_l = None
        best = float('inf')
        for i in range(len(self.vector_list)):
            local_ans, l = solv.greedy(i, None)
            if local_ans < best:
                best = local_ans
                best_l = l
                if name is not None:
                    self.update_graph(name, '', best, best_l)
        return self.count(best_l), copy.deepcopy(best_l)

    def count(self, l):
        new_l = np.vstack((l, l[0:1]))
        d = np.diff(new_l, axis=0)
        norms = np.linalg.norm(d, axis=1)
        return np.sum(norms)

#   def count(self, l):
#       n = len(l)
#       ret = 0
#       for i,j in zip(range(0,n-1), range(1,n)):
#           a = np.array(l[i,:])
#           b = np.array(l[j,:])
#           d = a - b
#           ret += np.linalg.norm(d)
#       a = np.array(l[0,:])
#       b = np.array(l[-1,:])
#       d = a - b
#       ret += np.linalg.norm(d)
#       return ret

    def ngr2opt(self, best_l, name:str):
        total = len(best_l)
        best = float('inf')
        org_list = copy.deepcopy(best_l)
        best_l2 = org_list

        for a,b in itertools.combinations(range(1,len(best_l2)-1), 2):
            new_route_a = np.vstack((best_l2[:a], best_l2[a:b+1][::-1]))
            new_route = np.vstack((new_route_a, best_l2[b+1:]))
            count = self.count(new_route)
            if count < best:
                best = count
                best_l2 = new_route
        if name is not None:
            self.update_graph(name, '', best, best_l2)
        return best, copy.deepcopy(best_l2)

    def ngr2opt_all(self, best_l, name:str):
        best = float('inf')
        best_l2 = best_l
        while True:
            ans, l = self.ngr2opt(best_l2, name)
            if ans < best:
                best = ans
                best_l2 = l
                if name is not None:
                    self.update_graph(name, '', best, best_l2)
            else:
                break
        if name is not None:
                self.update_graph(name, '', best, best_l2)
        return best, copy.deepcopy(best_l2)

    def sa2opt_all(self, name:str, best_l, t, c):
        best = float('inf')
        best_l2 = best_l
        while True:
            ans, l = self.ngr2opt(best_l2, None)
            d = ans - best
            r = random.random()
            r_output = round(r,2)
            e_output = round(np.e,2)
            d_output = round(-d,4)
            t_output = round(t,4)
            ret = np.e ** (-d / t)
            ret_output = round(ret,4)
            if ans < best:
                best = ans
                best_l2 = l
                self.update_graph(name, '', ans, best_l2)
            elif r <= ret:
                best_l2 = l
                self.update_graph(name, '', ans, best_l2)
            else:
                break
            t = t * c
        return self.count(best_l2), copy.deepcopy(best_l2)

    def update_graph(self, name:str, remarks:str, ans:int, best_l):
        local_l = np.vstack((best_l, best_l[0:]))
        #local_l = copy.deepcopy(best_l)
        #last = np.array(local_l[-1:])
        #local_l = np.append(local_l, [last], axis=0)

        graph = self.graph_dict[name]['graph']
        fig = self.graph_dict[name]['fig']
        ax = self.graph_dict[name]['ax']

        diff = ans - ANSWER
        divergence_ratio = round((diff / ANSWER) * 100, 2)
        ans_str = f'{name}({remarks}):ret={round(ans,2)}, diff={divergence_ratio}%'
        fig.suptitle(ans_str)
        ax.plot(local_l[:,0], local_l[:,1])
        """
        for no, (_x, _y) in enumerate(zip(local_l[:,0], local_l[:,1])):
            ax.text(_x,_y,no)
        """
        graph.pyplot(fig)
        plt.cla()
    '''

def draw(graph_area, coordinate_arr, name):
    x = coordinate_arr[:,0]
    y = coordinate_arr[:,1]
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', markersize=3, markerfacecolor='orange')
    norm_total = solver.count(coordinate_arr)
    divergence_ratio = round((norm_total / ANSWER) * 100, 2)
    title = f'{name}, diff={divergence_ratio}%, {i+1}/{city_num}'
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
        #draw(graph_area, coordinate_arr, 'nearest')
    draw(graph_area, coordinate_arr, 'nearest')

    solver.update(coordinate_arr)

    graph_area = st.empty()
    for i, ret_index_list in enumerate(solver.solve_2opt()):
        coordinate_arr = solver.idx2vec(ret_index_list + [ret_index_list[0]])
    draw(graph_area, coordinate_arr, '2opt')

    '''
    solv.add_graph('greedy all')
    ans, best_l = solv.greedy_all('greedy all')

    solv.add_graph('2opt')
    ans, best_l = solv.ngr2opt(best_l, '2opt')

    solv.add_graph('2opt all')
    ans, best_l = solv.ngr2opt_all(best_l, '2opt all')
    '''

    """
    solv.add_graph('simulated aneealing(2opt)')
    ans, best_l = solv.sa2opt_all('simulated aneealing(2opt)', solv.get_list(), 100, 0.9)
    """

    #d = l[1] - l[0]
    #l2 = np.linalg.norm(d)




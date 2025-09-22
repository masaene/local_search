#!/usr/bin/env python3

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import random
import math
import copy
import sys
import re
import itertools

TARGET='./st70'
ANSWER=675

#TARGET='./nrw1379'
#ANSWER=56638

#TARGET='./att48'
#ANSWER=10628

#TARGET='./a280'
#ANSWER=2579

#TARGET='./ts225'
#ANSWER=126643

class Solver:
	def __init__(self, vector_list:list):
		self.vector_list = vector_list
		self.city_num = len(self.vector_list)
		self.graph_dict = dict()

	def add_graph(self, name:str):
		graph = st.empty()
		fig, ax = plt.subplots()
		d = dict()
		d['graph'] = graph
		d['fig'] = fig
		d['ax'] = ax
		self.graph_dict[name] = d


	def get_list(self):
		return self.vector_list

	def greedy(self, init:int, name:str):
		org_list = copy.deepcopy(self.vector_list)
		start_pos = init
		ret_list = np.array([org_list[start_pos,:]])
		org_list = np.delete(org_list, start_pos, axis=0)
		while True:
			pre_ret = float('inf')
			best_vec = None
			best_idx = 0
			diff = ret_list[-1] - org_list
			all_l2 = np.linalg.norm(diff, axis=1)
			best_idx = np.argmin(all_l2)
			pre_ret = all_l2[best_idx]

			ret_list = np.append(ret_list, [org_list[best_idx,:]], axis=0)
			org_list = np.delete(org_list, best_idx, axis=0)

			if len(org_list) == 0:
				break
		if name is not None:
			count = self.count(ret_list)
			self.update_graph(name, '', count, ret_list)

		return self.count(ret_list), copy.deepcopy(ret_list)

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

#	def count(self, l):
#		n = len(l)
#		ret = 0
#		for i,j in zip(range(0,n-1), range(1,n)):
#			a = np.array(l[i,:])
#			b = np.array(l[j,:])
#			d = a - b
#			ret += np.linalg.norm(d)
#		a = np.array(l[0,:])
#		b = np.array(l[-1,:])
#		d = a - b
#		ret += np.linalg.norm(d)
#		return ret

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

if __name__ == '__main__':
	l = list()
	with open(TARGET, 'r') as f:
	#with open('./sato', 'r') as f:
		for v in f.read().splitlines():
			sp = re.split('\s+', v)
			x = int(sp[1])
			y = int(sp[2])
			l.append([x,y])
		vec = np.array(l)


	solv = Solver(vec)

	solv.add_graph('greedy')
	ans, best_l = solv.greedy(0, 'greedy')

	solv.add_graph('greedy all')
	ans, best_l = solv.greedy_all('greedy all')

	solv.add_graph('2opt')
	ans, best_l = solv.ngr2opt(best_l, '2opt')

	solv.add_graph('2opt all')
	ans, best_l = solv.ngr2opt_all(best_l, '2opt all')

	"""
	solv.add_graph('simulated aneealing(2opt)')
	ans, best_l = solv.sa2opt_all('simulated aneealing(2opt)', solv.get_list(), 100, 0.9)
	"""

	#d = l[1] - l[0]
	#l2 = np.linalg.norm(d)




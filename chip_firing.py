import numpy as np
import networkx as nx

from graph import UGraph

class Chip_Firing(UGraph):
    """Chip-firing游戏的实现"""
    def __init__(self,matrix,val=None):
        UGraph.__init__(self,matrix,val)

    def Firing(self,id):
        '''游戏中的firing操作'''
        flg = True
        op_node = self.Get_node(id) #操作的节点
        if op_node.Value >= op_node.degree:
            op_node.Value -= op_node.degree
            for i in op_node.Edge:
                self.Get_node(i).Value += 1
        else:
            flg = False
        return flg
    
    def Firing_plain(self,id,val:list):
        '''也是游戏中的firing操作,但对原图的权值无影响'''
        flg = True
        ret = val.copy() #由于val_lst中储存的是整数, 故仅用浅拷贝即可
        op_node = self.Get_node(id) #操作的节点
        if ret[id] >= op_node.Degree: #判断
            ret[id] -= op_node.Degree
            for x in op_node.Edge:
                ret[x] += 1
        else:
            flg = False
        return ret,flg

    def Lock_judge(self,val:np.ndarray) -> bool:
        '''锁死状态判断'''
        flg = False if np.all(self.Deg_lst > val) else True
        return flg

class Chip_firing_nx:
    def __init__(self,matrix,val=None):
        self.Graph = nx.DiGraph(matrix)
        
    def Firing(self,id):
        pass

    def Lock_judge(self,val:np.ndarray) -> bool:
        pass
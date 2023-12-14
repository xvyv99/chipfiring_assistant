import numpy as np

class Node:
    """图的节点"""
    __Version = "Node beta 0.2"
    def __init__(self,id):
        self.Id:int = id #节点的编号
        self.Value:float = 1.0 #节点的权值
        self.__Edge:set = set() #指向其他节点的边

    def Add_edge(self,node):
        '''增加单项边,从self指向node'''
        self.__Edge.add(node.Id)

    @property
    def Degree(self):
        '''获取节点的度'''
        return len(self.__Edge)

class UGraph:
    """
    无向图 注:除权值外, 图及节点之间的关系一经创建便无法修改
    """
    __Version = "UGraph beta 0.2"
    def __init__(self,matrix,vals = None):
        if self.Symmetric_check(matrix):
            self.Adj_matrix = matrix #图的邻接矩阵
        else:
            raise "[Error]The adjacency matrix can't generate a undirect graph."
        self.N = matrix.shape[0] #图的节点个数

        self.Node_lst = [Node(i) for i in range(self.N)] #图中所包含的节点列表, 编号 从0开始
        self.Val_lst = np.zeros(self.N,dtype=float) if vals == None else np.array(vals) #图中各个节点的权值组成的列表, 序号对应于节点编号
        self.Deg_lst = np.zeros(self.N,dtype=int)

        self.Translate()

    @staticmethod
    def Symmetric_check(matrix) -> bool:
        '''邻接矩阵对称性检查'''
        flg = True
        (m,n) = matrix.shape
        flg = False if m != n else True #检查是否为方阵
        flg = False if np.all(matrix == matrix.transpose()) else True #检查转置后是否等于原来的矩阵
        return flg
    
    def Get_node(self,id):
        return self.Node_lst[id]
    
    def Translate(self):
        '''将邻接矩阵转化为图'''
        for i,node_1 in enumerate(self.Node_lst):
            for j,node_2 in enumerate(self.Node_lst):
                if self.Adj_matrix[j,i]:
                    node_1.Add_edge(node_2)
                    node_1.Value = self.Val_lst[i]

    @property
    def Values(self) -> list:
        '''各节点的权值获取'''
        self.Val_lst = np.array([n.Value for n in self.Node_lst])
        return self.Val_lst
    
    @Values.setter
    def Values(self,lst:list) -> list:
        '''各节点的权值设置'''
        for i,node in enumerate(self.Node_lst):
            self.node.Value = lst[i]
    
    @property
    def Degrees(self) -> list:
        '''
        各节点的度获取 注:最好用Deg_lst, 因为每使用一次Degrees都要重新获取, 而各节点的度显然在初始化后是不会变的
        '''
        self.Deg_lst = np.array([n.Degree for n in self.Node_lst])
        return self.Deg_lst
import time,os,math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from chip_firing import Chip_Firing,Chip_firing_nx
from console import Console

from rich.progress import track
import hashlib,pickle

def hash_matrix(matrix):
    return hashlib.sha1(pickle.dumps(matrix)).hexdigest().replace("/","_")

class Chip_Firing_I(Chip_Firing,Console):
    __Version = "beta 0.2"
    def __init__(self,matrix=None):
        Console.__init__(self)
        self.WELCOME = "Welcome to Chip-Firing game assist. Get help by typing \"Help\" in this console." #欢迎文本
        self.VERSION = self.__Version

        self.time_rec = time.time()
        #os.mkdir(str(self.time_rec))

        self.Xs = None
        self.Shape = matrix.shape
        self.Matrix = matrix

        self.Xs = 0

        self.History = set()
        self.History_graph = nx.DiGraph()
        #初始化状态空间图和状态空间
        
        Chip_Firing.__init__(self,self.Matrix)

        self.Welcome()
        self.Analyzer()

    @property
    def Hash_code(self):
        return hash_matrix(self.Matrix)

    def RegisterC(func):
        return Console.RegisterC(func)

    @RegisterC
    def RandomAdjMatrix(self,shape) -> np.ndarray:
        '''创建一个随机无向图的邻接矩阵,可能效率很低,且无法保证其为连通图'''
        def check_matrix(matrix): #确保图为连通图,有问题,不起作用
            x = np.sum(matrix,axis=0)
            return 0 not in x
        flg = True
        while flg:
            rmatrix = np.random.randint(0,2,shape,dtype=int) #创建随机0,1矩阵

            np.fill_diagonal(rmatrix,0)
            rmatrix = np.triu(rmatrix)
            rmatrix += rmatrix.transpose()
            #随机矩阵对称化
            flg = False if check_matrix(rmatrix) else True
        return rmatrix
    
    @RegisterC
    def RandomValues(self) -> np.ndarray:
        '''创建随机权值列表'''
        Maximum = 10 #单个节点权值最大上限(不包括)
        flg = True
        while flg:
            VL = np.random.randint(0,Maximum,size=self.N,dtype=int)
            flg = self.Lock_judge(VL) #防止遍历开始就锁死
        return VL

    def Erg_vals(self,length,xs):
        '''不重复地生成长度一定,和为定值的非负整数列表 仍需改进'''
        if length==1:
            yield [xs]
        elif length<1:
            yield -1
        for x in range(xs+1):
            g = self.Erg_vals(length-1,xs-x)
            for y in g:
                if y == -1:
                    g.close()
                    continue #可能存在冗余
                else:
                    yield [x]+y

    @RegisterC
    def PlainSearch(self,vals:list,draw_option=True):
        '''构建子状态空间树和部分状态空间图'''
        history =set() #子状态空间
        history_graph = nx.DiGraph() #子状态空间树
        color_dict = {"Red":"#FF0000","Blue":"#1f78b4","Yellow":"#FFFF00","Green":"#33a02c","Grey":"#C0C0C0"} #颜色代码字典

        def step(val_lst:list):
            '''单步搜索归递函数'''
            gid = ','.join([str(x) for x in val_lst]) #父节点的标签
            for x in range(self.Shape[0]):
                res,flg = self.Firing_plain(x,val_lst) #Firing操作
                if flg:
                    node_id = ','.join([str(x) for x in res]) #生成子节点标签
                    if (tuple(res) not in self.History):
                        self.History.add(tuple(res))
                        if np.all(res <= (self.Degrees-1)): #判断节点是否死锁
                            self.History_graph.add_node(node_id,color=color_dict["Grey"])
                            self.History_graph.add_edge(gid,node_id,toward=x)
                            #是则将节点变为灰色
                        else:
                            self.History_graph.add_node(node_id,color=color_dict["Blue"])
                            self.History_graph.add_edge(gid,node_id,toward=x)
                            #否则将节点变为蓝色
                    else:
                        self.History_graph.add_edge(gid,node_id,toward=x)
                    #构建状态空间图过程

                    if (tuple(res) not in history):
                        history.add(tuple(res))
                        if np.all(res <= (self.Degrees-1)):
                            history_graph.add_node(node_id,color=color_dict["Grey"]) #灰色节点为死锁节点
                            history_graph.add_edge(gid,node_id,toward=x)
                        else:
                            history_graph.add_node(node_id,color=color_dict["Blue"]) #绿色节点为正常节点
                            history_graph.add_edge(gid,node_id,toward=x)
                        step(res)
                    else:
                        if np.all(res == val_lst):
                            history_graph.add_node(node_id,color=color_dict["Yellow"]) #黄色节点代表与初始状态相同的节点
                        else:
                            history_graph.add_node(node_id,color=color_dict["Red"]) #红色节点代表重复的结束节点
                        history_graph.add_edge(gid,node_id,toward=x)
                    #构建子状态空间树过程

        init_id = ','.join([str(x) for x in vals]) #生成子状态空间树的根节点的标签
        if tuple(vals) not in self.History:
            self.History.add(tuple(vals))
            self.History_graph.add_node(init_id,color=color_dict["Blue"])

        history.add(tuple(vals))
        history_graph.add_node(init_id,color=color_dict["Blue"])
        step(vals)
        #初始化

        #plt.savefig(hash_code+'\\all.png',dpi = 128)
        #nx.write_gexf(self.History_graph,hash_code+'\\All.gexf')

        if draw_option:
            color_map = nx.get_node_attributes(history_graph,"color").values() #节点的颜色表
            pos_tree = nx.nx_agraph.graphviz_layout(history_graph, prog="dot") #树状布局
            #nx.draw_networkx方法需要
            nx.draw_networkx(history_graph,node_color=color_map,pos=pos_tree,with_labels=True) #绘制子状态空间树
            plt.show()
            plt.close()
        
    def Search(self,vals:list):
        '''构建子状态空间树和部分状态空间图'''
        history =set() #子状态空间
        history_graph = nx.DiGraph() #子状态空间树
        color_dict = {"Red":"#FF0000","Blue":"#1f78b4","Yellow":"#FFFF00","Green":"#33a02c","Grey":"#C0C0C0"} #颜色代码字典

        def step(val_lst:list):
            '''单步搜索归递函数'''
            gid = ','.join([str(x) for x in val_lst]) #父节点的标签
            for x in range(self.Shape[0]):
                res,flg = self.Firing_plain(x,val_lst) #Firing操作
                if flg:
                    node_id = ','.join([str(x) for x in res]) #生成子节点标签
                    if (tuple(res) not in self.History):
                        self.History.add(tuple(res))
                        if np.all(res <= (self.Degrees-1)): #判断节点是否死锁
                            self.History_graph.add_node(node_id,color=color_dict["Grey"])
                            self.History_graph.add_edge(gid,node_id,toward=x)
                            #是则将节点变为灰色
                        else:
                            self.History_graph.add_node(node_id,color=color_dict["Blue"])
                            self.History_graph.add_edge(gid,node_id,toward=x)
                            #否则将节点变为蓝色
                    else:
                        self.History_graph.add_edge(gid,node_id,toward=x)
                    #构建状态空间图过程

                    if (tuple(res) not in history):
                        history.add(tuple(res))
                        step(res)

        init_id = ','.join([str(x) for x in vals]) #生成子状态空间树的根节点的标签
        if tuple(vals) not in self.History:
            self.History.add(tuple(vals))
            self.History_graph.add_node(init_id,color=color_dict["Blue"])
        history.add(tuple(vals))
        step(vals)
        #初始化

    @RegisterC
    def BuildGraph(self,xs,draw_option=False):
        '''构建完整的状态空间图'''
        self.Xs = xs
        print(self.Matrix)
        self.History = set()
        self.History_graph = nx.DiGraph()
        #初始化状态空间图和状态空间

        s = math.comb(xs+self.Shape[0]-1,self.Shape[0]-1)
        for x in track(self.Erg_vals(self.Shape[0],xs),total=s,description='Processing...'):
            self.PlainSearch(x,False)

        hash_code = self.Hash_code
        if os.path.isdir(hash_code):
            pass
        else:
            os.mkdir(hash_code)
        if os.path.isdir(hash_code+"\\"+str(xs)):
            pass
        else:
            os.mkdir(hash_code+"\\"+str(xs))
        nx.write_gexf(self.History_graph,hash_code+"\\"+str(xs)+"\\All.gexf")
        print("Visualization in progress...")

        print("Colorize the picture...",end=" ")
        #color_m = nx.get_node_attributes(self.History_graph,"color").values()
        print("Done.")

        print("Plot...",end=" ")
        #nx.draw_networkx(self.History_graph,node_color=color_m,with_labels=True)
        print("Done.")

        print("Archiving...",end=" ")
        #plt.savefig(hash_code+"\\"+str(xs)+"\\all.png",dpi = 128)
        print("Done.")
        #计算矩阵哈希,然后保存图

        if draw_option:
            plt.show()

    @RegisterC
    def Plot(self,graph_name):
        if graph_name == "G":
            G = nx.DiGraph(self.Matrix)
            nx.draw_networkx(G)

            hash_code = self.Hash_code
            if os.path.isdir(hash_code):
                pass
            else:
                os.mkdir(hash_code)
            if os.path.isdir(hash_code):
                pass
            else:
                os.mkdir(hash_code)

            plt.savefig(hash_code+'\\G.png',dpi = 128)
            plt.show()
            plt.close()
        elif graph_name == "Gs":
            color_m = nx.get_node_attributes(self.History_graph,"color").values()
            nx.draw_networkx(self.History_graph,node_color=color_m,with_labels=True)
            plt.show()
            plt.close()

    @RegisterC
    def SpecialGraph(self,graph_name,*para):
        ret = None
        if graph_name == "Petersen":
            ret = nx.to_numpy_array(nx.petersen_graph(*para))
        elif graph_name == "K1":
            ret = nx.to_numpy_array(nx.complete_graph(*para))
        elif graph_name == "K2":
            ret = nx.to_numpy_array(nx.complete_bipartite_graph(*para))
        elif graph_name == "tute":
            ret = nx.to_numpy_array(nx.tutte_graph())
        elif graph_name == "maze":
            ret = nx.to_numpy_array(nx.sedgewick_maze_graph())
        elif graph_name == "tet":
            ret = nx.to_numpy_array(nx.tetrahedral_graph())
        elif graph_name == "barbell":
            ret = nx.to_numpy_array(nx.barbell_graph(*para))
        elif graph_name == "barbell":
            ret = nx.to_numpy_array(nx.lollipop_graph(*para))
        else:
            return False
        self.Matrix = ret
        self.Shape = ret.shape
        Chip_Firing.__init__(self,self.Matrix)

    @RegisterC
    def GetInfo(self,op=None):
        '''获取此状态下游戏相关信息'''
        if op is None:
            print("<G>")
            print(f"    Adjacency matrix:\n{self.Matrix}")

            Tmp_1 = 0
            cycles = nx.simple_cycles(self.History_graph)
            for i,__ in enumerate(cycles):
                Tmp_1 += 1
            print(f"    Cycles:{Tmp_1}")

            print("<Gs>")
            print(f"    Xs:{self.Xs}")
            Tmp_0 = 0
            for i,__ in enumerate(nx.connected_components(nx.Graph(self.History_graph))): #循环获取状态空间图分量个数
                Tmp_0 += 1
            print(f"    Islands:{Tmp_0}")

            print(len(nx.connected_components(self.History_graph)))

            Tmp_1 = 0
            cycles = nx.simple_cycles(self.History_graph)
            for i,__ in enumerate(cycles):
                Tmp_1 += 1
            print(f"    Cycles:{Tmp_1}")
            #Gs的信息
        elif op == "G":
            print("<G>")
            print(f"    Adjacency matrix:\n{self.Matrix}")

            Tmp_1 = 0
            cycles = nx.simple_cycles(self.History_graph)
            for i,__ in enumerate(cycles):
                Tmp_1 += 1
            print(f"    Cycles:{Tmp_1}")
        elif op == "Gs":
            print("<Gs>")
            print(f"    Xs:{self.Xs}")
            Tmp_0 = 0
            for i,__ in enumerate(nx.connected_components(nx.Graph(self.History_graph))): #循环获取状态空间图分量个数
                Tmp_0 += 1
            print(f"    Islands:{Tmp_0}")

            print(len(nx.connected_components(self.History_graph)))

            Tmp_1 = 0
            cycles = nx.simple_cycles(self.History_graph)
            for i,__ in enumerate(cycles):
                Tmp_1 += 1
            print(f"    Cycles:{Tmp_1}")
            #Gs的信息

mat = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]])
mat_1 = nx.to_numpy_array(nx.petersen_graph())
mat_2 = np.array([[0,1,0,0,0,0,0],[1,0,1,0,1,0,0],[0,1,0,1,0,0,0],[0,0,1,0,0,1,1],[0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0]])
A = Chip_Firing_I(mat_2)

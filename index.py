import time,os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from chip_firing import Chip_Firing,Chip_firing_nx

class CF_Sim(Chip_Firing):
    def __init__(self,m):
        self.time_rec = time.time()
        
        self.History_data = []#Id_Set()
        self.History_list = []
        self.History_graph = nx.DiGraph()
        self.Shape = (m,m)
        self.Matrix = self.create_AM(self.Shape)
        Chip_Firing.__init__(self,self.Matrix)
        self.Init_values = self.Values =self.create_VL()

        #self.show_init()
    @staticmethod
    def Create_AM(shape) -> np.ndarray:
        '''创建一个随机无向图的邻接矩阵,可能效率很低,且无法保证其为连通图'''
        def check_matrix(matrix): #确保图为连通图,其实有问题
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

    def Create_VL(self) -> np.ndarray:
        '''创建随机权值列表'''
        Maximum = 10 #单个节点权值最大上限(不包括)
        flg = True
        while flg:
            VL = np.random.randint(0,Maximum,size=self.N,dtype=int)
            flg = self.Lock_judge(VL) #防止遍历开始就锁死
        return VL

    def Rand_search(self):
        '''随机搜索'''
        flg = False
        i = 0
        self.History_graph.add_node(i)
        while not flg:
            x = np.random.randint(0,self.N)
            flg = self.Firing(x)
        '''
        flg = True if self.History_data.add(tuple(self.Values)) else False
        if flg:
            i += 1
            self.History_graph.add_node(i)
            self.History_graph.add_edge(i-1,i)
        else:
            self.History_graph.add_edge(i-1,i)
            self.History_graph.add_edge(i-1,1+self.History_data.index(tuple(self.Values)))
        '''
        print("After firing ",x,self.Values,len(self.History_data))
        return x,self.Val_lst
    
    def show_init(self):
        ''''''
        print("Init Adj matrix: \n",self.Adj_matrix)
        print("Init Valve:",self.Init_values)
        print("Init graph:")
        print("Valve sum:",self.Init_values.sum())
        print("Degrees:",self.Degrees)
        Tmp = nx.DiGraph(self.Matrix)
        nx.draw(Tmp,with_labels=True)
        plt.savefig('IAM_'+str(self.time_rec)+'.png')
        plt.show()

    def plain_search(self):
        def step(val_lst):
            Gid = ','.join([str(x) for x in val_lst])
            for x in range(self.Shape[0]):
                res,flg = self.firing_plain(x,val_lst) #冗余
                if flg and (tuple(res) not in self.History_list):
                    node_id = ','.join([str(x) for x in res])
                    self.History_list.append(tuple(res))
                    self.History_graph.add_node(node_id,color='#33a02c')
                    self.History_graph.add_edge(Gid,node_id,toward=x)
                    #print(Gid,f'--[{x}]->',node_id)
                    #print(self.History_list)
                    step(res)
                    #continue
                elif flg == True:
                    node_id = ' '.join([str(x) for x in res])
                    if np.all(res == self.Init_values):
                        self.History_graph.add_node(node_id,color='#FFFF00')
                    else:
                        self.History_graph.add_node(node_id,color='#FF0000')
                    self.History_graph.add_edge(Gid,node_id,toward=x)
                    #print(Gid,f'--[{x}]->',node_id)
            #nx.draw(self.History_graph,with_labels=True)
            #plt.show()
        init_id = ','.join([str(x) for x in self.Init_values])
        self.History_list.append(tuple(self.Init_values))
        self.History_graph.add_node(init_id,color='#1f78b4')
        step(self.Init_values)

        color_map = nx.get_node_attributes(self.History_graph,"color").values()
        pos_n = nx.nx_agraph.graphviz_layout(self.History_graph, prog="dot")
        #color_map = ['#33a02c' if self.History_graph.nodes[x]['color'] == 0 else '#1f78b4' for x in self.History_graph.nodes()]
        nx.draw_networkx(self.History_graph,node_color=color_map,pos=pos_n,with_labels=True)
        plt.savefig('PST_'+str(self.time_rec)+'.png')
        plt.show()

        print("The number of state:",len(self.History_list))
        print("The state:",self.History_list)

        nx.write_gexf(self.History_graph,'test.gexf')

class CF_Sim_Fix_1(CF_Sim):
    def __init__(self,matrix:np,v_sum):
        self.time_rec = time.time()
        
        self.History_data = Id_Set()
        self.History_list = []
        self.History_graph = nx.DiGraph()
        self.Shape = matrix.shape
        self.Matrix = matrix
        Chip_Firing.__init__(self,self.Matrix)
        self.Init_values = self.Values = self.create_VL(v_sum) # self.Values 注意

        self.show_init()

    def show_init(self):
        print("Init Adj matrix: \n",self.Adj_matrix)
        print("Init Valve:",self.Init_values)
        print("Init graph:")
        print("Valve sum:",self.Init_values.sum())
        print("Degrees:",self.Degrees)
        Tmp = nx.DiGraph(self.Matrix)
        nx.draw(Tmp,with_labels=True)
        plt.savefig('IAM_'+str(self.time_rec)+'.png')
        plt.show()

    def plain_search(self):
        color_dict = {"Red":"#FF0000","Blue":"#1f78b4","Yellow":"#FFFF00","Green":"#33a02c","Grey":"#C0C0C0"}
        def step(val_lst):
            Gid = ','.join([str(x) for x in val_lst])
            for x in range(self.Shape[0]):
                res,flg = self.firing_plain(x,val_lst) #冗余
                if flg and (tuple(res) not in self.History_list):
                    node_id = ','.join([str(x) for x in res])
                    self.History_list.append(tuple(res))
                    if np.all(res <= (self.Degrees-1)):
                        self.History_graph.add_node(node_id,color=color_dict["Grey"])
                        self.History_graph.add_edge(Gid,node_id,toward=x)
                        continue    
                    else:
                        self.History_graph.add_node(node_id,color=color_dict["Green"])
                        self.History_graph.add_edge(Gid,node_id,toward=x)
                    print(Gid,f'--[{x}]->',node_id)
                    #print(self.History_list)
                    step(res)
                    #continue
                elif flg == True:
                    node_id = ' '.join([str(x) for x in res])
                    if np.all(res == self.Init_values):
                        self.History_graph.add_node(node_id,color=color_dict["Yellow"])
                    else:
                        self.History_graph.add_node(node_id,color=color_dict["Red"])
                    self.History_graph.add_edge(Gid,node_id,toward=x)
                    #print(Gid,f'--[{x}]->',node_id)
            #nx.draw(self.History_graph,with_labels=True)
            #plt.show()
        init_id = ','.join([str(x) for x in self.Init_values])
        self.History_list.append(tuple(self.Init_values))
        self.History_graph.add_node(init_id,color=color_dict["Blue"])
        step(self.Init_values)

        color_map = nx.get_node_attributes(self.History_graph,"color").values()
        pos_n = nx.nx_agraph.graphviz_layout(self.History_graph, prog="dot")
        nx.draw_networkx(self.History_graph,node_color=color_map,pos=pos_n,with_labels=True)
        plt.savefig('PST_'+str(self.time_rec)+'.png')
        plt.show()

        print("The number of state:",len(self.History_list))
        print("The state:",self.History_list)

        nx.write_gexf(self.History_graph,'test.gexf')

    def create_VL(self,val_sum): #有待考究
        ret:np.ndarray = CF_Sim.create_VL(self)
        flg = 1 if ret.sum()>val_sum else -1
        while ret.sum()!=val_sum:
            r = np.random.randint(0,self.Shape[0])
            ret[r] -= flg if ret[r]>0 else 0
        return ret
    
class CF_Sim_Fix_1(CF_Sim):
    def __init__(self,matrix:np,v_sum):
        self.time_rec = time.time()
        self.I
        self.History_data = Id_Set()
        self.History_list = []
        self.History_graph = nx.DiGraph()
        self.Shape = matrix.shape
        self.Matrix = matrix
        Chip_Firing.__init__(self,self.Matrix)
        self.Init_values = self.Values = self.create_VL(v_sum) # self.Values 注意

        self.show_init()

    def show_init(self):
        print("Init Adj matrix: \n",self.Adj_matrix)
        print("Init Valve:",self.Init_values)
        print("Init graph:")
        print("Valve sum:",self.Init_values.sum())
        print("Degrees:",self.Degrees)
        Tmp = nx.DiGraph(self.Matrix)
        nx.draw(Tmp,with_labels=True)
        plt.savefig('IAM_'+str(self.time_rec)+'.png')
        plt.show()

    def plain_search(self):
        color_dict = {"Red":"#FF0000","Blue":"#1f78b4","Yellow":"#FFFF00","Green":"#33a02c","Grey":"#C0C0C0"}
        def step(val_lst):
            Gid = ','.join([str(x) for x in val_lst])
            for x in range(self.Shape[0]):
                res,flg = self.firing_plain(x,val_lst) #冗余
                if flg and (tuple(res) not in self.History_list):
                    node_id = ','.join([str(x) for x in res])
                    self.History_list.append(tuple(res))
                    if np.all(res <= (self.Degrees-1)):
                        self.History_graph.add_node(node_id,color=color_dict["Grey"])
                        self.History_graph.add_edge(Gid,node_id,toward=x)
                        continue    
                    else:
                        self.History_graph.add_node(node_id,color=color_dict["Green"])
                        self.History_graph.add_edge(Gid,node_id,toward=x)
                    print(Gid,f'--[{x}]->',node_id)
                    #print(self.History_list)
                    step(res)
                    #continue
                elif flg == True:
                    node_id = ' '.join([str(x) for x in res])
                    if np.all(res == self.Init_values):
                        self.History_graph.add_node(node_id,color=color_dict["Yellow"])
                    else:
                        self.History_graph.add_node(node_id,color=color_dict["Red"])
                    self.History_graph.add_edge(Gid,node_id,toward=x)
                    #print(Gid,f'--[{x}]->',node_id)
            #nx.draw(self.History_graph,with_labels=True)
            #plt.show()
        init_id = ','.join([str(x) for x in self.Init_values])
        self.History_list.append(tuple(self.Init_values))
        self.History_graph.add_node(init_id,color=color_dict["Blue"])
        step(self.Init_values)

        color_map = nx.get_node_attributes(self.History_graph,"color").values()
        pos_n = nx.nx_agraph.graphviz_layout(self.History_graph, prog="dot")
        nx.draw_networkx(self.History_graph,node_color=color_map,pos=pos_n,with_labels=True)
        plt.savefig('PST_'+str(self.time_rec)+'.png')
        plt.show()

        print("The number of state:",len(self.History_list))
        print("The state:",self.History_list)

        nx.write_gexf(self.History_graph,'test.gexf')


class CF_sim_W(CF_Sim_Fix_1):
    def __init__(self,matrix,v_sum):
        self.time_rec = time.time()
        os.mkdir(str(self.time_rec))
        self.Xs = v_sum
        self.History = set()
        self.History_graph = nx.DiGraph()
        self.Shape = matrix.shape
        self.Matrix = matrix
        Chip_Firing.__init__(self,self.Matrix)
        self.Init_values = []

        self.show_init()

    def show_init(self):
        with open(f"{self.time_rec}\info.txt", "w") as f:
            f.write("Init Adj matrix: \n"+str(self.Adj_matrix))
            f.write("\nValve sum:"+str(self.Val_sum))
            f.write("\nDegrees:"+str(self.Degrees))
        #print("Init graph:")
        Tmp = nx.DiGraph(self.Matrix)
        nx.draw(Tmp,with_labels=True)
        plt.savefig(f'{str(self.time_rec)}\IAM.png')
        plt.close()
        #plt.show()

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

    def Plain_search(self,vals:list,draw_option=True):
        '''构建子状态空间树和部分状态空间图'''
        history =[] #子状态空间
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

                    else:
                        if np.all(res == val_lst):
                            history_graph.add_node(node_id,color=color_dict["Yellow"]) #黄色节点代表与初始状态相同的节点
                        else:
                            history_graph.add_node(node_id,color=color_dict["Red"]) #红色节点代表重复的结束节点
                        history_graph.add_edge(gid,node_id,toward=x)
                    #构建子状态空间树过程

                    step(res)

        init_id = ','.join([str(x) for x in vals]) #生成子状态空间树的根节点的标签
        if tuple(vals) not in self.History:
            self.History.add(tuple(vals))
            self.History_graph.add_node(init_id,color=color_dict["Blue"])

        history.add(tuple(vals))
        history_graph.add_node(init_id,color=color_dict["Blue"])
        step(vals)
        #初始化

        color_map = nx.get_node_attributes(history_graph,"color").values() #节点的颜色表
        pos_tree = nx.nx_agraph.graphviz_layout(history_graph, prog="dot") #树状布局
        #nx.draw_networkx方法需要
        if draw_option:
            nx.draw_networkx(history_graph,node_color=color_map,pos=pos_tree,with_labels=True) #绘制子状态空间树
            plt.show()
            plt.close()
        
    def Build_all(self,xs):
        '''构建完整的状态空间图'''
        for x in self.Erg_vals(self.Shape[0],xs):
            self.Plain_search(x)
        color_m = nx.get_node_attributes(self.History_graph,"color").values()
        nx.draw_networkx(self.History_graph,node_color=color_m,with_labels=True)
        plt.savefig(str(self.time_rec)+'\PST_all.png',dpi = 128)
        plt.show()
        nx.write_gexf(self.History_graph,str(self.time_rec)+'\PST.gexf')
        Tmp = 0
        for i,__ in enumerate(nx.connected_components(nx.Graph(self.History_graph))):
            Tmp += 1
        print(f"This Gs has {Tmp} islands.")
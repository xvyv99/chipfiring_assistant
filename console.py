import sys

func_lst = set()
special_lst = set()
# NOTE:由于在类中装饰器无法访问这个变量, 所以直接当全局变量了

def comm_conversion(comm):
    comm_sp = comm.split(' ')
    (main_p,*para) = comm_sp #整个命令解包为prog-主命令和para-参数
    para_0 = [x for x in para if x != ''] #去掉所有的无效项''
    return main_p,para_0

def para_reorganize(param):
    '''
    参数重整, 输入中的ABC,"ABC"或是'ABC'都算字符串"ABC"
    '''
    para_refresh = []
    symbol_left = ("\'","\"","[","(","{")
    symbol_right = ("\'","\"","]",")","}")

    for p in param:
        if p.isnumeric(): #数字转换
            para_refresh.append(int(p))
        elif(p[0] in symbol_left):
            i = symbol_left.index(p[0])
            if p[-1] == symbol_right[i] and len(p)>1:
                namespace = {}
                exec(f"Tmp = {p}",namespace)
                para_refresh.append(namespace['Tmp']) #此处取巧, 直接使用python实现
            else:
                para_refresh.append(p)

        elif p == "True":
            para_refresh.append(True)
        elif p == "False":
            para_refresh.append(False)
        else:
            para_refresh.append(p)
    return para_refresh
    #Allow the user to enter string without quotes.

class Console:
    """
    为程序创建一个命令行环境
    下一步加入选项参数
    """
    __Version = "Console beta 0.22"
    def __init__(self):
        self.PROMPT = "> " #命令行提示符
        self.WELCOME = "Welcome to Console. Get help by typing \"Help\" in this console." #欢迎文本
        self.VERSION = "Console beta 0.2"
        self.HELP_HEAD = f"Powered by {self.__Version}"
        #字符串常量

        self.Var_lst = {}
        #控制台命令列表

        self.Func_lst = {func.__name__:func for func in func_lst}
        self.Special_lst = {func.__name__:func for func in special_lst}
        #注册的命令函数字典, key:函数的名称, value:函数本身

    def RegisterS(func):
        '''
        装饰器, 用于注册特殊命令函数
        '''
        special_lst.add(func)
        return func

    def RegisterC(func):
        '''
        装饰器, 用于注册命令函数
        '''
        func_lst.add(func)
        return func

    def Analyzer(self):
        '''命令行环境模拟'''
        invaild_input = ["",None]
        while True:
            get = self.Get_command() #获取用户输入, 并去掉字符串头尾的空格
            if get in invaild_input:
                continue
            elif self.S_Exec(get):
                continue
            else:
                try: #检测键盘Ctrl+C中断
                    self.F_Exec(get)
                except KeyboardInterrupt:
                    print("[Console]Execute interruptions.")
                    continue

    def Get_command(self)->str:
        '''获取用户有效输入'''
        ret = input(self.PROMPT).strip()
        return ret

    def S_Exec(self,comm):
        '''
        特殊类命令的解析与执行
        '''
        (req,expr) = comm_conversion(comm)
        if req in self.Special_lst:
            self.Special_lst[req](self,expr)
            return True
        else:
            return False

    def F_Exec(self,comm):
        '''
        函数类命令的解析与执行
        '''
        def try_exec(program,param):
            '''用于处理命令执行中会出现的错误'''
            try:
                ret = self.Func_lst[program](self,*param)
            except TypeError as e:
                print(e)
                param_str = ','.join([str(x) for x in param]) #参数的字符串
                print(f"[Error]{program}({param_str}):Argument error.")
            else:
                return ret

        (prog,para_1) = comm_conversion(comm)

        if prog in self.Func_lst:
            para_re = para_reorganize(para_1)
            r = try_exec(prog,para_re)
            print("[Return]",r)
        else:
            self.NotFound(prog)

    @RegisterS
    def SET(self,expr):
        pass

    @RegisterS
    def ECHO(self,expr_lst):
        expr = ' '.join(expr_lst)
        try:
            namespace = {}
            exec(f"Tmp = {expr}",namespace)
        except Exception as e:
            print(e)
            print(f"[Error]\'{expr}\':Expression error.")
        else:
            print(namespace['Tmp'])

    def Welcome(self):
        '''开始的欢迎界面'''
        print(self.WELCOME)

    @RegisterC
    def RegComm(self):
        '''查看注册的命令函数'''
        print(self.Func_lst)

    @RegisterC
    def Version(self):
        '''显示程序版本信息'''
        print(self.VERSION)

    @RegisterC
    def Exec(self,comm):
        return exec(comm)

    @RegisterC
    def Echo(self,var_name:str):
        print(self.Var_lst[var_name])

    @RegisterC
    def Help(self,prog=None):
        if prog is None:
            print(self.HELP_HEAD)
            print("These shell commands are defined internally.  Type 'help' to see this list.")
            print("Type 'Help name' to find out more about the function 'name'.")
            #print("Use 'info bash' to find out more about the shell in general.")
            print("Command list:",self.Func_lst.keys())
        elif prog in self.Func_lst:
            print(self.Func_lst[prog].__doc__)
        else:
            print(f"[Error]Help: no help topics match\'{prog}\'.")

    @RegisterC
    def Hello(self):
        print("Hello World!")

    def NotFound(self,command):
        print(f"[Error]\'{command}\': command not found.")

    def Choice(self,msg):
        '''让用户选择Yes/No'''
        result = input(msg+" [Y/n] ")
        if(result in ('y','Y')):
            return True
        else:
            print("Abort.")
            return False

    @RegisterC
    def Quit(self):
        msg = "Do you really want to quit?"
        if self.Choice(msg):
            print("Bye.")
            sys.exit(0)
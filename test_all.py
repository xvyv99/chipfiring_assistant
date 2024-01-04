import pytest

import numpy as np
import graph,chip_firing,console,interact

class Test_graph:
    def setup_class(self):
        self.Test_matrix_1_ = np.array([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]])
        self.Test_matrix_2_ = np.array(np.mat("1 0 1;0 1 0;1 0 1"))
        self.Test_actual_1_ = graph.UGraph(self.Test_matrix_1,[])
        self.Test_actual_2_ = graph.UGraph(self.Test_matrix_1,[])

if __name__ == '__main__':
    pytest.main(["-s", "__file__"])
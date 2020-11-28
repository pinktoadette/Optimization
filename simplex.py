import numpy as np

class Simplex():
    def __init__(self, A, b, c, min_max, lb=0, ub=10000):  
        self.min_max = min_max
        self.A = A
        self.b = b
        self.c = c
        self.track = list()
        
        self._add_slack()
        self._set_objective()
    
    def _set_objective(self):
        if(self.min_max == "min"):
            for i in range(len(self.c)):
                self.c[i] = -1 * self.c[i]
    
    def _add_slack(self):
        num_x = len(self.c)
        num_slack = len(self.A)
        
        tbl1 = np.hstack(([None], [0], self.c, [0] * num_slack))
        basis = np.array([0] * num_slack)
        
        for i in range(0, len(basis)):
            basis[i] = num_slack + i
        A = self.A
        
        if (num_slack + num_x) != len(self.A[0]):
            slack = np.eye(num_slack)
            A = np.hstack((self.A, slack))
            
        tbl2 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))        
        self.tableau = np.array(np.vstack((tbl1, tbl2)), dtype ='float')
        self._track_pivot(pivot={})
        
    def _track_pivot(self, pivot):
        t = self.tableau.copy()
        result = {'tableau':t, 'pivot': pivot }
        self.track.append(result)

    def _find_basis(self):
        list_point = self.tableau[0, 2:].tolist()
        if self.min_max == "max":
            c = list_point.index(np.amax(self.tableau[0, 2:])) + 2
        else:
            c = list_point.index(np.amin(self.tableau[0, 2:])) + 2

        minimum = float('inf')
        r = -1

        for i in range(1, len(self.tableau)): 
            if(self.tableau[i, c] > 0):
                ratio_test = self.tableau[i, 1]/self.tableau[i, c]
                if ratio_test<minimum: 
                    minimum = ratio_test 
                    r = i

        pivot = self.tableau[r, c] 
        self.tableau[r, 1:] /= pivot 
        self._row_echelon(r, c)
        self._track_pivot({'pivot_row':r, 'pivot_col':c, 'pivot_at':self.tableau[r,c] } )
        
    def _row_echelon(self, row, col):
        for r in range(len(self.tableau)):
            if r != row:
                mult = self.tableau[r, col] / self.tableau[row, col]
                self.tableau[r, 1:] -= mult * self.tableau[row, 1:] 
    
    def _check(self):
        for x in self.tableau[0, 2:]:
            if (self.min_max == "min" and x < 0) or (self.min_max == "max" and x > 0):
                return True
        return False
    
    def solve(self):
        while self._check():
            self._find_basis()
        return self.track

if __name__ == "__main__":

    c = [-10, -12, -12]
    b = [20,20,20]
    A = [
        [1,2,2],
        [2,1,2],
        [2,2,1]
    ]

    s = Simplex(A,b,c,"min")
    ### This is written this way to send to an api
    ### for frontend JS to display
    pivot1 = s.solve()
    print (pivot1[len(pivot1)-1]['tableau'])

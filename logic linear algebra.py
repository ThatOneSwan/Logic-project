from pyDatalog import pyDatalog
import numpy as np
import numpy.linalg as la

#give relations and constants, so suppose we have three relations and 5 constants
pyDatalog.create_terms('X, Y, W, Z,r1, r2, r3')
#constants = ['e1', 'e2', 'e3', 'e4', 'e5']
+ r1('e1','e2')
+ r1('e4', 'e2')
+ r1('e5', 'e3')
+ r1('e2', 'e5')



+r3('e1', 'e4')
+r3('e3', 'e4')
+r3('e2', 'e3')
+r3('e5', 'e3')

r2(X,Z) <= r1(X, Z)
r2(X,Z) <= r1(X,Y) & r2(Y, W) & r3(W, Z)

_r1 = set()
_r3 = set()

_r1.add(('e1', 'e2'))
_r1.add(('e4', 'e2'))
_r1.add(('e5', 'e3'))


_r3.add(('e1', 'e4'))
_r3.add(('e5', 'e1'))
_r3.add(('e2', 'e3'))

#processes constants
constants = set.union(_r1, _r3)
constants = list(constants)
constants1, constants2 = zip(*constants)
constants1 = list(set(constants1+constants2))
constants1 = sorted(constants1)
print constants1

vectors= []

#constructs vectors for each constant
for i in range(0, len(constants1)):
    a = np.zeros((len(constants1), 1))
    a[i] = 1
    b = a
    vectors.append(b)
#constructs matrices for given relations
R1 = np.zeros((len(constants1), len(constants1)))
R3 = np.zeros((len(constants1), len(constants1)))

for i in range(0, len(constants1)):
    for j in range(0, len(constants1)):
        R1[i, j] = bool((constants1[i], constants1[j]) in _r1)


for i in range(0, len(constants1)):
    for j in range(0, len(constants1)):
        R3[i, j] = bool((constants1[i], constants1[j]) in _r3)

#these are the matrices it outputs
print "matrix R1 corresponding to r1(,):"
print R1
print "matrix R3 corresponding to r3(,):"
print R3

#we wish to solve for R2 using an iterative gradient descent algorithm,

#the function below is define to solve equations of the form R2 = R1 + R1*R2*R3
#the general equation sylvester equation is A*X*B +X = C
#we can reformat the equation A*X*B - C = -X => X = C -(A*X*B)
#explain
def iterative_gradient_solver(A,B, C, k):
    #initialization X as an NxN matrix, where is the number of constants and the values for every element in the matrix
    # is 10^-6

    Xnew = 10**(-6) * np.ones(np.shape(A))
    for i in range(0, k):
        X = Xnew
        mu = (((la.norm(A))**2)*(la.norm(B)**2) + 1)**-1
        X1 = X + mu*A.T.dot(C - A.dot(X).dot(X) - X).dot(B.T)
        X2 = X + mu*(C-A.dot(X).dot(B)-X)
        Xnew = 0.5 * (X1+X2)

    return Xnew

l = iterative_gradient_solver(R1, R3, R1, 100)
#need to round off matrix elements to bring them to 1,0
l = np.round(l)
#matrix
print "Matrix R2 corresponding to r2(,)"
print l

R2 = l

_r2 = set()

for i in range(0, np.shape(R2)[0]):
    for j in range(0, np.shape(R2)[0]):
        if R2[i, j] == 1:
            _r2.add((constants1[i], constants1[j]))

print "pairs of constants satisfying r2(,):"
print(_r2)



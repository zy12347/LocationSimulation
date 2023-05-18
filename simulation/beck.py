# Beck's method
import numpy

def beck(s, r, noise):
    N = s.shape[0]
    C = noise**2*numpy.eye(N)
    W = numpy.linalg.cholesky(2*numpy.multiply(C, C) + 4*numpy.multiply(r*r.T, C)).I
    Am1 = numpy.mat(2*s)
    Am2 = numpy.ones([N, 1])
    Am = W*numpy.concatenate([Am1, -Am2], axis=1)#A
    b = numpy.zeros([N, 1])
    for i in range(N):
        b[i, :] = numpy.dot(s[i, :], s[i, :]) - r[i]**2
    bm = W*b#b
    P = numpy.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])#D
    Pm = numpy.mat(P)
    q = [0, 0, -0.5]#f
    qm = numpy.mat(q).T

    def fun(lam):
        if numpy.linalg.det(Am.T*Am + lam*Pm) == 0:
            f = numpy.inf
        else:
            z = numpy.linalg.lstsq(Am.T*Am+lam*Pm, Am.T*bm-lam*qm, rcond=-1)[0]
            f = z.T*Pm*z + 2*qm.T*z
        return f

    # zero finder by bisection
    # NOTE: assume monotonic decreasing function
    def bisect(a,b,tol):
        funa = numpy.inf
        funb = fun(b)
        # if root is not within the interval, double the interval length and move it to the right
        if funb > 0:
            while funb > 0:
                oldb = b
                b = b + 2*(b-a)
                a = oldb
                funa = funb
                funb = fun(b)
        # bisection
        while (b-a) > tol:
            # pick bisection point
            if numpy.isfinite(funa):
                # regula falsi
                p_rf = a + funa*(b-a)/(funa - funb)
                # interval midpoint
                p_mid = (a+b)/2
                # weighted average of the midpoint and regula falsi point
                bsfac = 0.95
                p = bsfac*p_rf + (1-bsfac)*p_mid
                p = p[0, 0]
            else:
                p = (a+b)/2
            fp = fun(p)
            if fp==0:
                # find the zero
                t = p
                return t
            elif fp > 0:
                # discard the leftmost point
                a = p
                funa = fp
            else:
                # discard the rightmost point
                b = p
                funb = fp
        t = (a+b)/2
        return t

    eigvalue, eigvector = numpy.linalg.eig(numpy.linalg.cholesky(Am.T*Am).I*Pm*numpy.linalg.cholesky(Am.T*Am).I)
    starting_lam = -1/max(eigvalue)
    lam = bisect(starting_lam, 0, 1e-12)
    theta = numpy.linalg.solve(Am.T*Am + lam*Pm, Am.T*bm - lam*qm)
    sol = [theta[0,0], theta[1,0]]
    return sol
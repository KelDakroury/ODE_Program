import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


root = tk.Tk()
root.wm_title("Embedding in Tk")


class DENumericalMethod:
    def __init__(self, derivative_expr):
        self.derivative_expr = lambda x,y: derivative_expr(y, x)

    def compute(self, x0, y0, x_limit, step):
        raise NotImplementedError('Override this method in child classes.')


class Euler(DENumericalMethod):
    def compute(self, x0, y0, x_limit, step):
        x = x0
        y = y0
        xs = [x]
        ys = [y]

        while x < x_limit:
            y += self.derivative_expr(y, x) * step
            x += step
            xs.append(x)
            ys.append(y)

        return xs, ys


class ImprovedEuler(DENumericalMethod):
    def compute(self, x0, y0, x_limit, step):
        x = x0
        y = y0
        xs = [x]
        ys = [y]
        while x < x_limit:
            k1 = self.derivative_expr(y, x)
            k2 = self.derivative_expr(y + step * k1, x + step)
            x += step
            y += step / 2 * (k1 + k2)
            xs.append(x)
            ys.append(y)

        return xs, ys


class RungeKutta(DENumericalMethod):
    def compute(self, x0, y0, x_limit, step):
        x = x0
        y = y0
        xs = [x]
        ys = [y]

        while x < x_limit:
            k1 = self.derivative_expr(y, x)
            k2 = self.derivative_expr(y + step * k1 / 2, x + step / 2)
            k3 = self.derivative_expr(y + step * k2 / 2, x + step / 2)
            k4 = self.derivative_expr(y + step * k3, x + step)

            y += (k1 + k2 * 2 + k3 * 2 + k4) * step / 6
            x += step

            xs.append(x)
            ys.append(y)

        return xs, ys


class Exact:
    def __init__(self, exact_expr, ivp_constant):
        self.exact_expr = exact_expr
        self.ivp_constant = ivp_constant

    def compute(self, x0, y0, x_limit, step):
        C = self.ivp_constant(y0, x0)
        x = x0
        xs = [x]
        ys = [self.exact_expr(x, C)]

        while x < x_limit:
            x += step
            xs.append(x)
            ys.append(self.exact_expr(x, C))

        return xs, ys



class equation:
    def __init__(self, x0, y0, X, N):
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.X = int(X)
        self.N = int(N)
    def draw(self):
        x = np.linspace(self.x0, self.X, self.N)
        # c = - (2 * C / (C + x(0) ** (1 / 3)) - 4) / (2 * x(0))

        y_exact = lambda x, C: - (2 * C / (C + x ** (1 / 3)) - 4) / (2 * x)
        c = 0
        y = y_exact(x, c)

        plt.ion()

        fig = plt.figure()
        fig.add_subplot(111).plot(x, y, '-r')
        fig.canvas.draw()
        fig.canvas.flush_events()

class Approximations(equation):
    def __init__(self, x0, y0, X, N):
        super().__init__(x0, y0, X, N)

    def draw(self):

        euler = Euler(df)
        step = (self.X - self.x0)/self.N
        x, y = euler.compute(self.x0, self.y0, self.X, step)
        x = np.array(x[:-1])
        y = np.array(y[:-1])

        '''x = [self.x0]
        y = [self.y0]
        dy = [2]  # dy(0)
        for i in range(1, int(self.N)):
            x.append(x[i-1]+step)
            y.append(y[i-1] + step*dy[i-1])
            dy.append(df(x[i], y[i]))'''
        plt.ion()
        tx = np.linspace(self.x0, self.X+1, self.N+1)
        tc = 1.0/(self.y0 - np.exp(self.x0)) + self.x0
        ty = 1.0/(tc - tx) + np.exp(tx)

        le = np.abs(ty-y)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x, y)
        axs[0, 0].set_title('euler')

        axs[0, 1].plot(x, le, 'tab:orange')
        axs[0, 1].set_title('local error')

        g = []
        for i in range(1, int(self.N)):
            x = [self.x0]
            y = [self.y0]
            dy = [df(self.x0, self.y0)] #[2]  # dy(0)
            step = (self.X-self.x0)/i
            for i in range(1, int(i)):
                x.append(x[i-1]+step)
                y.append(y[i-1] + step*dy[i-1])
                dy.append(df(x[i], y[i]))

            tx = np.linspace(self.x0, self.X, i+1)
            tc = 1.0/(self.y0-np.exp(self.x0))+self.x0
            ty = 1.0/(tc-tx) + np.exp(tx)
            le = np.abs(ty-y)
            g.append(max(le))

        axs[1, 0].plot(np.linspace(self.x0, self.X, self.N-1), g, 'tab:green')
        axs[1, 0].set_title('global error')

        axs[1, 1].remove()

        fig.canvas.draw()
        fig.canvas.flush_events()



def df(x, y):
    return -y**2 / 3 - 2/(3 * x**2)
    # return np.exp(2*x) + np.exp(x) + y**2 - 2*y*np.exp(x)

def improvedEulers(x0, y0, X, N):
    improvedEuler = ImprovedEuler(df)
    x, y = improvedEuler.compute(x0, y0, X, (X-x0)/N)
    x = np.array(x[:-1])
    y = np.array(y[:-1])
    '''print(x0, y0, X, N)
    x = [x0]
    y = [y0]
    dy = [df(x0, y0)]
    dy1 = [df(x0, y0)]
    step = (X-x0)/N
    for i in range(1, int(N)):
        x.append(x[i-1]+step)
        y.append(y[i-1] + step/2*(dy[i-1] + dy1[i-1]))
        dy.append(df(x[i], y[i]))
        x1 = x[i]+step
        y1 = y[i-1]+step*dy[i-1]
        dy1.append(df(x1, y1))
'''
    tx = np.linspace(int(x0), int(X), int(N)+1)
    tc = 1.0/(y0-np.exp(x0)) + x0
    ty = 1.0/(tc-tx) + np.exp(tx)
    le = np.abs(ty-y)

    plt.ion()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('improved Eulers')

    axs[0, 1].plot(x, le, 'tab:orange')
    axs[0, 1].set_title('local error')

    '''g = []
    for i in range(1, int(N+1)):
        x = [x0]
        y = [y0]
        dy = [df(x0, y0)]  # dy(x0,y0)
        dy1 = [df(x0, y0)]
        step = (X-x0)/i
        for i in range(1, int(i)):
            x.append(x[i-1]+step)
            y.append(y[i-1] + step/2*(dy[i-1] + dy1[i-1]))
            dy.append(df(x[i], y[i]))
            x1 = x[i]+step
            y1 = y[i-1]+step*dy[i-1]
            dy1.append(df(x1, y1))

        tx = np.linspace(x0, X, i+1)
        tc = 1.0/(y0-np.exp(x0))+x0
        ty = 1.0/(tc-tx) + np.exp(tx)
        le = np.abs(ty-y)
        g.append(max(le))'''
    g = [abs(ex - act) for ex, act in zip(exact, y)]
    print(g)

    axs[1, 0].plot(np.linspace(1, N, N), g, 'tab:green')
    axs[1, 0].set_title('global error')

    axs[1, 1].remove()

    fig.canvas.draw()
    fig.canvas.flush_events()


def rungeKutta(x0, y0, X, N):
    x0 = int(x0)
    X = int(X)
    N = int(N)
    runge = RungeKutta(df)
    x, y = runge.compute(x0, y0, X, (X-x0)/N)
    x = np.array(x[:-1])
    y = np.array(y[:-1])
    X1 = np.linspace(int(x0), int(X), int(N)+1)
    '''x_cur = x0
    y_cur = x0
    Y.append(y_cur)
    h = (X-x0)/(N)
    for i in range(int(N)):
        k1 = df(x_cur, y_cur)
        k2 = df(x_cur+h/2, y_cur+h/2*k1)
        k3 = df(x_cur+h/2, y_cur+h/2*k2)
        k4 = df(x_cur+h, y_cur+h*k3)
        y_cur = y_cur+h/6*(k1+2*k2+2*k3+k4)
        Y.append(y_cur)
        x_cur += h
    Y = np.array(Y)'''
    tx = np.linspace(x0, X, N+1)
    tc = 1.0/(y0-np.exp(x0))+x0
    ty = 1.0/(tc-tx) + np.exp(tx)
    le = np.abs(ty-y)

    plt.ion()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('rungeKutta')

    axs[0, 1].plot(X1, le, 'tab:orange')
    axs[0, 1].set_title('local error')

    g = []
    for i in range(1, int(N+2)):
        X1 = np.linspace(x0, X, i+1)
        Y = []
        x_cur = x0
        y_cur = x0
        Y.append(y_cur)
        h = (X-x0)/(i)
        for i in range(int(i)):
            k1 = df(x_cur, y_cur)
            k2 = df(x_cur+h/2, y_cur+h/2*k1)
            k3 = df(x_cur+h/2, y_cur+h/2*k2)
            k4 = df(x_cur+h, y_cur+h*k3)
            y_cur = y_cur+h/6*(k1+2*k2+2*k3+k4)
            Y.append(y_cur)
            x_cur += h
        Y = np.array(Y)

        tx = np.linspace(x0, X, i+2)
        tc = 1.0/(y0-np.exp(x0))+x0
        ty = 1.0/(tc-tx) + np.exp(tx)
        le = np.abs(ty-Y)
        g.append(max(le))
    X1 = np.linspace(x0, X, N+1)
    axs[1, 0].plot(np.linspace(1, N, N+1), g, 'tab:green')
    axs[1, 0].set_title('global error')

    axs[1, 1].remove()

    fig.canvas.draw()
    fig.canvas.flush_events()


def _update(_class):
    x0 = -1
    y0 = -1
    X = -1
    N = -1
    try:
        if e_x0.get():
            x0 = float(e_x0.get())
        if e_y0.get():
            y0 = float(e_y0.get())
        if e_X.get():
            X = float(e_X.get())
        if e_N.get():
            N = float(e_N.get())
        print("x=%d fd=%d sf=%d fsd=%d", x0, y0, X, N)
    except ValueError:
        l_status.config(text="Error")
        return False
    e = _class(x0, y0, X, N)
    e.draw()
    l_status.config(text=" ")

def _update2(_fun):
    x0 = -1
    y0 = -1
    X = -1
    N = -1
    try:
        if e_x0.get():
            x0 = float(e_x0.get())
        if e_y0.get():
            y0 = float(e_y0.get())
        if e_X.get():
            X = float(e_X.get())
        if e_N.get():
            N = float(e_N.get())
        print("x=%d fd=%d sf=%d fsd=%d", x0, y0, X, N)
    except ValueError:
        l_status.config(text="Error")
        return False
    _fun(x0,y0,X,N)
    l_status.config(text=" ")


def _exact():
    _update(equation)


def _euler():
    _update(Approximations)


def _improvedEulers():
    _update2(improvedEulers)


def _rungeKutta():
    _update2(rungeKutta)


b_quit = tk.Button(master=root, text="Exact", command=_exact)
b_quit.pack(side=tk.RIGHT)
b_update = tk.Button(master=root, text="Euler", command=_euler)
b_update.pack(side=tk.RIGHT)
b_update = tk.Button(master=root, text="ImprovedEulers",
                     command=_improvedEulers)
b_update.pack(side=tk.RIGHT)
b_update = tk.Button(master=root, text="RungeKutta", command=_rungeKutta)
b_update.pack(side=tk.RIGHT)

l_x0 = tk.Label(root, text="x0")
l_x0.pack(side=tk.LEFT)
e_x0 = tk.Entry(root)
e_x0.pack(side=tk.LEFT)

l_y0 = tk.Label(root, text="y0")
l_y0.pack(side=tk.LEFT)
e_y0 = tk.Entry(root)
e_y0.pack(side=tk.LEFT)

l_X = tk.Label(root, text="X")
l_X.pack(side=tk.LEFT)
e_X = tk.Entry(root)
e_X.pack(side=tk.LEFT)

l_N = tk.Label(root, text="N")
l_N.pack(side=tk.LEFT)
e_N = tk.Entry(root)
e_N.pack(side=tk.LEFT)

l_status = tk.Label(root, text="     ")
l_status.pack(side=tk.LEFT)


tk.mainloop()

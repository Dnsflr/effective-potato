import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
plt.rcParams['toolbar'] = 'none'



#CONSTANTS
a = 1.0
b = 1.0
c = 0.2
freq_ratio = 0.213
omega = freq_ratio * 2 * np.pi
y0 = [0.0, 0.15]
physics_args = (a, b, c, omega)

# Sim settings
t_span = [0, 300] #Time span
num_points = 2000 
t_eval = np.linspace(t_span[0], t_span[1], num_points)



#Fun for duffing system equations
def duffing_system(t, y, f, physics_touple):
    x, v = y
    a, b, c, omega = physics_touple
    
    dxdt = v
    dvdt = b * x - a * x**3 - c * v + f * np.cos(omega * t)
    return [dxdt, dvdt]

#Fun for computing solution for each f value (kinda pain in the ass )
def compute_solution(f_val, t_eval, y0, args_tuple):
    sol = solve_ivp(
        fun=duffing_system,
        t_span=[t_eval.min(), t_eval.max()],
        y0=y0,
        t_eval=t_eval,
        args=(f_val, args_tuple), #f_val is passing here, i have tried deleting args touple from here since its global but idk (i know its stupid but cant figoure out)
        method='RK45' #runge kutta supremancy
    )
    return sol.t, sol.y[0], sol.y[1]

# Sim settings
t_span = [0, 300] #Time span
num_points = 2000 
t_eval = np.linspace(t_span[0], t_span[1], num_points)


# Plotting settings
fig, ax = plt.subplots(figsize=(10, 9), facecolor = 'black')
plt.subplots_adjust(bottom=0.25)
ax = plt.gca()
ax.set_facecolor("black")

#INITIAL STATE 
f_initial = 0.2
t, x, v = compute_solution(f_initial, t_eval, y0, physics_args)
ax.set_title(f'Interactive Duffing atractor (f = {f_initial:.2f})', fontsize=16, color= 'white')

# Points of equilibrium (kinda of)
x_eq_unstable = 0.0
x_eq_stable_1 = np.sqrt(b / a)
x_eq_stable_2 = -np.sqrt(b / a)
v_eq = 0.0

ax.plot(x_eq_unstable, v_eq, 'x', color='azure', markersize=12, mew=2, 
        label=f'unstable equilibrium (0, 0)')
ax.plot([x_eq_stable_1, x_eq_stable_2], [v_eq, v_eq], 'o', color='lightblue', 
        markersize=10, label=f'stable equilibrium (±1, 0)')

# Dynamical elements ? why ,
background_line, = ax.plot(x, v, lw=0.5, alpha=0.3, color='whitesmoke', label='Whole attractor trajectory')
line, = ax.plot([], [], lw=2, color='darkslategray', label='time evolution trajectory')
point, = ax.plot([], [], 'o', color='deepskyblue', markersize=8, label='actual position')

ax.set_xlabel('Position (x)', color = 'white')
ax.set_ylabel('Velocity (v)', color = 'white')
ax.tick_params(axis='x', labelcolor='white')
ax.tick_params(axis='y', labelcolor='white')
ax.legend(loc='upper right', facecolor='slategray')
ax.grid(True)

# Setting whole plot bettwen limits
_, x_safe, v_safe = compute_solution(0.5, t_eval, y0, physics_args)
ax.set_xlim(x_safe.min() * 1.1, x_safe.max() * 1.1)
ax.set_ylim(v_safe.min() * 1.1, v_safe.max() * 1.1)

# Slider for amplitude 'f'
ax_slider_f = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
f_slider = Slider(
    ax=ax_slider_f,
    label='Amplitude (f)',
    valmin=0.1,   
    valmax=0.5,   
    valinit=f_initial,
    valstep=0.01
)
f_slider.label.set_color('white')

#Fun for anims
def init_animation():
    #cleaning shit
    line.set_data([], [])
    point.set_data([], [])

    return line, point

def animate_frame(i):
    # in each iteraztion we are using diffreng x and v values from precomputed arrays
    line.set_data(x[:i], v[:i])
    point.set_data([x[i]], [v[i]])
    return line, point


anim_object = FuncAnimation(
    fig, 
    animate_frame, 
    init_func=init_animation,
    frames=num_points, 
    interval=10,      # tim ebeetwen frames in ms
    blit=True         # for optimization (idk)
)

def update_f(val):
    global t, x, v, anim_object 
    
    new_f = f_slider.val
    
    #1 Stop the aimation
    if anim_object is not None:
        anim_object.event_source.stop()

    #2 Compute new values
    t, x, v = compute_solution(new_f, t_eval, y0, physics_args)
    
    #3 Ubdate 
    background_line.set_data(x, v)
    ax.set_title(f'Interactive Duffing atractor (f = {new_f:.2f})', fontsize=16)

    #4 create and run NEW sim
    anim_object = FuncAnimation(
        fig, 
        animate_frame, 
        init_func=init_animation,
        frames=num_points, 
        interval=10, 
        blit=True
    )
    
    #5 revresh the view
    fig.canvas.draw_idle()


f_slider.on_changed(update_f)


plt.get_current_fig_manager().full_screen_toggle()


plt.show()

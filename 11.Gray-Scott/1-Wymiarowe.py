import numpy as np
import matplotlib.pyplot as plt

L = 2.0
N = 100
dx = 0.02
dt = 1.0
T_max = 10000

Du = 2e-5
Dv = 1e-5
F = 0.025
k = 0.055

u = np.ones(N)
v = np.zeros(N)

N1 = N // 4
N2 = 3 * N // 4
u[N1:N2+1] = 0.5 + np.random.rand(N2 - N1 + 1) * 0.02 
u[N1:N2+1] = 0.4 + np.random.rand(N2 - N1 + 1) * 0.2
v[N1:N2+1] = 0.2 + np.random.rand(N2 - N1 + 1) * 0.2

history_u = []
save_interval = 10

plot_times = [0, 500, 1500, 2500, 10000]
plots_data = {}

steps = int(T_max / dt)

for step in range(steps + 1):
    current_time = step * dt
    
    if current_time in plot_times:
        plots_data[current_time] = (u.copy(), v.copy())
    
    if step % save_interval == 0:
        history_u.append(u.copy())
    
    #Laplasjan
    lap_u = (np.roll(u, 1) - 2*u + np.roll(u, -1)) / (dx**2)
    lap_v = (np.roll(v, 1) - 2*v + np.roll(v, -1)) / (dx**2)
    
    uvv = u * v**2
    du = Du * lap_u - uvv + F * (1 - u)
    dv = Dv * lap_v + uvv - (F + k) * v
    
    u += du * dt
    v += dv * dt

fig, axes = plt.subplots(len(plot_times), 1, figsize=(8, 12), sharex=True)
if len(plot_times) == 1: axes = [axes] 

for i, t in enumerate(plot_times):
    u_t, v_t = plots_data[t]
    axes[i].plot(u_t, label='u', color='blue')
    axes[i].plot(v_t, label='v', color='red')
    axes[i].set_title(f'Czas t = {t}')
    axes[i].set_ylabel('Stężenie')
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

plt.xlabel('x (indeks)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(history_u, aspect='auto', extent=[0, N, T_max, 0], cmap='viridis')
plt.colorbar(label='Stężenie u')
plt.xlabel('Położenie x')
plt.ylabel('Czas t')
plt.title('Ewolucja czasoprzestrzenna u(x,t)')
plt.show()
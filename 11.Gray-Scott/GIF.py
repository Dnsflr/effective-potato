import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# F = 0.0545
# k = 0.0620

# F = 0.0367
# k = 0.0640

F = 0.025
k = 0.055

N = 100
dx = 0.02
dt = 1.0
Du, Dv = 2e-5, 1e-5

u = np.ones((N, N))
v = np.zeros((N, N))

N1, N2 = N // 4, 3 * N // 4
u[N1:N2, N1:N2] = 0.5 + np.random.rand(N2-N1, N2-N1) * 0.02
v[N1:N2, N1:N2] = 0.25 + np.random.rand(N2-N1, N2-N1) * 0.02

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
im = ax.imshow(u, cmap='Spectral', interpolation='bicubic', vmin=0, vmax=1)
ax.set_title(f"Gray-Scott: F={F}, k={k}")
ax.axis('off')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')

steps_per_frame = 40 

def update(frame):
    global u, v
    for _ in range(steps_per_frame):
        lu = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
              np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx**2)
        lv = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
              np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx**2)
        
        uvv = u * v**2
        du = Du * lu - uvv + F * (1 - u)
        dv = Dv * lv + uvv - (F + k) * v
        
        u += du * dt
        v += dv * dt
        
    im.set_array(u)
    time_text.set_text(f'Kroki: {frame * steps_per_frame}')
    return [im, time_text]

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

print("Generowanie GIF-a... to może chwilę potrwać.")
ani.save('gray_scott_simulation.gif', writer='pillow', fps=20)
print("Gotowe! Zapisano jako 'gray_scott_simulation.gif'.")
plt.show()
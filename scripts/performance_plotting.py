import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import host_subplot

# Data
params = [39.13, 7.22, 13.46]
pq_short = [53.4, 52.1, 53.7]
pq_long = [31.4, 28.9, 29.8]
inf_time = [71.64, 64.12, 65.75]
grupos  = ['PowerBEV', 'Ours (tiny)', 'Ours']
colors = ['red', 'green', 'blue']



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np



# Posiciones de las barras en el eje x
posiciones = np.arange(len(grupos))

# Tamaño de la figura
plt.figure(figsize=(10, 6))

# Crear ejes
host = host_subplot(111)
par = host.twinx()  # Eje adicional

# Dibujar las barras del eje izquierdo
p1 = host.bar(posiciones - 0.25, params, width=0.2, align='center', label='Parameters')

# Dibujar las barras del eje derecho
p3 = host.bar(posiciones, pq_long, width=0.2, align='center', label='VPQ long range', color='green')
p2 = par.bar(posiciones + 0.25, pq_short, width=0.2, align='center', label='VPQ short range', color='orange')


# Configurar etiquetas y título
plt.xticks(posiciones, grupos, fontsize=16)
# host.set_xlabel('Grupos')
host.set_ylabel('Parameters (M)', fontsize=16)
par.set_ylabel('VPQ (%)', fontsize=16)
par.set_ylim(30, 80)

# Ajustar tamaño de las etiquetas de los ejes
host.tick_params(axis='both', which='major', labelsize=12)
par.tick_params(axis='both', which='major', labelsize=12)



# Combinar leyendas
lines = [p1, p2, p3]
host.legend(lines, [line.get_label() for line in lines], loc='upper right')

plt.savefig('grafico_de_barras.png', dpi=400)

# Mostrar el gráfico
plt.show()



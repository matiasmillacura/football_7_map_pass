import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Cargar el archivo Excel
archivo_excel = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\codes\\posiciones.xlsx"  # Cambia por la ruta de tu archivo
df = pd.read_excel(archivo_excel)

# Verificar columnas
print("Columnas del DataFrame:", df.columns)

# Asegurarse de que las columnas existen y limpiar NaN
if 'Ball X' in df.columns and 'Ball Y' in df.columns:
    df = df.dropna(subset=['Ball X', 'Ball Y'])
else:
    raise ValueError("Las columnas 'Ball X' o 'Ball Y' no están en el archivo Excel.")

# Reiniciar índice para evitar problemas
df.reset_index(drop=True, inplace=True)

# Extraer las posiciones de la pelota
ball_x = df['Ball X'].values
ball_y = df['Ball Y'].values

# Parámetro de umbral de distancia máxima permitida entre frames
distance_threshold = 900  # Ajusta según tu necesidad (en cm)

# Filtrar posiciones incorrectas
filtered_x = [ball_x[0]]
filtered_y = [ball_y[0]]

for i in range(1, len(ball_x)):
    # Verificar si hay valores válidos
    if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
        #print(f"Posición VALIDA detectada en el índice {i}: ({ball_x[i]}, {ball_y[i]})")
        continue

    # Calcular la distancia euclidiana entre el punto actual y el anterior
    distance = np.sqrt((ball_x[i] - filtered_x[-1])**2 + (ball_y[i] - filtered_y[-1])**2)

    # Evaluar si la distancia es válida
    if distance <= distance_threshold:
        filtered_x.append(ball_x[i])
        filtered_y.append(ball_y[i])
    #else:
    #    print(f"Posición incorrecta detectada en el índice {i}: ({ball_x[i]}, {ball_y[i]})")




# Crear un gradiente de color para los puntos filtrados
points = np.array([filtered_x, filtered_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Usar un mapa de colores (colormap)
cmap = plt.get_cmap("viridis")  # Cambia el colormap si deseas otro estilo
norm = plt.Normalize(0, len(filtered_x) - 1)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(np.arange(len(filtered_x)))

# Configurar el plano de la cancha
plt.figure(figsize=(10, 5))
plt.gca().add_collection(lc)
plt.scatter(filtered_x[0], filtered_y[0], color='red', label='Primer registro', zorder=5)
plt.scatter(filtered_x[-1], filtered_y[-1], color='blue', label='Último registro', zorder=5)

# Ajustar los límites para representar la cancha
plt.xlim(0, 4000)
plt.ylim(0, 2000)

# Etiquetas y título
plt.title('Ruta limpia del balón en la cancha con gradiente de tiempo')
plt.xlabel('Posición X (cm)')
plt.ylabel('Posición Y (cm)')
plt.colorbar(lc, label="Orden temporal")
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()
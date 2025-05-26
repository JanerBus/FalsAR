import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def false_position(f, a, b, tol=1e-6, max_iter=100):
    """
    Implementación del método de falsa posición
    Args:
        f: función a evaluar
        a, b: intervalo inicial [a, b]
        tol: tolerancia para el criterio de parada
        max_iter: máximo número de iteraciones
    Returns:
        (raíz aproximada, lista de iteraciones)
    """
    iterations = []
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        iterations.append((a, b, c, fa, fb, fc))
        
        if abs(fc) < tol:
            break
            
        if fa * fc < 0:
            b = c
        else:
            a = c
    return c, iterations

def create_ar_overlay(iterations, f, current_iter=None):
    """
    Crea una visualización mejorada del método de falsa posición
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Configuración del gráfico
    title = "Método de Falsa Posición\n"
    if current_iter is not None:
        title += f"Iteración {current_iter+1}/{len(iterations)}"
    else:
        title += "Solución Final"
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True)
    
    # Graficar la función
    x_min = min(min(i[0], i[1]) for i in iterations) - 0.5
    x_max = max(max(i[0], i[1]) for i in iterations) + 0.5
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)
    ax.plot(x_vals, y_vals, 'b-', label='f(x)', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # Graficar iteraciones
    if current_iter is None:
        current_iter = len(iterations) - 1
    
    a, b, c, fa, fb, fc = iterations[current_iter]
    
    # Línea secante
    x_sec = np.linspace(a, b, 2)
    y_sec = fa + (fb - fa)/(b - a) * (x_sec - a)
    ax.plot(x_sec, y_sec, 'r--', alpha=0.7, label='Secante')
    
    # Puntos a, b, c
    ax.plot(a, fa, 'go', markersize=8, label='a')
    ax.plot(b, fb, 'yo', markersize=8, label='b')
    ax.plot(c, fc, 'ro', markersize=8, label='c')
    
    # Anotaciones
    ax.text(a, fa, f'  a={a:.3f}', verticalalignment='bottom', fontsize=10)
    ax.text(b, fb, f'  b={b:.3f}', verticalalignment='bottom', fontsize=10)
    ax.text(c, fc, f'  c={c:.3f}', verticalalignment='bottom', fontsize=10)
    
    # Cuadro informativo
    info_text = (f"Iteración: {current_iter+1}/{len(iterations)}\n"
                f"a = {a:.5f}\nb = {b:.5f}\nc = {c:.5f}\n"
                f"f(c) = {fc:.2e}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right')
    
    # Conversión a imagen (compatible con todas las versiones de matplotlib)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img

def main():
    # Configuración de cámara (para teléfono por USB o IP)
    # USB (DroidCam): cap = cv2.VideoCapture(1)
    # IP Webcam: cap = cv2.VideoCapture("http://192.168.1.100:8080/video")
    cap = cv2.VideoCapture(1)  # Cambia según tu configuración
    
    # Configuración ArUco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    
    # Función a resolver (ejemplo: e^x - 3x)
    f = lambda x: np.exp(x) - 3*x
    
    # Ejecutar el método
    root, iterations = false_position(f, 0, 2, tol=1e-4)
    print(f"Raíz encontrada: {root:.6f}")
    print(f"Total de iteraciones: {len(iterations)}")
    
    current_iter = 0
    paused = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame")
            break
            
        # Detección de marcadores
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            # Dibujar marcador detectado
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Crear overlay con la iteración actual
            overlay = create_ar_overlay(iterations, f, current_iter if paused else None)
            
            # Redimensionar y superponer
            h, w = frame.shape[:2]
            overlay = cv2.resize(overlay, (w, h))
            
            # Combinar frame y overlay
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Mostrar estado
            status_text = f"Iter: {current_iter+1}/{len(iterations)} | {'PAUSADO' if paused else 'AUTO'}"
            cv2.putText(frame, status_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('FalsAR', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break
        elif paused:
            if key == ord('a'):  # Flecha izquierda
                current_iter = max(0, current_iter - 1)  # No menor que 0
            elif key == ord('b'):  # Flecha derecha
                current_iter = min(len(iterations) - 1, current_iter + 1)  # No mayor que el máximo
        elif not paused and current_iter < len(iterations) - 1:
            current_iter += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
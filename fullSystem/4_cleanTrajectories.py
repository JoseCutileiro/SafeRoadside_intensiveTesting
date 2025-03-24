import re
import math

# Função para calcular a distância Euclidiana entre dois pontos
def calcular_distancia(ponto1, ponto2):
    return math.sqrt((ponto2[0] - ponto1[0]) ** 2 + (ponto2[1] - ponto1[1]) ** 2)

# Função de suavização de média móvel
def suavizar_trajetoria(pontos, janela=3):
    pontos_suavizados = []
    
    for i in range(len(pontos)):
        # Definir o intervalo para a média móvel
        start = max(i - janela // 2, 0)
        end = min(i + janela // 2 + 1, len(pontos))
        
        # Obter os pontos da janela
        janela_pontos = pontos[start:end]
        
        # Calcular a média dos pontos na janela
        x_avg = sum(p[0] for p in janela_pontos) / len(janela_pontos)
        y_avg = sum(p[1] for p in janela_pontos) / len(janela_pontos)
        
        # Adicionar o ponto suavizado à lista
        pontos_suavizados.append((int(x_avg), int(y_avg)))
    
    return pontos_suavizados

# Abrir o arquivo de entrada contendo as trajetórias
with open('trajetorias.txt', 'r') as infile:
    # Abrir o arquivo de saída para escrever as trajetórias processadas
    with open('trajetoriasClean.txt', 'w') as outfile:
        for line in infile:
            # Remover espaços extras e nova linha
            line = line.strip()

            # Ignorar linhas vazias
            if not line:
                continue

            # Remover a parte 'Traj:' e dividir pelos parênteses fechados ')'
            points_raw = line.replace("Traj:", "").strip()  # Remover 'Traj:' no início

            # Usar regex para capturar corretamente os pontos
            points = re.findall(r'\((-?\d+),\s*(-?\d+)\)', points_raw)
            # Converter as strings para tuplas de inteiros
            points = [(int(x), int(y)) for x, y in points]

            # Selecionar pontos com distância maior que 20 pixels em relação ao último ponto selecionado
            selected_points = []
            last_point = None

            for point in points:
                if last_point is None:
                    selected_points.append(point)  # Adiciona o primeiro ponto
                    last_point = point  # Atualiza o último ponto selecionado
                else:
                    # Calcula a distância entre o ponto atual e o último ponto selecionado
                    distancia = calcular_distancia(last_point, point)
                    if distancia > 20:  # Se a distância for maior que 20 pixels
                        selected_points.append(point)
                        last_point = point  # Atualiza o último ponto selecionado
                
                

            # Suavizar a trajetória selecionada
            suavizados = suavizar_trajetoria(selected_points, janela=5)  # Ajuste da janela de suavização

            # Converter os pontos suavizados de volta para o formato desejado
            output = 'Traj: ' + ', '.join(f'({x}, {y})' for x, y in suavizados)

            # Escrever a linha processada no arquivo de saída
            outfile.write(output + '\n')

print("Arquivo 'trajetoriasClean.txt' foi criado com sucesso.")
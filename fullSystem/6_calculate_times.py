import os
import numpy as np
import sys

def process_file(filepath):
    try:
        with open(filepath, "r") as f:
            # Ler todas as linhas e converter para float
            times = [float(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Erro ao ler o ficheiro {filepath}: {e}")
        return None

    if not times:
        return None

    # Calcula a média e o desvio padrão (populacional)
    media = np.mean(times)
    desvio = np.std(times)

    # Determina quantos elementos serão considerados para os 100 piores
    n = min(100, len(times))
    # Considera os piores tempos (maiores valores)
    piores = sorted(times, reverse=True)[:n]
    media_piores = np.mean(piores)

    return media, desvio, media_piores

def main(folder="results"):
    if not os.path.isdir(folder):
        print(f"A pasta '{folder}' não foi encontrada.")
        return

    # Percorre todos os ficheiros na pasta
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            result = process_file(filepath)
            if result:
                media, desvio, media_piores = result
                print(f"{filename} {media:.3f} {desvio:.3f} {media_piores:.3f}")
            else:
                print(f"{filename} sem dados ou erro na leitura.")

if __name__ == "__main__":
    # Permite especificar a pasta via linha de comando, se desejado.
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

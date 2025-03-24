import pygame
import random

# Inicializa o Pygame
pygame.init()

# Configurações da janela
LARGURA = 1920
ALTURA = 1080
TAMANHO = (LARGURA, ALTURA)
FPS = 60

# Cores
BRANCO = (255, 255, 255)

# Configura a tela e o título
tela = pygame.display.set_mode(TAMANHO)
pygame.display.set_caption("Simulador de Movimentos")

# Carregar a imagem de fundo (certifique-se de que o caminho está correto)
background = pygame.image.load('withguide.png')
background = pygame.transform.scale(background, TAMANHO)

# Função para gerar cor aleatória
def cor_aleatoria():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))  # Evita cores muito escuras

# Função para salvar as coordenadas no arquivo
def salvar_coordenadas(trajetorias):
    with open("trajetorias.txt", "a") as f:
        for traj in trajetorias:
            f.write("Traj: ")
            for coord in traj['pontos']:
                f.write(f"({coord[0]}, {coord[1]}),")
            f.write("\n")

# Função para interpolar pontos entre dois pontos (equação da reta)
def interpolar_pontos(p1, p2, num_pontos=30):
    """Retorna uma lista de pontos entre p1 e p2, dividindo a linha em num_pontos partes"""
    x1, y1 = p1
    x2, y2 = p2
    pontos = []
    for i in range(num_pontos + 1):  # Garante que o último ponto também seja incluído
        t = i / num_pontos
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        pontos.append((x, y))
    return pontos

# Função principal
def main():
    clock = pygame.time.Clock()

    # Lista para armazenar as trajetórias (cada uma com cor e pontos)
    trajetorias = []
    movimento_ativo = False
    coordenadas_atuais = []
    cor_atual = cor_aleatoria()  # Cor inicial aleatória
    ponto_reta = None  # Armazena o primeiro ponto para desenhar linha reta

    while True:
        tela.fill(BRANCO)  # Preenche o fundo com branco
        tela.blit(background, (0, 0))  # Coloca a imagem de fundo

        # Eventos
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                if coordenadas_atuais:  # Se houver coordenadas a salvar
                    trajetorias.append({"pontos": coordenadas_atuais, "cor": cor_atual})
                salvar_coordenadas(trajetorias)  # Salva as trajetórias no arquivo
                pygame.quit()
                return

            if evento.type == pygame.MOUSEBUTTONDOWN:
                x, y = evento.pos  # Posição do clique

                if evento.button == 1:  # Clique esquerdo inicia desenho livre
                    movimento_ativo = True
                    cor_atual = cor_aleatoria()  # Nova cor para cada nova trajetória
                    coordenadas_atuais = [(x, y)]  # Começa uma nova trajetória

                elif evento.button == 3:  # Clique direito -> Criar linha reta
                    if ponto_reta is None:
                        # Primeiro clique: marca o ponto inicial
                        ponto_reta = (x, y)
                    else:
                        # Segundo clique: desenha a reta entre o ponto anterior e o atual
                        pontos_interpolados = interpolar_pontos(ponto_reta, (x, y), num_pontos=30)
                        trajetorias.append({"pontos": pontos_interpolados, "cor": cor_atual})
                        ponto_reta = None  # Mantém o último ponto para continuidade

            if evento.type == pygame.MOUSEBUTTONUP:
                if evento.button == 1:  # Se o botão esquerdo do mouse for solto
                    movimento_ativo = False
                    if coordenadas_atuais:
                        trajetorias.append({"pontos": coordenadas_atuais, "cor": cor_atual})
                    coordenadas_atuais = []  # Reinicia as coordenadas

            if evento.type == pygame.MOUSEMOTION:
                if movimento_ativo:
                    x, y = evento.pos
                    coordenadas_atuais.append((x, y))  # Adiciona a coordenada na lista

        # Desenhar todas as trajetórias armazenadas
        for traj in trajetorias:
            cor = traj["cor"]
            pontos = traj["pontos"]
            for i in range(1, len(pontos)):
                pygame.draw.line(tela, cor, pontos[i - 1], pontos[i], 3)  # Linhas conectando os pontos

        # Desenha a trajetória atual enquanto o usuário desenha
        if movimento_ativo and len(coordenadas_atuais) > 1:
            for i in range(1, len(coordenadas_atuais)):
                pygame.draw.line(tela, cor_atual, coordenadas_atuais[i - 1], coordenadas_atuais[i], 3)

        pygame.display.flip()  # Atualiza a tela
        clock.tick(FPS)  # Controla o FPS

if __name__ == "__main__":
    main()

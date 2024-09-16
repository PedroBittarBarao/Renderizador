#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time  # Para operações com tempo
import math  # Funções matemáticas
import gpu  # Simula os recursos de uma GPU
import numpy as np  # Biblioteca do Numpy
import custom_funcs as cf


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.5  # plano de corte próximo
    far = 100  # plano de corte distante
    p = None

    @staticmethod
    def setup(width, height, near=0.5, far=100):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.stack = [np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])]

    @staticmethod
    def pushMatrix(matrix):
        """Função usada para empilhar uma matriz."""
        GL.stack.append(GL.getMatrix() @ matrix)

    @staticmethod
    def popMatrix():
        """Função usada para desempilhar uma matriz."""
        GL.stack.pop()

    @staticmethod
    def getMatrix():
        """Função usada para retornar a matriz do topo da pilha."""
        return GL.stack[-1]

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Polypoint2D : pontos = {0}".format(point)) # imprime no terminal pontos
        # print("Polypoint2D : colors = {0}".format(colors)) # imprime no terminal as cores

        emissive = colors["emissiveColor"]
        emissive = [int(i * 255) for i in emissive]
        n_points = int(len(point) / 2)
        for i in range(n_points):
            if (
                point[i * 2] >= 0
                and point[i * 2 + 1] >= 0
                and point[i * 2] < GL.width
                and point[i * 2 + 1] < GL.height
            ):
                gpu.GPU.draw_pixel(
                    [math.floor(point[i * 2]), math.floor(point[i * 2 + 1])],
                    gpu.GPU.RGB8,
                    emissive,
                )

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        # print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        # print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores

        emissive = colors["emissiveColor"]
        emissive = [int(i * 255) for i in emissive]

        n_lines = int(len(lineSegments) / 2 - 1)

        for i in range(n_lines):
            x0 = lineSegments[2 * i]
            y0 = lineSegments[2 * i + 1]
            x1 = lineSegments[2 * i + 2]
            y1 = lineSegments[2 * i + 3]
            delta_x = x1 - x0
            delta_y = y1 - y0

            if delta_x == 0:
                if delta_y > 0:
                    y = y0
                    while y <= y1:
                        if x0 >= 0 and y >= 0 and x0 < GL.width and y < GL.height:
                            gpu.GPU.draw_pixel(
                                [int(x0), math.floor(y)], gpu.GPU.RGB8, emissive
                            )
                        y += 1
                else:
                    y = y1
                    while y <= y0:
                        if x0 >= 0 and y >= 0 and x0 < GL.width and y < GL.height:
                            gpu.GPU.draw_pixel(
                                [int(x0), math.floor(y)], gpu.GPU.RGB8, emissive
                            )
                        y += 1

            else:
                m = delta_y / delta_x
                if abs(m) <= 1:
                    if delta_x > 0:
                        x = x0
                        y = y0
                        while x <= x1:
                            if x >= 0 and y >= 0 and x < GL.width and y < GL.height:
                                gpu.GPU.draw_pixel(
                                    [math.floor(x), math.floor(y)],
                                    gpu.GPU.RGB8,
                                    emissive,
                                )
                            y += m
                            x += 1
                    else:
                        x = x1
                        y = y1
                        while x <= x0:
                            if x >= 0 and y >= 0 and x < GL.width and y < GL.height:
                                gpu.GPU.draw_pixel(
                                    [math.floor(x), math.floor(y)],
                                    gpu.GPU.RGB8,
                                    emissive,
                                )
                            y += m
                            x += 1
                else:
                    if delta_y > 0:
                        y = y0
                        x = x0
                        while y <= y1:
                            if x >= 0 and y >= 0 and x < GL.width and y < GL.height:
                                gpu.GPU.draw_pixel(
                                    [math.floor(x), math.floor(y)],
                                    gpu.GPU.RGB8,
                                    emissive,
                                )
                            y += 1
                            x += 1 / m
                    else:
                        y = y1
                        x = x1
                        while y <= y0:
                            if x >= 0 and y >= 0 and x < GL.width and y < GL.height:
                                gpu.GPU.draw_pixel(
                                    [math.floor(x), math.floor(y)],
                                    gpu.GPU.RGB8,
                                    emissive,
                                )
                            y += 1
                            x += 1 / m

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius))  # imprime no terminal
        print("Circle2D : colors = {0}".format(colors))  # imprime no terminal as cores

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        # print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        # print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores

        emissive = colors["emissiveColor"]
        emissive = [int(i * 255) for i in emissive]

        n_trigs = int(len(vertices) / 6)
        for i in range(n_trigs):
            x0 = vertices[6 * i]
            y0 = vertices[6 * i + 1]
            x1 = vertices[6 * i + 2]
            y1 = vertices[6 * i + 3]
            x2 = vertices[6 * i + 4]
            y2 = vertices[6 * i + 5]

            area = cf.area([x0, y0], [x1, y1], [x2, y2])  # signed area
            if area > 0:  # clockwise
                x0, y0, x1, y1, x2, y2 = x1, y1, x0, y0, x2, y2  # swap points
                area = -area  # invert area

            # print("COORDS",x0,y0,x1,y1,x2,y2)

            min_x = math.floor(min(x0, x1, x2))
            max_x = math.ceil(max(x0, x1, x2))
            min_y = math.floor(min(y0, y1, y2))
            max_y = math.ceil(max(y0, y1, y2))

            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    inside = cf.dentro([x0, y0], [x1, y1], [x2, y2], [x + 0.5, y + 0.5])
                    if inside and x >= 0 and y >= 0 and x < GL.width and y < GL.height:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, emissive)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        emissive = colors["emissiveColor"]
        emissive = [int(i * 255) for i in emissive]
        n_trigs = int(len(point) / 9)
        for t in range(n_trigs):
            x1, y1, z1 = point[9 * t : 9 * t + 3]
            x2, y2, z2 = point[9 * t + 3 : 9 * t + 6]
            x3, y3, z3 = point[9 * t + 6 : 9 * t + 9]
            trig_p = np.array([[x1, x2, x3],
                               [y1, y2, y3],
                               [z1, z2, z3],
                               [1 , 1 , 1 ]])
            trig_p = GL.getMatrix() @ trig_p

            # print(GL.pm)
            aux_m = GL.pm @ trig_p  # perspective X rotation X translation X points
            NDC_m = aux_m / aux_m[3][0]  # NDC
            screen_m = cf.NDC_to_screen_matrix(GL.width, GL.height) @ NDC_m
            screen_m = np.array(screen_m)
            GL.triangleSet2D(
                [
                    screen_m[0][0],screen_m[1][0],
                    screen_m[0][1],screen_m[1][1],
                    screen_m[0][2],screen_m[1][2],
                ],
                colors,
            )

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.
        fovy = 2 * np.arctan(
            np.tan(fieldOfView / 2) * GL.height / (np.sqrt(GL.height**2 + GL.width**2))
        )

        aspect = GL.width / GL.height
        top = GL.near * np.tan(fovy)
        bottom = -top
        right = top * aspect
        left = -right

        cam_pos = np.matrix(
            [
                [1.0, 0.0, 0.0, position[0]],
                [0.0, 1.0, 0.0, position[1]],
                [0.0, 0.0, 1.0, position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        rotation_m = cf.rotate_quat(orientation)

        look_at_trans = np.linalg.inv(cam_pos)
        look_at_rot = np.linalg.inv(rotation_m)
        look_at_mat = look_at_rot @ look_at_trans

        perspective_m = np.array(
            [[GL.near / right, 0, 0, 0],
            [0, GL.near / top, 0, 0],
            [0,0,-(GL.far + GL.near) / (GL.far - GL.near),-(2 * GL.far * GL.near) / (GL.far - GL.near),],
            [0, 0, -1, 0]]
        )

        GL.camPos = np.array([position[0], position[1], position[2], 1])
        GL.camRot = orientation

        GL.pm = perspective_m @ look_at_mat

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        """ print("Viewpoint : ", end='')
        print("position = {0} ".format(position), end='')
        print("orientation = {0} ".format(orientation), end='')
        print(f"aspect{aspect} ",end="")
        print("fieldOfView = {0} ".format(fieldOfView)) """

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas.
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        GL.t_translation = [0, 0, 0]
        GL.t_scale = [1, 1, 1]
        GL.t_rotation = [0, 0, 0, 0]

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Transform : ", end='')
        if translation:
            GL.t_translation = translation
            # print("translation = {0} ".format(GL.t_translation), end='') # imprime no terminal
        if scale:
            GL.t_scale = scale
            # print("scale = {0} ".format(GL.t_scale), end='') # imprime no terminal
        if rotation:
            GL.t_rotation = rotation
            # print("rotation = {0} ".format(GL.t_rotation), end='') # imprime no terminal
        # print("")

        translate_m = cf.translation_matrix(
            GL.t_translation[0], GL.t_translation[1], GL.t_translation[2]
        )
        rotate_m = cf.rotate_quat(rotation)
        scale_m = np.array(
            [
                [GL.t_scale[0], 0, 0, 0],
                [0, GL.t_scale[1], 0, 0],
                [0, 0, GL.t_scale[2], 0],
                [0, 0, 0, 1],
            ]
        )
        t_transform = translate_m @ rotate_m @ scale_m
        np.set_printoptions(suppress=True)
        GL.pushMatrix(t_transform)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        GL.popMatrix()
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Saindo de Transform")

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        vertices = []
        for i in range(0, len(point) - 6, 3):  #
            for u in range(0, 9):  # appending each vertex, 3 vertices
                vertices.append(point[i + u])

        GL.triangleSet(vertices, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""

        vertices = []
        num_indices = len(index)

        for i in range(num_indices - 2):
            if index[i] == -1 or index[i + 1] == -1 or index[i + 2] == -1:
                continue

            # Pre-calculate the coordinates for each index only once
            coord1 = index[i] * 3
            coord2 = index[i + 1] * 3
            coord3 = index[i + 2] * 3

            # Extend vertices list directly with all coordinates
            vertices.extend(point[coord1 : coord1 + 3])  # Vertex 1
            vertices.extend(point[coord2 : coord2 + 3])  # Vertex 2
            vertices.extend(point[coord3 : coord3 + 3])  # Vertex 3

        # Now pass the collected vertices to the rendering function
        GL.triangleSet(vertices, colors)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size))  # imprime no terminal pontos
        print("Box : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""

        # Handle texture cases first
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            # Apply texture logic here (omitted for now)

        # Handle case when colors are provided per vertex
        elif colorPerVertex and color and colorIndex:
            # Apply color-per-vertex logic here (omitted for now)
            pass

        # General case without texture and colors per vertex
        else:
            # First step: parse coordIndex into faces (groups of vertices)
            faces = []
            vertices = []
            
            # Collect the faces based on -1 delimiters
            for i in coordIndex:
                if i == -1:
                    if vertices:  # If there are vertices collected, add as a face
                        faces.append(vertices)
                    vertices = []  # Reset for the next face
                else:
                    vertices.append(i)

            # Initialize strips for triangle strip set
            strips = []

            # Convert each face into a triangle strip
            for face in faces:
                if len(face) < 3:
                    continue  # Skip faces with fewer than 3 vertices
                
                # Add triangles in a fan-like pattern from the first vertex
                strips.extend([face[0], face[i], face[i + 1], -1] for i in range(1, len(face) - 1))

            # Flatten the list of strips
            strips_flat = [item for sublist in strips for item in sublist]

            # Call the triangle strip rendering function
            GL.indexedTriangleStripSet(coord, strips_flat, colors)


    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "Sphere : radius = {0}".format(radius)
        )  # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors))  # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "NavigationInfo : headlight = {0}".format(headlight)
        )  # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color))  # imprime no terminal
        print(
            "DirectionalLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "DirectionalLight : direction = {0}".format(direction)
        )  # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color))  # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity))  # imprime no terminal
        print("PointLight : location = {0}".format(location))  # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""

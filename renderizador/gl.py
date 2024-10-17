#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Pedro Bittar Barão
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time  # Para operações com tempo
import math  # Funções matemáticas
import gpu  # Simula os recursos de uma GPU
import numpy as np  # Biblioteca do Numpy
import custom_funcs as cf
from random import randint


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.5  # plano de corte próximo
    far = 100  # plano de corte distante
    p = None
    z_buffer = None
    ambient_intensity = 0.0
    start_time = 0.0

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
        GL.super_buffer_0 = np.zeros((GL.width*2, GL.height*2, 3))
        GL.super_buffer_1 = np.zeros((GL.width*2, GL.height*2, 3))
        GL.current_super_buffer = GL.super_buffer_0
        GL.current_super_buffer_index = 0
        GL.z_buffer = - np.inf * np.ones((GL.width*2, GL.height*2))
        GL.directional_light = {"direction": np.array([0, 0, -1]), "color": np.array([1, 1, 1]), "intensity": 0}
        GL.point_light = {"position": np.array([0, 0, 0]), "color": np.array([1, 1, 1]), "intensity": 0}
        GL.start_time = time.time()

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
    def triangleSet2D(vertices, colors,
                    colorPerVertex = False,vertexColors = None ,zs = None,
                    texture_values = None,image = None,
                    transparency = 0,
                    vertex_normals = None,
                    face_normal = None,
                    h = None):
        """Função usada para renderizar TriangleSet2D."""

        if image is not None:
            mipmaps = cf.generate_mipmap(image)

        # Get the emissive color, convert to 8-bit RGB
        emissive_color = colors["emissiveColor"]
        diffuse_color = colors["diffuseColor"]
        specular_color = colors["specularColor"]
        shininess = colors["shininess"]
        emissive_color = [int(i * 255) for i in emissive_color]
        diffuse_color = [int(i * 255) for i in diffuse_color]
        specular_color = [int(i * 255) for i in specular_color]

        has_lights = GL.directional_light["intensity"] > 0 or GL.point_light["intensity"] > 0

        # Number of triangles to process
        n_trigs = int(len(vertices) / 6)

        # Iterate over each triangle
        for i in range(n_trigs):
            random_color = [randint(0,255),randint(0,255),randint(0,255)]
            #initialize diffuse and specular colors as 0
            diffuse_light  = np.array([0, 0, 0])
            specular_light = np.array([0, 0, 0])

            # Extract triangle vertices
            x0, y0 = vertices[6 * i], vertices[6 * i + 1]
            x1, y1 = vertices[6 * i + 2], vertices[6 * i + 3]
            x2, y2 = vertices[6 * i + 4], vertices[6 * i + 5]

            """ print(f"zs = {zs}")
            print(f"fn = {face_normal}") """

            # Calculate the signed area of the triangle
            area = cf.area([x0, y0], [x1, y1], [x2, y2])

            # If the triangle is in clockwise order, swap vertices to make it counterclockwise
            if area > 0:
                #print("Clockwise triangle detected, swapping vertices")
                x0, y0, x1, y1 = x1, y1, x0, y0  # Swap vertices to ensure counterclockwise order
            
            super_x0 = x0*2
            super_y0 = y0*2
            super_x1 = x1*2
            super_y1 = y1*2
            super_x2 = x2*2
            super_y2 = y2*2

            # Pre-calculate bounding box of the triangle
            super_min_x = max(0, math.floor(min(super_x0, super_x1, super_x2)))
            super_max_x = min(GL.width*2 - 1, math.ceil(max(super_x0, super_x1, super_x2)))
            super_min_y = max(0, math.floor(min(super_y0, super_y1, super_y2)))
            super_max_y = min(GL.height*2 - 1, math.ceil(max(super_y0, super_y1, super_y2)))

            min_x = max(0, math.floor(min(x0, x1, x2)))
            max_x = min(GL.width - 1, math.ceil(max(x0, x1, x2)))
            min_y = max(0, math.floor(min(y0, y1, y2)))
            max_y = min(GL.height - 1, math.ceil(max(y0, y1, y2)))

            # Loop over the bounding box only
            for sx in range(super_min_x, super_max_x + 1):
                for sy in range(super_min_y, super_max_y + 1):
                    # Check if the pixel center (x+0.5, y+0.5) is inside the triangle
                    if cf.dentro([super_x0, super_y0], [super_x1, super_y1], [super_x2, super_y2], [sx + 0.5, sy + 0.5]):
                        alpha, beta, gamma = cf.calculate_barycentric_coordinates(super_x0, super_y0,
                                                                                super_x1, super_y1,
                                                                                super_x2, super_y2,
                                                                                sx+0.5, sy+0.5)
                        if zs is not None:
                            z = 1/(alpha/zs[0] + beta/zs[1] + gamma/zs[2])

                            if z > GL.z_buffer[sx, sy]:
                                GL.z_buffer[sx, sy] = z
                            else:
                                continue # Discard pixel if it is behind another triangle

                        if colorPerVertex: # Draw color
                            r = vertexColors[3*i][0] * alpha + vertexColors[3*i+1][0] * beta + vertexColors[3*i+2][0] * gamma
                            g = vertexColors[3*i][1] * alpha + vertexColors[3*i+1][1] * beta + vertexColors[3*i+2][1] * gamma
                            b = vertexColors[3*i][2] * alpha + vertexColors[3*i+1][2] * beta + vertexColors[3*i+2][2] * gamma
                            cr = z * r/zs[0]
                            cg = z * g/zs[1]
                            cb = z * b/zs[2]
                            draw_color = [int(cr), int(cg), int(cb)]

                        elif texture_values is not None: # Draw texture
                            image_shape = image.shape[0]
                            u0, v0 = texture_values[6*i], texture_values[6*i + 1]
                            u1, v1 = texture_values[6*i + 2], texture_values[6*i + 3]
                            u2, v2 = texture_values[6*i + 4], texture_values[6*i + 5]
                            u = u0 * alpha + u1 * beta + u2 * gamma
                            v = v0 * alpha + v1 * beta + v2 * gamma
                            
                            # u e v vizinhos
                            alpha_10, beta_10, gamma_10 = cf.calculate_barycentric_coordinates(super_x0, super_y0,
                                                                                            super_x1, super_y1,
                                                                                            super_x2, super_y2,
                                                                                            sx + 1 + 0.5, sy + 0.5)
                            u_10 = u0 * alpha_10 + u1 * beta_10 + u2 * gamma_10
                            v_10 = v0 * alpha_10 + v1 * beta_10 + v2 * gamma_10

                            alpha_01, beta_01, gamma_01 = cf.calculate_barycentric_coordinates(super_x0, super_y0,   
                                                                                            super_x1, super_y1,
                                                                                            super_x2, super_y2,
                                                                                            sx + 0.5, sy + 1 + 0.5)
                            u_01 = u0 * alpha_01 + u1 * beta_01 + u2 * gamma_01
                            v_01 = v0 * alpha_01 + v1 * beta_01 + v2 * gamma_01

                            del_u_del_x = image_shape*(u_10 - u)
                            del_v_del_x = image_shape*(v_10 - v)
                            del_u_del_y = image_shape*(u_01 - u)
                            del_v_del_y = image_shape*(v_01 - v)
                            d = cf.mipmap_level(del_u_del_x,del_u_del_y,del_v_del_x,del_v_del_y)

                            current_mipmap = mipmaps[d]
                            current_shape = current_mipmap.shape[0]
                            
                            u_z0 = u0 / zs[0]
                            v_z0 = v0 / zs[0]
                            u_z1 = u1 / zs[1]
                            v_z1 = v1 / zs[1]
                            u_z2 = u2 / zs[2]
                            v_z2 = v2 / zs[2]

                            u_interpolated = (alpha * u_z0 + beta * u_z1 + gamma * u_z2) / (alpha /zs[0] + beta /zs[1] + gamma /zs[2])
                            v_interpolated = (alpha * v_z0 + beta * v_z1 + gamma * v_z2) / (alpha /zs[0] + beta /zs[1] + gamma /zs[2])

                            flipped_image = np.flip(current_mipmap[:, :, :3],axis=1)
                            r, g, b = flipped_image[min(255, int(u_interpolated * current_shape)), min(255, int(v_interpolated * current_shape))]

                            draw_color = [min(255,int(r)), min(255,int(g)), min(255,int(b))]
                    
                        else:
                            if not has_lights: # Draw only emissive color if there are no lights
                                draw_color = emissive_color
                            else:
                                if face_normal is not None:
                                    # pixel normal = face normal
                                    normal = face_normal
                                else:
                                    # pixel normal = interpolated vertex normal
                                    normal = np.array(alpha * vertex_normals[:, 0] + beta * vertex_normals[:, 1] + gamma * vertex_normals[:, 2]).T[0]
                                    normal = normal / np.linalg.norm(normal)
                                
                                #interpolate h vector
                                h_interpolated = alpha * h[0] + beta * h[1] + gamma * h[2]

                                # cos of the angle between the normal and the light direction
                                cos_n_l = max(0,- (normal[0]*GL.directional_light["direction"][0] + 
                                                   normal[1]*GL.directional_light["direction"][1] + 
                                                   normal[2]*GL.directional_light["direction"][2])) 
                                
                                cos_n_h = max(0,- (normal[0]*h_interpolated[0] + normal[1]*h_interpolated[1] + normal
                                [2]*h_interpolated[2]))
                                
                                diffuse_intensity = GL.directional_light["intensity"] * (cos_n_l + GL.ambient_intensity)
                                diffuse_light = diffuse_intensity * np.array(diffuse_color)
                                diffuse_light = np.clip(diffuse_light, 0, 255)
                                # complete with specular, ambient and emissive light
                                specular_intensity = 255 * GL.directional_light["intensity"] * (cos_n_h ** (shininess*128)) # FIXME
                                specular_light = specular_intensity * np.array(specular_color)
                                specular_light = np.clip(specular_light, 0, 255)

                                draw_color = emissive_color + diffuse_light + specular_light
                                draw_color = np.clip(draw_color, 0, 255)

                            draw_color = [int(i) for i in draw_color]

                        # Transparency
                        previous_color = GL.current_super_buffer[sx, sy]
                        draw_color = [int((draw_color[0] * (1 - transparency) + previous_color[0] * transparency)),
                                        int((draw_color[1] * (1 - transparency) + previous_color[1] * transparency)),
                                        int((draw_color[2] * (1 - transparency) + previous_color[2] * transparency))]
                        
                        # Debug: draw normal components xyz as rgb color in the triangle
                        #draw_color = cf.vector_to_color(normal)
                        #draw_color = [int(cos_n_h*255), int(cos_n_h*255), int(cos_n_h*255)]
                        #draw_color = random_color
        
                        GL.current_super_buffer[sx, sy] = draw_color

            
            # Draw the triangle on the screen (supersampled)
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                        p0 = GL.current_super_buffer[x*2, y*2]
                        p1 = GL.current_super_buffer[x*2 + 1, y*2]
                        p2 = GL.current_super_buffer[x*2, y*2 + 1]
                        p3 = GL.current_super_buffer[x*2 + 1, y*2 + 1]
                        #print(p0,p1,p2,p3)
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, (p0 + p1 + p2 + p3) / 4)    


    @staticmethod
    def triangleSet(point, colors,colorPerVertex = False,vertexColors = None,
                    texture_values = None,image = None,
                    vertex_normals = None):
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

        has_lights = GL.directional_light["intensity"] > 0 or GL.point_light["intensity"] > 0

        transparency = colors["transparency"]
        face_normal = None
        n_trigs = int(len(point) / 9)
        tranform_m = GL.getMatrix()
        cam_to_screen = cf.NDC_to_screen_matrix(GL.width, GL.height)
        for t in range(n_trigs):
            x1, y1, z1 = point[9 * t : 9 * t + 3]
            x2, y2, z2 = point[9 * t + 3 : 9 * t + 6]
            x3, y3, z3 = point[9 * t + 6 : 9 * t + 9]
            if (x1 == x2 and y1 == y2 and z1 == z2):
                raise ValueError("Vertices 1 and 2 are the same",f"{x1},{y1},{z1}] in triangle {t} of {n_trigs}")
            if (x1 == x3 and y1 == y3 and z1 == z3):
                raise ValueError("Vertices 1 and 3 are the same",f"{x1},{y1},{z1} in triangle {t} of {n_trigs}")
            if (x2 == x3 and y2 == y3 and z2 == z3):
                raise ValueError("Vertices 2 and 3 are the same",f"{x2},{y2},{z2} in triangle {t} of {n_trigs}")
            triangle_p = np.array([[x1, x2, x3],
                               [y1, y2, y3],
                               [z1, z2, z3],
                               [1 , 1 , 1 ]])
            transformed_p = tranform_m @ triangle_p
            looked_at = GL.look_at @ transformed_p

            if vertex_normals is None:
                normal = np.cross(
                    looked_at[:3].transpose()[1] - looked_at[:3].transpose()[0],
                    looked_at[:3].transpose()[2] - looked_at[:3].transpose()[0])
                
                face_normal = normal[0] / np.linalg.norm(normal)
            else:
                # Extract normal vectors and convert them to column vectors (3x1)
                normal_0 = np.matrix(vertex_normals[9*t:9*t+3]).T  # (3,1)
                normal_1 = np.matrix(vertex_normals[9*t+3:9*t+6]).T  # (3,1)
                normal_2 = np.matrix(vertex_normals[9*t+6:9*t+9]).T  # (3,1)
                
                # Convert normal vectors to homogeneous coordinates by appending 0
                normal_0_h = np.vstack([normal_0, [0]])  # (4,1)
                normal_1_h = np.vstack([normal_1, [0]])  # (4,1)
                normal_2_h = np.vstack([normal_2, [0]])  # (4,1)

                # Combine the homogenous normal vectors into a 4x3 matrix
                normal_m_h = np.array([normal_0_h, normal_1_h, normal_2_h]).T  # (4,3)

                # Now you can multiply the 4x4 transformation matrix with the 4x3 normal matrix
                transformed__normals_m = np.linalg.inv(GL.look_at @ tranform_m).T @ normal_m_h

                # Extract the transformed normal vectors to a list in xyz order
                vertex_normals_transformed = transformed__normals_m[:3]

            aux_m = GL.pm @ transformed_p  # perspective X rotation X translation X points
            NDC_m = aux_m / aux_m[3][0]  # NDC
            zs = (looked_at)[2]
            screen_m = cam_to_screen @ NDC_m
            screen_m = np.array(screen_m)

            h = transformed_p[:3] + GL.directional_light["direction"] # h = l + v
            h = h / np.linalg.norm(h)

            GL.triangleSet2D(
                [
                    screen_m[0][0],screen_m[1][0],
                    screen_m[0][1],screen_m[1][1],
                    screen_m[0][2],screen_m[1][2],
                ],
                colors,
                colorPerVertex,
                vertexColors[3*t:3*t+3] if colorPerVertex else None,
                np.array(zs)[0],
                texture_values[6*t:6*t+6] if texture_values is not None else None,
                image = image,
                transparency = transparency,
                vertex_normals = vertex_normals_transformed if vertex_normals is not None else None,
                face_normal = face_normal,
                h=h if has_lights else None
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
        GL.look_at = look_at_mat

        perspective_m = np.array(
            [[GL.near / right, 0, 0, 0],
            [0, GL.near / top, 0, 0],
            [0,0,-(GL.far + GL.near) / (GL.far - GL.near),-(2 * GL.far * GL.near) / (GL.far - GL.near),],
            [0, 0, -1, 0]]
        )

        GL.camPos = np.array([position[0], position[1], position[2], 1])
        GL.camRot = orientation

        GL.pm = perspective_m @ look_at_mat


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
        if translation is not None:
            GL.t_translation = np.array(translation).flatten()
            # print("translation = {0} ".format(GL.t_translation), end='') # imprime no terminal
        if scale:
            GL.t_scale = scale
            # print("scale = {0} ".format(GL.t_scale), end='') # imprime no terminal
        if rotation is not None:
            GL.t_rotation = np.array(rotation).flatten()
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

        # Initialize the list to store all the vertices for the triangle strips
        vertices = []
        offset = 0  # To keep track of the starting point of each strip

        for count in stripCount:
            # Process each strip based on its count
            for i in range(count - 2):  # Each triangle in the strip reuses 2 points from the previous one
                # Append the vertices for each triangle in the strip
                vertices.extend(point[offset + i * 3: offset + i * 3 + 9])

            # Move the offset to the next strip
            offset += count * 3

        # Call the triangle rendering function with the prepared vertices
        GL.triangleSet(vertices, colors)


    @staticmethod
    def indexedTriangleStripSet(point, index, colors,
                                colorPerVertex=False, vertexColors=None,colorIndex=None,
                                texCoord=None, image=None):
        """Função usada para renderizar IndexedTriangleStripSet."""
        
        vertices = []
        color_values = []  # Will store colors for each vertex if colorPerVertex is True
        texture_values = []  # Will store texture coordinates for each vertex if texture is enabled
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
            # If colorPerVertex is enabled, collect colors for these vertices

            if colorPerVertex and colorIndex is not None:
                color1 = colorIndex[i] * 3
                color2 = colorIndex[i + 1] * 3
                color3 = colorIndex[i + 2] * 3

                color_values.extend(vertexColors[color1 : color1 + 3])  # Color 1
                color_values.extend(vertexColors[color2 : color2 + 3])  # Color 2
                color_values.extend(vertexColors[color3 : color3 + 3])  # Color 3
            
            elif texCoord is not None:
                t1 = index[i] * 2
                t2 = index[i + 1] * 2
                t3 = index[i + 2] * 2
                texture_values.extend(texCoord[t1 : t1 + 2])
                texture_values.extend(texCoord[t2 : t2 + 2])
                texture_values.extend(texCoord[t3 : t3 + 2])

        



        # Now pass the collected vertices and colors to the rendering function
        GL.triangleSet(vertices, colors,
                    colorPerVertex, vertexColors = color_values if colorPerVertex else None,
                    texture_values = texture_values if image is not None else None,image = image)

        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.





    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        point = cf.box(size)
        

        # Call the triangle rendering function with the prepared vertices and triangles
        GL.triangleSet(point, colors)

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

        vertex_colors = None  # Will store per-vertex colors if needed
        

        # Handle case when colors are provided per vertex
        if colorPerVertex and color and colorIndex:
            # If we have per-vertex colors, create a list of vertex colors
            vertex_colors = []
            for idx in colorIndex:
                if idx != -1:
                    # Since color is a flat list, extract RGB values as a triplet
                    start = idx * 3  # Each color is represented by 3 consecutive floats
                    rgb = color[start:start + 3]
                    
                    
                    # Convert each component from 0-1 float to 0-255 integer
                    vertex_colors.append([int(c * 255) for c in rgb])
                    

        # General case without texture and colors per vertex
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
        GL.indexedTriangleStripSet(coord, strips_flat,colors,
                                colorPerVertex and color and colorIndex, vertex_colors if vertex_colors else colors,colorIndex,
                                texCoord, image if current_texture else None) 
        
    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        points,vertex_normals = cf.sphere(radius,14,14)
        GL.triangleSet(points, colors,vertex_normals=vertex_normals)


    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        triangles = cf.cone(bottomRadius, height)
        GL.triangleSet(triangles, colors)


    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        triangles = cf.cylinder(radius, height)
        GL.triangleSet(triangles, colors)

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        if headlight:
            GL.directionalLight(0.0, [1, 1, 1], 1, [0, 0, -1])

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

        directional_light_direction = np.array(direction)

        GL.directional_light = {"ambient_intensity":ambientIntensity,
                                "color":color,
                                "intensity":intensity,
                                "direction":directional_light_direction}

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
        GL.point_light = {"ambient_intensity":ambientIntensity,
                            "color":color,
                            "intensity":intensity,
                            "location":location}

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

        cf.clear_framebuffer(GL.current_super_buffer)
        GL.z_buffer = np.full((GL.width * 2, GL.height * 2), -np.inf)

        # Esse método já está implementado para os alunos como exemplo
        if loop:
            epoch = (time.time())  # time in seconds since the epoch as a floating point number.
            relative_time = ((epoch - GL.start_time) % cycleInterval) / cycleInterval
        else:
            relative_time = np.clip(epoch - GL.start_time / cycleInterval, 0, 1)

        #print(f"TimeSensor : fraction_changed = {relative_time}")
        #print("TimeSensor : cycleInterval = {0}".format(cycleInterval))  # imprime no terminal
        #print("TimeSensor : loop = {0}".format(loop))

        return relative_time

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        key = np.array(key)
        keyValue = np.array(keyValue)

        k_i_before = 0
        for k_i_before in range(len(key) - 1): # Find the key interval
            if key[k_i_before] <= set_fraction <= key[k_i_before + 1]:
                k_i_before = k_i_before
                k_i_plus = k_i_before + 1
                break

        key_value_parsed = keyValue.reshape(-1, 3)
        print(f"SplinePositionInterpolator : key_value_parsed = \n{key_value_parsed}")

        if set_fraction == key[k_i_before]:
            return key_value_parsed[k_i_before:k_i_before+1]
        if set_fraction == key[k_i_plus]:
            return key_value_parsed[k_i_plus:k_i_plus+1]
        
        delta_key = key[k_i_plus] - key[k_i_before]
        s_m = np.array([
            ((set_fraction - key[k_i_before])/delta_key)**3,
            ((set_fraction - key[k_i_before])/delta_key)**2,
            (set_fraction - key[k_i_before])/delta_key,
            1
        ])

        # Handle boundary cases for delta_value_0
        if k_i_before == 0:
            if closed:
                deriv_0 = (key_value_parsed[-1] - key_value_parsed[1]) * 0.5
            else:
                deriv_0 = np.array([0, 0, 0])
        else:
            deriv_0 = (key_value_parsed[k_i_before-1] - key_value_parsed[k_i_before+1]) * 0.5


        # Handle boundary cases for delta_value_1
        if k_i_plus == len(key) - 1:
            if closed:
                deriv_1 = (key_value_parsed[0] - key_value_parsed[-2]) * 0.5
            else:
                deriv_1 = np.array([0, 0, 0])
        else:
            deriv_1 = (key_value_parsed[k_i_plus-1] - key_value_parsed[k_i_plus+1]) * 0.5

        print(f"SplinePositionInterpolator : k_i_before = {k_i_before}")
        print(f"SplinePositionInterpolator : k_i_plus = {k_i_plus}")

        c = np.array([
            key_value_parsed[k_i_before],
            key_value_parsed[k_i_plus],
            deriv_0,
            deriv_1
        ])

        print(f"SplinePositionInterpolator : set_fraction = {set_fraction}")
        print(f"SplinePositionInterpolator : key = {key}")
        print(f"SplinePositionInterpolator : keyValue = {keyValue}")
        print(f"SplinePositionInterpolator : closed = {closed}")
        print(f"SplinePositionInterpolator : c = \n{c}")

        # Calcular a interpolação
        value_changed = s_m @ cf.get_hermite_m() @ c

        print(f"SplinePositionInterpolator : value_changed = \n{value_changed}")

        return value_changed


    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação específicos."""
        key = np.array(key)
        keyValue = np.array(keyValue)

        # Find the correct interval for set_fraction
        k_i_before = 0
        for k_i_before in range(len(key) - 1):
            if key[k_i_before] <= set_fraction <= key[k_i_before + 1]:
                k_i_before = k_i_before
                k_i_plus = k_i_before + 1
                break

        key_value_parsed = keyValue.reshape(-1, 4)

        # Debug print statements (can be removed)
        """ print(f"OrientationInterpolator : key_value_parsed = \n{key_value_parsed}")
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))
        print("OrientationInterpolator : keyValue = {0}".format(keyValue)) """

        # Get the rotations at the two keyframes
        rotation_before = key_value_parsed[k_i_before]
        rotation_after = key_value_parsed[k_i_plus]

        # Compute the interpolation factor between the two keyframes
        t = (set_fraction - key[k_i_before]) / (key[k_i_plus] - key[k_i_before])

        # SLERP between rotation_before and rotation_after
        interpolated_rotation = cf.slerp(rotation_before, rotation_after, t)

        return interpolated_rotation

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""

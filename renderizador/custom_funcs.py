"""
Module: custom_funcs.py
This module contains custom functions for rendering.
Functions:
- faz_normal(vec): Calculates the normal vector of a given vector.
- produto_escalar(vec0, vec1): Calculates the dot product of two vectors.
- L(p0, p1, p): Determines if a point is on the left side of a line.
- dentro(p0, p1, p2, p): Determines if a point is inside a triangle.
- diamond(px, py, x, y): Determines if a point is inside a diamond shape.
- cameraToScreen(p, d): Converts a point from camera space to screen space.
- rotate_quat(u, theta): Rotates a point around a given axis using quaternion rotation.
- NDCToScreenMatrix(w, h): Generates a matrix for converting from NDC space to screen space.
- translationMatrix(x, y, z): Generates a translation matrix.
Note: This module also contains a main block with commented-out code for testing the functions.
"""

import numpy as np

def faz_normal(vec):
    """
    Calculates the normal vector of a given 2D vector.

    Args:
        vec (tuple): A tuple representing a 2D vector.

    Returns:
        tuple: A tuple representing the normal vector of the input vector.
    """
    a = vec[0]
    b = vec[1]
    return (b,-a)

def produto_escalar(vec0,vec1):
    """
    Calculates the dot product between two vectors.

    Parameters:
    vec0 (list): The first vector represented as a list of two elements.
    vec1 (list): The second vector represented as a list of two elements.

    Returns:
    float: The dot product of the two vectors.
    """
    return vec0[0]*vec1[0]+vec0[1]*vec1[1]

def l(p0,p1,p):
    """
    Determines if a point `p` is on the left side of the line segment defined by `p0` and `p1`.

    Parameters:
    - p0 (tuple): The starting point of the line segment.
    - p1 (tuple): The ending point of the line segment.
    - p (tuple): The point to be checked.

    Returns:
    - bool: True if the point is on the left side of the line segment, False otherwise.
    """
    vd = (p1[0]-p0[0],p1[1]-p0[1])
    vn = (vd[1],-vd[0])
    vp = (p[0]-p0[0],p[1]-p0[1])
    return produto_escalar(vp,vn) >= 0

def dentro(p0,p1,p2,p):
    """
    Determines if a point `p` is inside a triangle defined by three points `p0`, `p1`, and `p2`.

    Parameters:
    - p0: The first point of the triangle.
    - p1: The second point of the triangle.
    - p2: The third point of the triangle.
    - p: The point to check.

    Returns:
    - True if the point is inside the triangle, False otherwise.
    """
    l0 = l(p0,p1,p)
    l1 = l(p1,p2,p)
    l2 = l(p2,p0,p)
    return l0 and l1 and l2


def rotate_quat(rotation):
    """ 
    Rotate a point using a quaternion rotation.
    Parameters:
    u (numpy.ndarray): The rotation axis.
    theta (float): The angle of rotation.
    Returns:
    numpy.ndarray: The rotation matrix.
    p = point
    u = rotation axis
    theta = angle
    """


    qi = rotation[0]*np.sin(rotation[3]/2)
    qj = rotation[1]*np.sin(rotation[3]/2)
    qk = rotation[2]*np.sin(rotation[3]/2)
    qr = np.cos(rotation[3]/2)

    rm = [[1-2*(qj**2+qk**2),2*(qi*qj - qk*qr)   , 2*(qi*qk + qj*qr)     ,0],
          [2*(qi*qj + qk*qr),1-2*(qi**2 + qk**2) , 2*(qj*qk - qi*qr)     ,0],
          [2*(qi*qk - qj*qr),2*(qj*qk+qi*qr)     ,1-2*(qi**2 + qj**2)    ,0],
          [0                ,0                   ,0                      ,1]]
    return rm

def NDC_to_screen_matrix(w,h):
    """
    Generates a matrix for converting normalized device coordinates (NDC) to screen coordinates.

    Parameters:
    w (float): The width of the screen.
    h (float): The height of the screen.

    Returns:
    numpy.ndarray: The matrix for converting NDC to screen coordinates.
    """
    return np.array([[w/2, 0,  0, w/2],
                     [0,  -h/2,0, h/2],
                     [0,   0,  1, 0],
                     [0,   0,  0, 1]])

def translation_matrix(x,y,z):
    """
    Generates a translation matrix for a given translation vector (x, y, z).

    Parameters:
    x (float): The translation along the x-axis.
    y (float): The translation along the y-axis.
    z (float): The translation along the z-axis.

    Returns:
    numpy.ndarray: The translation matrix.

    """
    return np.array([[1,0,0,x],
                     [0,1,0,y],
                     [0,0,1,z],
                     [0,0,0,1]])

def area(p0,p1,p2):
    """
    Calculates the signed area of a triangle defined by three points.

    Parameters:
    - p0: The first point of the triangle.
    - p1: The second point of the triangle.
    - p2: The third point of the triangle.

    Returns:
    - float: The area of the triangle.
    """
    x1,y1 = p0
    x2,y2 = p1
    x3,y3 = p2
    return 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))

def calculate_barycentric_coordinates(x0, y0, x1, y1, x2, y2, x, y):
        a_total = area([x0, y0], [x1, y1], [x2, y2])
        a0 = area([x1, y1], [x2, y2], [x, y])
        a1 = area([x2, y2], [x0, y0], [x, y])
        a2 = area([x0, y0], [x1, y1], [x, y])
        alpha = abs(a0 / a_total)
        beta = abs(a1 / a_total)
        gamma = abs(a2 / a_total)
        return alpha,beta,gamma

def mipmap_level(del_u_del_x,del_u_del_y,del_v_del_x,del_v_del_y):
    l = max(np.sqrt(del_u_del_x**2 + del_v_del_x**2),np.sqrt(del_u_del_y**2 + del_v_del_y**2))
    return int(np.log2(l))

""" def generate_mipmap(image : np.array,level):
    if int(level) == 0 or image.shape[0] == 1 or image.shape[1] == 1:
        return image
    else:
        return generate_mipmap(image[::2,::2],int(level)-1) """

def generate_mipmap(image):
    """
    Generates a mipmap pyramid for a given image.

    Parameters:
    image (numpy.ndarray): Input image as a NumPy array (height, width, channels).

    Returns:
    mipmap_levels (list): List of mipmap levels where each level is a smaller version of the original image.
    """
    mipmap_levels = [image]  # Start with the original image as the first mipmap level

    current_image = image
    while current_image.shape[0] > 1 and current_image.shape[1] > 1:
        # Reduce image size by half in both dimensions
        new_height = max(1, current_image.shape[0] // 2)
        new_width = max(1, current_image.shape[1] // 2)

        # Downsample by averaging neighboring pixels (bilinear-like filtering)
        # Take a 2x2 block of pixels and compute their average for each channel
        reduced_image = np.zeros((new_height, new_width, current_image.shape[2]), dtype=current_image.dtype)

        for y in range(new_height):
            for x in range(new_width):
                # Average the 2x2 block of pixels
                block = current_image[2 * y:2 * y + 2, 2 * x:2 * x + 2]
                reduced_image[y, x] = np.mean(block, axis=(0, 1))

        # Append the reduced image to the mipmap levels
        mipmap_levels.append(reduced_image)

        # Update current_image to the newly reduced image for the next iteration
        current_image = reduced_image

    return mipmap_levels

def sphere(raio, div_lon, div_lat):
    """
    Generates a 3D sphere mesh with specified radius and divisions.
    Args:
        raio (float): The radius of the sphere.
        div_lon (int): The number of longitudinal divisions (slices).
        div_lat (int): The number of latitudinal divisions (stacks).
    Returns:
        tuple: A tuple containing:
            - triangles (numpy.ndarray): A flattened array of triangle vertices.
            - triangle_normals (numpy.ndarray): A flattened array of triangle normals.
    """
    points = []
    triangles = []
    normals = []
    triangle_normals = []

    delta_theta = 2 * np.pi / div_lon
    delta_phi = np.pi / div_lat

    # Gerando pontos da esfera
    for i in range(div_lon + 1):
        theta = i * delta_theta
        for j in range(1, div_lat):  # Evitamos os pólos direto aqui
            phi = j * delta_phi

            x = raio * np.sin(phi) * np.cos(theta)
            y = raio * np.sin(phi) * np.sin(theta)
            z = raio * np.cos(phi)
            points.append([x, y, z])

            norm = np.array([x, y, z]) / np.linalg.norm([x, y, z])
            normals.append(norm)

    # Adiciona o pólo superior e inferior
    points.append([0, 0, raio])  # Pólo norte
    normals.append([0, 0, 1])

    points.append([0, 0, -raio])  # Pólo sul
    normals.append([0, 0, -1])

    pole_north = len(points) - 2
    pole_south = len(points) - 1

    # Conectando vértices em triângulos
    for i in range(div_lon):
        for j in range(div_lat - 2):
            # Índices dos vértices do triângulo (dois triângulos por quad)
            p1 = i * (div_lat - 1) + j
            p2 = p1 + div_lat - 1
            p3 = p1 + 1
            p4 = p2 + 1

            # Primeiro triângulo
            triangles.append([points[p1], points[p2], points[p3]])
            triangle_normals.append([normals[p1], normals[p2], normals[p3]])
            # Segundo triângulo
            triangles.append([points[p3], points[p2], points[p4]])
            triangle_normals.append([normals[p3], normals[p2], normals[p4]])

        # Conectar os triângulos com o pólo norte
        triangles.append([points[pole_north], points[i * (div_lat - 1)], points[(i + 1) % div_lon * (div_lat - 1)]])
        triangle_normals.append([normals[pole_north], normals[i * (div_lat - 1)], normals[(i + 1) % div_lon * (div_lat - 1)]])

        # Conectar os triângulos com o pólo sul
        triangles.append([points[pole_south], points[(i + 1) % div_lon * (div_lat - 1) + (div_lat - 2)], points[i * (div_lat - 1) + (div_lat - 2)]])
        triangle_normals.append([normals[pole_south], normals[(i + 1) % div_lon * (div_lat - 1) + (div_lat - 2)], normals[i * (div_lat - 1) + (div_lat - 2)]])

    triangles = np.array(triangles).flatten()
    triangle_normals = np.array(triangle_normals).flatten()

    return triangles, triangle_normals


def cone(bottom_radius, height):
    """
    Generates a 3D cone mesh with the specified bottom radius and height.
    Args:
        bottom_radius (float): The radius of the cone's base.
        height (float): The height of the cone.
    Returns:
        np.ndarray: A flattened array of triangles representing the cone mesh. Each triangle is defined by three points.
    """
    points = []
    triangles = []
    
    div_lon = 20  # Número de divisões longitudinais (fatias ao redor)
    delta_theta = 2 * np.pi / div_lon  # Ângulo entre as fatias

    # Adicionar o vértice do topo
    top_vertex = [0, height, 0]
    points.append(top_vertex)

    # Gerar vértices ao redor da base
    for i in range(div_lon):
        theta = i * delta_theta
        x = bottom_radius * np.cos(theta)
        z = bottom_radius * np.sin(theta)
        points.append([x, 0, z])

    # Adicionar o vértice central da base
    base_center = [0, 0, 0]
    points.append(base_center)

    # Conectar o topo aos vértices da base para formar as faces laterais
    for i in range(div_lon):
        p1 = 0  # Vértice do topo
        p2 = i + 1  # Vértice na base
        p3 = (i + 1) % div_lon + 1  # Próximo vértice na base

        # Triângulo lateral
        triangles.append([points[p1], points[p2], points[p3]])

    # Conectar os vértices da base ao centro da base para formar a base do cone
    base_center_index = len(points) - 1
    for i in range(div_lon):
        p1 = base_center_index  # Vértice central da base
        p2 = i + 1  # Vértice na base
        p3 = (i + 1) % div_lon + 1  # Próximo vértice na base

        # Triângulo da base
        triangles.append([points[p1], points[p3], points[p2]])

    triangles = np.array(triangles).flatten()

    return triangles

def box(size):
    sx, sy, sz = size
    # Define the 8 vertices of the box
    vertices = [
        # Front face
        -sx / 2, -sy / 2, sz / 2,
        sx / 2, -sy / 2, sz / 2,
        sx / 2, sy / 2, sz / 2,
        -sx / 2, sy / 2, sz / 2,
        # Back face
        -sx / 2, -sy / 2, -sz / 2,
        sx / 2, -sy / 2, -sz / 2,
        sx / 2, sy / 2, -sz / 2,
        -sx / 2, sy / 2, -sz / 2,
    ]


    # Define the 12 triangles of the box
    triangles = [
        # Front face
        0, 1, 2,
        0, 2, 3,
        # Back face
        4, 6, 5,
        4, 7, 6,
        # Top face
        3, 2, 6,
        3, 6, 7,
        # Bottom face
        0, 4, 1,
        1, 4, 5,
        # Right face
        1, 5, 2,
        2, 5, 6,
        # Left face
        0, 3, 7,
        0, 7, 4,
    ]

    point = []
    for i in range(len(triangles)):
        point.append(vertices[triangles[i]*3])
        point.append(vertices[triangles[i]*3+1])
        point.append(vertices[triangles[i]*3+2])
    #print(f"point = {point}")
    
    return np.array(point)

def cylinder(radius,height):
    points = []
    triangles = []
    
    div_lon = 20
    div_lat = 20

    delta_theta = 2*np.pi / div_lon
    delta_phi = height / div_lat

    for i in range(div_lon + 1):
        theta = i * delta_theta
        for j in range(div_lat + 1):
            phi = j * delta_phi

            x = radius * np.cos(theta)
            y = phi
            z = radius * np.sin(theta)
            points.append([x, y, z])

    # Agora conectando os vértices em triângulos
    for i in range(div_lon):
        for j in range(div_lat):
            # Índices dos vértices do triângulo (dois triângulos por quad)
            p1 = i * (div_lat + 1) + j
            p2 = p1 + div_lat + 1
            p3 = p1 + 1
            p4 = p2 + 1

            # Primeiro triângulo
            triangles.append([points[p1], points[p2], points[p3]])
            # Segundo triângulo
            triangles.append([points[p3], points[p2], points[p4]])
    
    triangles = np.array(triangles).flatten()

    return triangles

def vector_to_color(normal):
    return [int((normal[0] + 1) * 127.5), int((normal[1] + 1) * 127.5), int((normal[2] + 1) * 127.5)]

def color_to_vector(color):
    return [(color[0] / 127.5) - 1, (color[1] / 127.5) - 1, (color[2] / 127.5) - 1]

def get_hermite_m():
    return np.array([
        [ 2, -2,  1,  1],
        [-3,  3, -2, -1],
        [ 0,  0,  1,  0],
        [ 1,  0,  0,  0]
    ])


def clear_framebuffer(buffer):
    """
    Clears the framebuffer by setting all pixels to the background color.

    Parameters:
    buffer (numpy.ndarray): The framebuffer to clear.
    """
    buffer.fill(0)

def slerp(rotation_before, rotation_after, t):
    """Performs spherical linear interpolation (SLERP) between two rotations."""
    # Extract the axis and angle components
    axis_before = rotation_before[:3]
    angle_before = rotation_before[3]

    axis_after = rotation_after[:3]
    angle_after = rotation_after[3]

    # Calculate the shortest path between the two angles
    # Normalize the rotation axes (just in case)
    axis_before = axis_before / np.linalg.norm(axis_before)
    axis_after = axis_after / np.linalg.norm(axis_after)

    # Ensure the two axes are aligned for interpolation (they should be for rotations)
    if not np.allclose(axis_before, axis_after):
        print("Warning: Axes are not aligned. Interpolating using axis of first keyframe.")
        axis_after = axis_before

    # Perform linear interpolation on the angle component (SLERP for angle)
    interpolated_angle = (1 - t) * angle_before + t * angle_after

    # Return the interpolated rotation (axis remains the same, only the angle changes)
    return np.hstack([axis_before, interpolated_angle])

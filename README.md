# Computer Graphics AUTh 2023

This repository contains the source code for the three exercises of the course Computer Graphics, taken at the school of Electrical and Computer Engineering during the summer semester of 2023.

# Exercise 1

In the first exercise, two methods are used to shade triangles in a canvas with given vertices and colors.

1. 'flats': This method implements flat shading. It calculates the average color of the triangle and sets the color of all pixels within the triangle to this average color.

2. 'gourauds': This method implements Gouraud shading. It interpolates the colors of the vertices in both the y and x directions. This results in a smooth gradient of colors across the triangle, which gives a more realistic appearance compared to flat shading.

# Exercise 2

The functions implemented for this exercise perform a series of transformations and calculations to prepare the 3D object for rendering. 

The process includes:
 * calculating the rotation matrix for a given angle and unit vector. This matrix is used to rotate points in 3D space.

 * rotating and translating a set of points. We first center the points, then apply the rotation matrix, and finally translate the points.

 * changing the coordinate system of a set of points. 

 * calculating the camera's coordinate system, then projecting the points onto the image plane.

 * rasterizing the 2d points into pixel coordinates, then translating them so that the center of the points is at the center of the image.

We then call the shade_triangle method from Exercise 1 to handle shading.

# Exercise 3

Exercise 3 introduces more advanced shading and lighting techniques to create a more realistic rendering of the 3D object. Namely, we now use:

 * phong shading, which is a more advanced shading technique compared to Gouraud shading used in Exercise 2. Phong shading considers the angle of light and the viewer's position to create a more realistic rendering.

 * phong materials and point lights. 'PhongMaterial' represents the material properties of the object being rendered, including ambient, diffuse, and specular coefficients, and the Phong exponent. 'PointLight' represents a light source in the scene, including its position and intensity.

 * lighting models, including `ambient`, `diffusion`, `specular`, `all`. These models determine how the light interacts with the object's material.

These new methods produce higher quality renders, and the difference between the results of Exercise 1 and Exercise 3 are shown below.

# Image from Exercise 1
![exercise_1](https://github.com/toniker/computer-graphics-2023/assets/39350193/891d95b3-9eca-430d-b88a-9d7a99472e78)

# Image from Exercise 3
![exercise_3](https://github.com/toniker/computer-graphics-2023/assets/39350193/d4a4cdfb-540f-40cc-afe7-8df9a8743894)

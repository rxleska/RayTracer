#This document is a guide to the scene file format created for the project. It is pretty simple and follows this structure:

The Scene is Broken up into Sections. Each section is started by a keyword and a new line. Then ended by a keyword and a new line. The data is then stored in the section. 


### Camera Section
```txt
Camera
Aspect Ratio in format "width height"
Look From Format "x y z" floating point precision
Look At Format "x y z" floating point precision
distance_to_focus "t" floating point precision (-1 will default t to lookfrom - lookat length)
aperture "t" floating point precision
```

Resolution will be calculated based on the aspect ratio and a width provided by the user. The user will also provide a sample rate and a msaa multiplier. The msaa multiplier increases the samples near edges by the multiple. 

Sample
```txt  
Camera
500 500
278.0 278.0 -400.0
278.0 278.0 0.0
10.0
0.1
```

### Material Section


Materials 

Lambertian "name" "r g b" [0,1] floating point precision

LambertianBordered "name" "r1 g1 b1 r2 g2 b2" [0,1] floating point precision rgb1 is the color of the material and rgb2 is the border color if no border color is provided it will default to black

Metal "name" "r g b f" [0,1] floating point precision f is the fuzziness of the material

Dielectric "name" "r g b ior" [0,1] floating point precision ior is the index of refraction

Light "name" "r g b" [0,MAX_FLOAT] floating point precision

Textured "name" "path" path to the texture file (path from running directory currently only ppm files are supported)

Sample
```txt
Materials
Lambertian red 0.9 0.7 0.5
LambertianBordered wall 0.5 0.5 0.5 0.0 0.0 0.0
Metal mirror 0.8 0.8 0.8 0.0
Dielectric glass 0.9 0.9 0.9 1.5
Light light 15.0 15.0 15.0
Textured monkey "imTexts/Monkey.ppm"
```


### Object Section

Objects

Sphere "center_x center_y center_z radius" "material_name" floating point precision radius 

Polygon "v1x v1y v1z v2x v2y v2z ... vnz" "material_name" floating point precision

Polygon "v1x v1y v1z v2x v2y v2z ... vnz" Textured "v1u v1v v2u v2v ... vnv" floating point precision

Mesh "path" "material_name" floating point precision (path from running directory) does not support textures


Sample
```txt
Objects
Sphere 278.0 278.0 0.0 100.0 wall
Polygon 0.0 0.0 0.0 555.0 0.0 0.0 555.0 0.0 555.0 0.0 0.0 555.0 wall
Polygon 0.0 0.0 0.0 -555.0 0.0 0.0 -555.0 0.0 -555.0 0.0 0.0 -555.0 monkey 1.0 1.0 0.0 1.0 0.0 0.0 1.0 0.0  
```
# RayTracer in CUDA C++

This is a ray tracer implemented in CUDA C++. It is based on the book "Ray Tracing in One Weekend" by Peter Shirley. [Here](https://raytracing.github.io) is the link to the book. For the most part I used the textbook as a refrence for math and light physics. Due to the nature of CUDA, the structure of the code is quite different from the code in the book but the underlying principles are the same.

## Features

- Camera Configuration
    - Position
    - Depth of Field
    - Field of View
- Anti-Aliasing 
    - Edge and quality detection for anti-aliasing (adaptive sampling)
- Hittable Objects
    - Sphere
    - Polygons
        - Triangle
        - Rectangle
        - n-gons
    - Meshes (.gltf files only) (converted to polygons)
    - Boxes
    - Box Continuous Mediums
- Materials
    - Lambertian
    - Metal
    - Dielectric
    - Light
    - Textured (ppm files only)
    - IsoTropic
    - Phong (experimental must specify point light sources)
    - Phong Lambertian (experimental must specify point light sources)
    - Lambertian Bordered (bordered by a different color)
- Octree Acceleration Data Structure (A BVH might have been better but I wanted to try something different)
- Importance Sampling
    - Direct Light Sampling
    - Cosine Weighted Sampling
- Hittable Object Instancing 
    - Translation
    - Rotation
    - Motion Blur
    
## Rendering Examples

### Cornell Box (2k, 200 Samples per Pixel, Max Depth 10) (95.6 Seconds To Render)

![Cornell Box](./BestImages/CornellBox2k200spls10bnc.png)


### Final Render (8k, 200 Samples per Pixel, Max Depth 20) (~ 2 Hours To Render)

![Final Render](./BestImages/Final.png)
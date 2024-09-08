# RayTracer in C++

This is a ray tracer I wrote in C++ for a computer graphics course. 

The course follows the book ["Ray Tracing in One Weekend"](https://raytracing.github.io/) by Peter Shirley. 
 
I started by following book 1 directly, then I worked on adding custom features and optimizations, while reading through the other two books in the series.


## Features
- [x] Basic ray tracing
- [-] Materials
    - [x] Lambertian
    - [x] Metal
    - [x] Dielectric
    - [x] Diffuse light (kinda done, emission cannot be combined with other materials) 
    - [ ] png/image mapped materials
- [x] Camera
    - [x] positionable
    - [x] focusable
    - [x] aperture
- [x] Multi-threading
- [ ] Optimization Data Structures
    - [ ] Bounding Volume Hierarchy 
    - [ ] Octree 
- [x] hittable objects
    - [x] Sphere
    - [x] Polygons (imo this is better than creating quadralaterals)
- [ ] other ideas
    - [ ] live render preview 
    - [ ] dynamic sample rate anti-aliasing (edges of objects send signal to increase sample rate like msaa)



## preconfigured scenes 
- [x] Cornell Box (2 rectangular prisums)
    - Seen here 1000x1000px, 1000 samples per pixel, 100 Max bounce Depth ![Cornell Box](BestImages\CornellBox1000sols1000px100bnc.png)
- [x] Ray Tracing in One Weekend Cover (modified to have metal ball be gold and there is a light in the sky)
    - Seen here 3840x2160px, 64 samples per pixel, 100 Max bounce Depth ![Ray Tracing in One Weekend Cover](BestImages\4k64x.png)
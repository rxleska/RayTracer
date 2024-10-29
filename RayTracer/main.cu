// hittable
#include "lib/hittable/headers/Hitable.hpp"
#include "lib/hittable/headers/HitRecord.hpp"
#include "lib/hittable/headers/Octree.hpp"
#include "lib/hittable/headers/Sphere.hpp"
#include "lib/hittable/headers/Polygon_T.hpp"
#include "lib/hittable/headers/Octree.hpp"
// materials
#include "lib/materials/headers/Material.hpp"
#include "lib/materials/headers/Lambertian.hpp"
#include "lib/materials/headers/Metal.hpp"
#include "lib/materials/headers/Dielectric.hpp"
#include "lib/materials/headers/Light.hpp"
#include "lib/materials/headers/LambertianBordered.hpp"
#include "lib/materials/headers/Textured.hpp"
#include "lib/materials/headers/Phong.hpp"
#include "lib/materials/headers/PhongLamb.hpp"
// processing
#include "lib/processing/headers/Camera.hpp"
#include "lib/processing/headers/Ray.hpp"
#include "lib/processing/headers/Vec3.hpp"

#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <fstream>

#include <vector>

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "lib/external/tiny_gltf.h"

#define MAX_OBJECTS 1000

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << cudaGetErrorString(result) << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// __global__ void rand_init(curandState *rand_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         curand_init(1984, 0, 0, rand_state);
//     }
// }

__global__ void rand_init_singleton(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void init_texture(Vec3 *textures, float *texture, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        textures[idx] = Vec3(texture[idx * 3], texture[idx * 3 + 1], texture[idx * 3 + 2]);
    }
}


__host__ void allocate_texture(const char *filename, Vec3 **textures, int texture_index) {
    int width, height;
    float *im1 = load_texture(filename, width, height);

    // Allocate memory for device image and copy
    float *d_im1;
    checkCudaErrors(cudaMalloc((void **)&d_im1, width * height * 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_im1, im1, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate memory for the texture on device and initialize in parallel
    Vec3 *d_texture;
    checkCudaErrors(cudaMalloc((void **)&d_texture, width * height * sizeof(Vec3)));
    init_texture<<<(width * height + 255) / 256, 256>>>(d_texture, d_im1, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&textures[texture_index], &d_texture, sizeof(Vec3 *), cudaMemcpyHostToDevice));

    free(im1);
}

__global__ void init_mesh(Vec3 *d_mesh, float *mesh, int num_points, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        d_mesh[idx] = Vec3(mesh[idx * 3], mesh[idx * 3 + 1], mesh[idx * 3 + 2]) * scale;
    }
}

__global__ void place_mesh(Vec3 **meshes, int *mesh_lengths, Vec3 * d_mesh, int mesh_index, int mesh_length) {
   if(threadIdx.x == 0 && blockIdx.x == 0){
        meshes[mesh_index] = d_mesh;
        mesh_lengths[mesh_index] = mesh_length;
    }

}


/*
* Load mesh from file
* gltf_position is found in the .gltf file "meshes": [ { "primitives": [ { "attributes": { "POSITION": 0 } } ] } ]
* gltf_bufferView is found in the .gltf file "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 0 } ]
*/
__host__ void allocate_mesh(const char *filename, Vec3 **meshes, int mesh_index, int *mesh_length) {

    //use tinygltf to load the mesh
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        std::cout << "Warn: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "Err: " << err << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to parse glTF\n";
        return;
    }

    std::vector<float> floatBuffer;

    for (const auto &mesh : model.meshes) {
        for (const auto &primitive : mesh.primitives) {
            // Ensure the primitive is a triangle mesh
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                continue;
            }

            const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];

            const tinygltf::Accessor &positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView &positionBufferView = model.bufferViews[positionAccessor.bufferView];
            const tinygltf::Buffer &positionBuffer = model.buffers[positionBufferView.buffer];

            // Index data type (unsigned short, unsigned int, etc.)
            const unsigned char *indexData = indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;
            size_t indexStride = tinygltf::GetComponentSizeInBytes(indexAccessor.componentType);

            // Position data (vec3)
            const unsigned char *positionData = positionBuffer.data.data() + positionBufferView.byteOffset + positionAccessor.byteOffset;
            size_t positionStride = positionAccessor.ByteStride(positionBufferView);

            // Iterate over indices in groups of 3 (for triangles)
            for (size_t i = 0; i < indexAccessor.count; i += 3) {
                // Get the 3 indices of the triangle
                unsigned int indices[3];
                for (size_t j = 0; j < 3; ++j) {
                    if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        indices[j] = ((const unsigned short *)indexData)[i + j];
                    } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        indices[j] = ((const unsigned int *)indexData)[i + j];
                    } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        indices[j] = ((const unsigned char *)indexData)[i + j];
                    }
                }

                // Fetch the positions corresponding to the indices
                for (size_t j = 0; j < 3; ++j) {
                    const float *vertex = (const float *)(positionData + indices[j] * positionStride);
                    floatBuffer.push_back(vertex[0]);
                    floatBuffer.push_back(vertex[1]);
                    floatBuffer.push_back(vertex[2]);
                }
            }
        }
    }

    //output file
    // FILE *f = fopen("mesh.txt", "w");
    // for(int i = 0; i < floatBuffer.size()/3; i++){
    //     fprintf(f, "%f %f %f\n", floatBuffer[i*3], floatBuffer[i*3+1], floatBuffer[i*3+2]);
    // }
    // fclose(f);

    // // //print out all the vertices
    // for(int i = 0; i < floatBuffer.size(); i++){
    //     printf("%f %f %f\n", floatBuffer[i*3], floatBuffer[i*3+1], floatBuffer[i*3+2]);
    // }

    int gltf_accessor = floatBuffer.size() / 3;
    //convert vector to array
    float *floatBufferArray = floatBuffer.data();

    // // Allocate memory for device image and copy
    float *d_mesh_f; 
    checkCudaErrors(cudaMalloc((void **)&d_mesh_f, gltf_accessor * 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_mesh_f, floatBufferArray, gltf_accessor * 3 * sizeof(float), cudaMemcpyHostToDevice));
    Vec3 *d_mesh;
    checkCudaErrors(cudaMalloc((void **)&d_mesh, gltf_accessor * sizeof(Vec3)));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    init_mesh<<<(gltf_accessor + 255) / 256, 256>>>(d_mesh, d_mesh_f, gltf_accessor, 2.0f);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    place_mesh<<<1,1>>>(meshes, mesh_length, d_mesh, mesh_index, gltf_accessor);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
} 


__device__ Vec3 getColor(const Ray &r, Camera **cam, Scene **world, curandState *local_rand_state, bool &edge_hit) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0,1.0,1.0);
    for(int i = 0; i < (*cam)->bounces; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            int did_scatter = rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state);
            edge_hit = rec.edge_hit;
            if(did_scatter == 1) {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else if(did_scatter == 2){ //light hit return color
                return cur_attenuation * attenuation;
            }
            else if(did_scatter == 3) { //phong hit return color
                return (*world)->handlePhong(rec, cam) * cur_attenuation;
            }
            else if(did_scatter == 4) { //phong hit return color
                int bcCount = ((PhongLamb*) rec.mat)->bc;
                if(bcCount < i){
                    return (*world)->handlePhongLamb(rec, cam, scattered, local_rand_state, true) * cur_attenuation;
                }
                else{
                    cur_attenuation = cur_attenuation * (*world)->handlePhongLamb(rec, cam, scattered, local_rand_state, false);
                    cur_ray = scattered;
                }
            }
            else {
                return Vec3(0.0,0.0,1.0);
            }
        }
        else {
            float ambient = (*cam)->ambient_light_level;
            Vec3 unit_direction = (cur_ray.direction).normalized();
            float t = 0.5f*(unit_direction.y + 1.0f);
            Vec3 c = Vec3(1.0, 1.0, 1.0)*(1.0f-t) + Vec3(0.5, 0.7, 1.0)*t;
            return cur_attenuation * (c*ambient);
        }
    }
    return Vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void free_world(Hitable **device_object_list, Scene **d_world, Camera **d_camera) {
    for(int i=0; i < (*d_world)->hitable_count; i++) {
        delete ((Sphere *)device_object_list[i])->mat;
        delete device_object_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void rand_init_render(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ float clamp(float x, float min, float max) {
    if(x < min) return min;
    if(x > max) return max;
    return x;
}

__global__ void render(uint8_t *fb, int max_x, int max_y, int ns, Camera **cam, Scene **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = (max_y - j - 1)*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0,0,0);
    bool edge_hit = false;
    bool edge_hit_check = false;
    int samples = (*cam)->samples;
    for(int s=0; s < samples; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col = col + getColor(r, cam, world, &local_rand_state, edge_hit_check);

        if(!edge_hit && edge_hit_check) {
            edge_hit = true;
            samples = samples * (*cam)->msaa_x;
        }
    }
    rand_state[pixel_index] = local_rand_state;
    col = col / float(samples);
    

    fb[pixel_index*3+0] = uint8_t(int(255.99*clamp(sqrt(col.x), 0.0f, 1.0f)));
    fb[pixel_index*3+1] = uint8_t(int(255.99*clamp(sqrt(col.y), 0.0f, 1.0f)));
    fb[pixel_index*3+2] = uint8_t(int(255.99*clamp(sqrt(col.z), 0.0f, 1.0f)));
}

#define RND (curand_uniform(&local_rand_state))


#include "lib/Scenes/TestScene.hpp"
#include "lib/Scenes/RTIAW.hpp"
#include "lib/Scenes/CornellBox.hpp"
#include "lib/Scenes/PhongCornellBox.hpp"
#include "lib/Scenes/PhongMixCornellBox.hpp"
#include "lib/Scenes/CornellRoomOfMirrors.hpp"
#include "lib/Scenes/Billards.hpp"

__global__ void create_world(Hitable **device_object_list, Scene **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, Vec3 **textures, int num_textures, Vec3 ** meshes, int * mesh_lengths, int num_meshes){
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        create_RTIAW_sample(device_object_list, d_world, d_camera, nx, ny, rand_state);
        // create_test_scene(device_object_list, d_world, d_camera, nx, ny, rand_state, textures, num_textures, meshes, mesh_lengths, num_meshes);
        // create_Cornell_Box_Octree(device_object_list, d_world, d_camera, nx, ny, rand_state);
        // create_Cornell_Box_Octree_ROM(device_object_list, d_world, d_camera, nx, ny, rand_state, textures, num_textures, meshes, mesh_lengths, num_meshes);
        // create_Billards_Scene(device_object_list, d_world, d_camera, nx, ny, rand_state, textures, num_textures, meshes, mesh_lengths, num_meshes);
        // create_Phong_Cornell_Box_Octree(device_object_list, d_world, d_camera, nx, ny, rand_state);
        // create_Phong_Mix_Cornell_Box_Octree(device_object_list, d_world, d_camera, nx, ny, rand_state);
    }
}


int main() {
    //increase stack size
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    // int nx = 512*8;
    // int nx = 500*1;
    int nx = 1440;
    // int ny = 512*8;
    // int ny = 500*1;
    int ny = 900;
    int ns = 100;
    // int tx = 20;
    // int ty = 12;
    // int tx = 16;
    // int ty = 10;
    int tx = 512;
    int ty = 1;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    // size_t fb_size = num_pixels*sizeof(vec3);
    size_t fb_size = num_pixels*sizeof(uint8_t)*3;

    // allocate Frame Buffer (fb)
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // timing
    clock_t start, stop;
    start = clock();


    // allocate random state for each pixel
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    // initialize random state for Octree generation
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init_singleton<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //LOAD IMAGES
    int num_textures = 2;
    Vec3 **textures; // array of texture arrays
    checkCudaErrors(cudaMalloc((void **)&textures, num_textures * sizeof(Vec3 *)));

    // Load image
    allocate_texture("imTexts/Monkey.ppm", textures, 0);
    allocate_texture("imTexts/ExampleImage.ppm", textures, 1);


    // LOAD MESHES
    int num_meshes = 2;
    Vec3 **meshes; // array of mesh point arrays
    int *num_points_meshes; // array of number of points in each mesh
    checkCudaErrors(cudaMalloc((void **)&meshes, num_meshes * sizeof(Vec3 *)));
    checkCudaErrors(cudaMalloc((void **)&num_points_meshes, num_meshes * sizeof(int)));

    allocate_mesh("meshFiles/man/scene.gltf", meshes, 0, num_points_meshes);
    allocate_mesh("meshFiles/knight/scene.gltf", meshes, 1, num_points_meshes);


    // make our world of hitables & the camera
    Hitable **device_object_list;
    // int num_hitables = 22*22+1+3;
    int num_hitables = MAX_OBJECTS;
    checkCudaErrors(cudaMalloc((void **)&device_object_list, num_hitables*sizeof(Hitable *)));
    Scene **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Octree *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    create_world<<<1,1>>>(device_object_list, d_world, d_camera, nx, ny, d_rand_state2, textures, num_textures, meshes, num_points_meshes, num_meshes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // print world created 
    printf("World created\n");
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    rand_init_render<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    start = clock();

    //open file
    FILE *f = fopen("image.ppm", "wb");
    fprintf(f, "P6 %d %d 255\n", nx, ny);
    uint8_t *fb2 = (uint8_t *)malloc(fb_size*3);
    //direct memory copy
    checkCudaErrors(cudaMemcpy(fb2, fb, fb_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    fwrite(fb2, sizeof(uint8_t), 3*nx*ny, f);
    fclose(f);

    stop = clock();
    timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "output took " << timer_seconds << " seconds.\n";

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(device_object_list, d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(device_object_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
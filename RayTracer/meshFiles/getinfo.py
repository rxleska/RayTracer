import json

# read info needed for the mesh

# read gltf as json
with open('meshFiles/scene.gltf') as f:
    gltf = json.load(f)

    # C++ Json::Value& primitive = json["meshes"][0]["primitives"][0];
    primitive = gltf["meshes"][0]["primitives"][0]

    #Json::Value& positionAccessor = json["accessors"][primitive["attributes"]["POSITION"].asInt()];
    positionAccessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
    print("positionAccessor: ")
    print(positionAccessor)

    #Json::Value& bufferView = json["bufferViews"][positionAccessor["bufferView"].asInt()];
    bufferView = gltf["bufferViews"][positionAccessor["bufferView"]]
    print("bufferView: ")
    print(bufferView)

    #bufferView["byteOffset"].asInt()
    byteOffset = bufferView["byteOffset"]
    print("byteOffset: ")
    print(byteOffset)

    #positionAccessor["count"].asInt()
    count = positionAccessor["count"]
    print("count: ")
    print(count)
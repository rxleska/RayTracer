import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Function to read points from mesh.txt and group them in sets of 3
def read_points(file_name):
    points = []
    with open(file_name, 'r') as file:
        for line in file:
            # Convert the line into a list of floats
            point = list(map(float, line.strip().split()))
            points.append(point)
    return points

# Function to draw polygons for each group of 3 points
def draw_polygons(points):
    # Create a new 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Group every 3 points to form a triangle (polygon)
    for i in range(0, len(points), 3):
        if i + 2 < len(points):
            polygon = [points[i], points[i+1], points[i+2]]
            # Add polygon to the plot
            ax.add_collection3d(Poly3DCollection([polygon], facecolors='red', edgecolors='black', linewidths=1, alpha=0.9))

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()

# Main function to read file and draw polygons
def main():
    file_name = 'test/mesh.txt'
    points = read_points(file_name)
    draw_polygons(points)

if __name__ == "__main__":
    main()

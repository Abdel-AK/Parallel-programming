#include "naive_cuda_simulation.cuh"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_wrappers.cuh"
#include "simulation/constants.h"

#include <exception>

void NaiveCudaSimulation::allocate_device_memory(Universe &universe, void **d_weights, void **d_forces, void **d_velocities, void **d_positions)
{
    auto error = parprog_cudaMalloc(d_weights, universe.num_bodies * sizeof(double));
    error = parprog_cudaMalloc(d_forces, universe.num_bodies * sizeof(double2));
    error = parprog_cudaMalloc(d_velocities, universe.num_bodies * sizeof(double2));
    error = parprog_cudaMalloc(d_positions, universe.num_bodies * sizeof(double2));
}

void NaiveCudaSimulation::free_device_memory(void **d_weights, void **d_forces, void **d_velocities, void **d_positions)
{
    auto error = parprog_cudaFree(*d_weights);
    d_weights = nullptr;
    error = parprog_cudaFree(*d_forces);
    d_forces = nullptr;
    error = parprog_cudaFree(*d_velocities);
    d_velocities = nullptr;
    error = parprog_cudaFree(*d_positions);
    d_positions = nullptr;
}

void NaiveCudaSimulation::copy_data_to_device(Universe &universe, void *d_weights, void *d_forces, void *d_velocities, void *d_positions)
{
    auto error = parprog_cudaMemcpy(d_weights, universe.weights.data(), universe.num_bodies * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    // convert the Vector2d<double> to double2
    std::vector<double2> converted{universe.num_bodies};
    for (int i = 0; i < universe.num_bodies; ++i)
    {
        converted[i] = make_double2(universe.forces[i][0], universe.forces[i][1]);
    }

    error = parprog_cudaMemcpy(d_forces, converted.data(), universe.num_bodies * sizeof(double2), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    for (int i = 0; i < universe.num_bodies; ++i)
    {
        converted[i] = make_double2(universe.velocities[i][0], universe.velocities[i][1]);
    }
    error = parprog_cudaMemcpy(d_velocities, converted.data(), universe.num_bodies * sizeof(double2), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    for (int i = 0; i < universe.num_bodies; ++i)
    {
        converted[i] = make_double2(universe.positions[i][0], universe.positions[i][1]);
    }
    error = parprog_cudaMemcpy(d_positions, converted.data(), universe.num_bodies * sizeof(double2), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void NaiveCudaSimulation::copy_data_from_device(Universe &universe, void *d_weights, void *d_forces, void *d_velocities, void *d_positions)
{
    auto error = parprog_cudaMemcpy(universe.weights.data(), d_weights, universe.num_bodies * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    std::vector<double2> converted{universe.num_bodies};

    error = parprog_cudaMemcpy(converted.data(), d_forces, universe.num_bodies * sizeof(double2), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    for (int i = 0; i < universe.num_bodies; ++i)
    {
        universe.forces[i].set(converted[i].x, converted[i].y);
    }

    error = parprog_cudaMemcpy(converted.data(), d_velocities, universe.num_bodies * sizeof(double2), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    for (int i = 0; i < universe.num_bodies; ++i)
    {
        universe.velocities[i].set(converted[i].x, converted[i].y);
    }

    error = parprog_cudaMemcpy(converted.data(), d_positions, universe.num_bodies * sizeof(double2), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (error != 0)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    for (int i = 0; i < universe.num_bodies; ++i)
    {
        universe.positions[i].set(converted[i].x, converted[i].y);
    }
}

__global__ void calculate_forces_kernel(std::uint32_t num_bodies, double2 *d_positions, double *d_weights, double2 *d_forces)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_bodies)
        return;
    double2 body_position = d_positions[i];
    // get body mass
    double body_mass = d_weights[i];

    double2 applied_force_vector = make_double2(0, 0);

    for (int distant_body_idx = 0; distant_body_idx < num_bodies; distant_body_idx++)
    {
        if (i == distant_body_idx)
        {
            continue;
        }
        // get distant body positions
        double2 distant_body_position = d_positions[distant_body_idx];

        // calculate vector between bodies to get the direction of the gravitational force
        double2 direction_vector = make_double2(distant_body_position.x - body_position.x, distant_body_position.y - body_position.y);

        // calculate the distance between the bodies
        double distance = sqrt(pow(direction_vector.x, 2) + pow(direction_vector.y, 2));

        // calculate gravitational force between the bodies
        double force = gravitational_constant * (body_mass * d_weights[distant_body_idx]) / pow(distance, 2);

        // create the force vector
        double2 force_vector = make_double2(direction_vector.x * (force / distance), direction_vector.y * (force / distance));

        // sum forces applied to body
        applied_force_vector = make_double2(applied_force_vector.x + force_vector.x, applied_force_vector.y + force_vector.y);
    }

    // store applied force
    d_forces[i] = applied_force_vector;
}

void NaiveCudaSimulation::calculate_forces(Universe &universe, void *d_positions, void *d_weights, void *d_forces)
{
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;

    if (universe.num_bodies % block_dim == 0)
    {
        grid_dim = universe.num_bodies / block_dim;
    }
    else
    {
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }
    calculate_forces_kernel<<<grid_dim, block_dim>>>(universe.num_bodies, (double2 *)d_positions, (double *)d_weights, (double2 *)d_forces);
    cudaDeviceSynchronize();
}

__global__ void calculate_velocities_kernel(std::uint32_t num_bodies, double2 *d_forces, double *d_weights, double2 *d_velocities)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_bodies)
        return;
    d_velocities[i].x = d_velocities[i].x + d_forces[i].x / d_weights[i] * epoch_in_seconds;
    d_velocities[i].y = d_velocities[i].y + d_forces[i].y / d_weights[i] * epoch_in_seconds;
}

void NaiveCudaSimulation::calculate_velocities(Universe &universe, void *d_forces, void *d_weights, void *d_velocities)
{
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;

    if (universe.num_bodies % block_dim == 0)
    {
        grid_dim = universe.num_bodies / block_dim;
    }
    else
    {
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }
    calculate_velocities_kernel<<<grid_dim, block_dim>>>(universe.num_bodies, (double2 *)d_forces, (double *)d_weights, (double2 *)d_velocities);
    cudaDeviceSynchronize();
}

__global__ void calculate_positions_kernel(std::uint32_t num_bodies, double2 *d_velocities, double2 *d_positions)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_bodies)
        return;

    // update position
    d_positions[i].x = d_positions[i].x + d_velocities[i].x * epoch_in_seconds;
    d_positions[i].y = d_positions[i].y + d_velocities[i].y * epoch_in_seconds;
}

void NaiveCudaSimulation::calculate_positions(Universe &universe, void *d_velocities, void *d_positions)
{
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;

    if (universe.num_bodies % block_dim == 0)
    {
        grid_dim = universe.num_bodies / block_dim;
    }
    else
    {
        grid_dim = (universe.num_bodies - (universe.num_bodies % block_dim) + block_dim) / block_dim;
    }
    calculate_positions_kernel<<<grid_dim, block_dim>>>(universe.num_bodies, (double2 *)d_velocities, (double2 *)d_positions);
    cudaDeviceSynchronize();
}

void NaiveCudaSimulation::simulate_epochs(Plotter &plotter, Universe &universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs)
{
    void *d_weights;
    void *d_forces;
    void *d_velocities;
    void *d_positions;

    allocate_device_memory(universe, &d_weights, &d_forces, &d_velocities, &d_positions);
    copy_data_to_device(universe, d_weights, d_forces, d_velocities, d_positions);
    for (int i = 0; i < num_epochs; i++)
    {
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs, d_weights, d_forces, d_velocities, d_positions);
    }
    copy_data_from_device(universe, d_weights, d_forces, d_velocities, d_positions);
    free_device_memory(&d_weights, &d_forces, &d_velocities, &d_positions);
}

__global__ void get_pixels_kernel(std::uint32_t num_bodies, double2 *d_positions, std::uint8_t *d_pixels, std::uint32_t plot_width, std::uint32_t plot_height, double plot_bounding_box_x_min, double plot_bounding_box_x_max, double plot_bounding_box_y_min, double plot_bounding_box_y_max)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_bodies)
        return;
    auto pos = d_positions[i];
    if (pos.x > plot_bounding_box_x_max || pos.x < plot_bounding_box_x_min || pos.y > plot_bounding_box_y_max || pos.y < plot_bounding_box_y_min)
    {
        return;
    }
    std::size_t pixel_x = ((pos.x - plot_bounding_box_x_min) / (plot_bounding_box_x_max - plot_bounding_box_x_min)) * (plot_width - 1);
    std::size_t pixel_y = ((pos.y - plot_bounding_box_y_min) / (plot_bounding_box_y_max - plot_bounding_box_y_min)) * (plot_height - 1);
    d_pixels[pixel_x + pixel_y * plot_width] = 1;
}

std::vector<std::uint8_t> NaiveCudaSimulation::get_pixels(std::uint32_t plot_width, std::uint32_t plot_height, BoundingBox plot_bounding_box, void *d_positions, std::uint32_t num_bodies)
{
    auto plot_size = plot_width * plot_height;
    void *d_pixels;
    parprog_cudaMalloc(&d_pixels, plot_size * sizeof(std::uint8_t));
    std::vector<std::uint8_t> pixels(plot_size, 0);
    parprog_cudaMemcpy(d_pixels, pixels.data(), plot_size * sizeof(std::uint8_t), cudaMemcpyKind::cudaMemcpyHostToDevice);
    std::uint32_t block_dim = 512;
    std::uint32_t grid_dim;

    if (num_bodies % block_dim == 0)
    {
        grid_dim = num_bodies / block_dim;
    }
    else
    {
        grid_dim = (num_bodies - (num_bodies % block_dim) + block_dim) / block_dim;
    }
    get_pixels_kernel<<<grid_dim, block_dim>>>(num_bodies, (double2 *)d_positions, (std::uint8_t *)d_pixels, plot_width, plot_height, plot_bounding_box.x_min, plot_bounding_box.x_max, plot_bounding_box.y_min, plot_bounding_box.y_max);
    cudaDeviceSynchronize();
    parprog_cudaMemcpy(pixels.data(), d_pixels, plot_size * sizeof(std::uint8_t), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    parprog_cudaFree(d_pixels);
    return pixels;
}
__global__ void compress_pixels_kernel(std::uint32_t num_raw_pixels, std::uint8_t *d_raw_pixels, std::uint8_t *d_compressed_pixels)
{
    std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ std::uint8_t pixels[8];
    pixels[threadIdx.x] = 0;
    __syncthreads();
    if (d_raw_pixels[i] != 0)
    {
        pixels[threadIdx.x] = pixels[threadIdx.x] | (1 << threadIdx.x);
    }
    __syncthreads();
    if(threadIdx.x == 0){
        std::uint8_t pixel = 0;
        for(int i = 0; i < blockDim.x; ++i){
            pixel |= pixels[i];
        }
        d_compressed_pixels[blockIdx.x] = pixel;
    }
}

void NaiveCudaSimulation::compress_pixels(std::vector<std::uint8_t> &raw_pixels, std::vector<std::uint8_t> &compressed_pixels)
{
    auto num_raw_pixels = raw_pixels.size();
    void *d_raw_pixels;
    void *d_compressed_pixels;
    parprog_cudaMalloc(&d_raw_pixels, num_raw_pixels);
    parprog_cudaMalloc(&d_compressed_pixels, num_raw_pixels / 8);
    parprog_cudaMemcpy(d_raw_pixels, raw_pixels.data(), num_raw_pixels, cudaMemcpyKind::cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_compressed_pixels, compressed_pixels.data(), num_raw_pixels / 8, cudaMemcpyKind::cudaMemcpyHostToDevice);
    std::uint32_t block_dim = 8;
    std::uint32_t grid_dim= num_raw_pixels / block_dim;
    compress_pixels_kernel<<<grid_dim, block_dim>>>(num_raw_pixels, (std::uint8_t *)d_raw_pixels, (std::uint8_t *)d_compressed_pixels);
    cudaDeviceSynchronize();

    parprog_cudaMemcpy(compressed_pixels.data(), d_compressed_pixels, num_raw_pixels / 8, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    parprog_cudaFree(d_compressed_pixels);
    parprog_cudaFree(d_raw_pixels);
}

void NaiveCudaSimulation::simulate_epoch(Plotter &plotter, Universe &universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs, void *d_weights, void *d_forces, void *d_velocities, void *d_positions)
{
    calculate_forces(universe, d_positions, d_weights, d_forces);
    calculate_velocities(universe, d_forces, d_weights, d_velocities);
    calculate_positions(universe, d_velocities, d_positions);

    universe.current_simulation_epoch++;
    if (create_intermediate_plots)
    {
        if (universe.current_simulation_epoch % plot_intermediate_epochs == 0)
        {
            std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), plotter.get_plot_bounding_box(), d_positions, universe.num_bodies);
            plotter.add_active_pixels_to_image(pixels);

            // This is a dummy to use compression in plotting, although not beneficial performance-wise
            // ----
            // std::vector<std::uint8_t> compressed_pixels;
            // compressed_pixels.resize(pixels.size()/8);
            // compress_pixels(pixels, compressed_pixels);
            // plotter.add_compressed_pixels_to_image(compressed_pixels);
            // ----

            plotter.write_and_clear();
        }
    }
}

void NaiveCudaSimulation::calculate_forces_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void *d_positions, void *d_weights, void *d_forces)
{
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(num_bodies, (double2 *)d_positions, (double *)d_weights, (double2 *)d_forces);
}

void NaiveCudaSimulation::calculate_velocities_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void *d_forces, void *d_weights, void *d_velocities)
{
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_velocities_kernel<<<gridDim, blockDim>>>(num_bodies, (double2 *)d_forces, (double *)d_weights, (double2 *)d_velocities);
}

void NaiveCudaSimulation::calculate_positions_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void *d_velocities, void *d_positions)
{
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel<<<gridDim, blockDim>>>(num_bodies, (double2 *)d_velocities, (double2 *)d_positions);
}

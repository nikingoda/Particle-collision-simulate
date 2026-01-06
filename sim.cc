#include <omp.h>

#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

#include "collision.h"
#include "io.h"
#include "sim_validator.h"

//Cell to contain particles
struct Cell {
    std::vector<int> particlesId;
};

struct CollideList {
    std::vector<int> particlesId;
};

void updatePositions(std::vector<Particle>& particles) {
    int numOfParticles = particles.size();
    #pragma omp parallel for
    for(int i = 0; i < numOfParticles; i++) {
        particles[i].loc.x += particles[i].vel.x;
        particles[i].loc.y += particles[i].vel.y;
    }
}

void allocateParticlesInCell(std::vector<std::vector<Cell>>& grid, std::vector<Particle>& particles, int cellSize, int numOfCells) {
    int N = particles.size();
    //reset particles allocate in grid
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < numOfCells; i++) {
        for(int j = 0; j < numOfCells; j++) {
            grid[i][j].particlesId.clear();
        }
    }
    //allocate particles into cells
    for(int i = 0; i < N; i++) {
        int xCellId = particles[i].loc.x/cellSize;
        xCellId = std::max(xCellId, 0);
        xCellId = std::min(xCellId, numOfCells - 1);
        int yCellId = particles[i].loc.y/cellSize;
        yCellId = std::max(yCellId, 0);
        yCellId = std::min(yCellId, numOfCells - 1);
        grid[xCellId][yCellId].particlesId.push_back(i);
    }
}

bool getCollideWallList(int N, std::vector<Particle>& particles, std::vector<int>& collideWallList, int squareSize, int radius) {
    bool isCollide = false;
    for(int id = 0; id < N; id++) {
        auto &p = particles[id];
        if(is_wall_overlap(p.loc, squareSize, radius)) {
            isCollide = true;
            collideWallList.push_back(id);
        }
    }
    return isCollide;
}

bool getCollideListInCell(std::vector<std::vector<Cell>>& grid,  std::vector<Particle>& particles, int numOfCells, int radius, int xCellId, int yCellId, std::vector<CollideList>& collideGrid) {
    bool isCollide = false;
    auto &cell = grid[xCellId][yCellId];
    int N = cell.particlesId.size();
    //save overlap particles within a cell

    for(int id1 = 0; id1 < N; id1++) {
        for(int id2 = 0; id2 < N; id2++) {
            auto &collideList = collideGrid[cell.particlesId[id1]].particlesId;
            auto &p1 = particles[cell.particlesId[id1]];
            auto &p2 = particles[cell.particlesId[id2]];
            if(is_particle_overlap(p1.loc, p2.loc, radius)) {
                isCollide = true;
                collideList.push_back(cell.particlesId[id2]);
            }
        }
    }

    //save particles overlap with adjacent cell
    for(int xCellId1 = xCellId - 1; xCellId1 <= xCellId + 1; xCellId1++) {
        for(int yCellId1 = yCellId; yCellId1 <= yCellId + 1; yCellId1++) {
            bool invalid = (xCellId1<0) | (yCellId1<0) | (xCellId1>=numOfCells) | (yCellId1>=numOfCells) | ((yCellId1 == yCellId) & (xCellId1 <= xCellId));
            if(invalid) {
                continue;
            }
            auto &cell1 = grid[xCellId1][yCellId1];
            int N1 = cell1.particlesId.size();
            for(int id1 = 0; id1 < N; id1++) {
                auto &collideList = collideGrid[cell.particlesId[id1]].particlesId;
                for(int id2 = 0; id2 < N1; id2++) {
                    auto &p1 = particles[cell.particlesId[id1]];
                    auto &p2 = particles[cell1.particlesId[id2]];
                    if(is_particle_overlap(p1.loc, p2.loc, radius)) {
                        isCollide = true;
                        collideList.push_back(cell1.particlesId[id2]);
                    }
                }
            }
        }
    }
    return isCollide;
}

bool getCollideList(std::vector<std::vector<Cell>>& grid, std::vector<Particle>& particles, int radius, int xStartId, int yStartId, int numOfCells, std::vector<CollideList>& collideGrid) {
    bool isCollide = false;
    #pragma omp parallel for collapse(2)
    for(int xCellId = xStartId; xCellId < numOfCells; xCellId+=2) {
        for(int yCellId = yStartId; yCellId < numOfCells; yCellId+=2) {
            bool isCollideInCell = getCollideListInCell(grid, particles, numOfCells, radius, xCellId, yCellId, collideGrid);
            #pragma omp atomic
            isCollide |= isCollideInCell;
        }
    }
    return isCollide;
}

bool resolveWallCollide(std::vector<Particle>& particles, int squareSize, std::vector<int>& collideWallList, int radius) {
    bool isCollide = false;
    #pragma omp parallel for
    for(int pId : collideWallList) {
        auto &p = particles[pId];
        if(is_wall_collision(p.loc, p.vel, squareSize, radius)) {
            isCollide = true;
            resolve_wall_collision(p.loc, p.vel, squareSize, radius);
        }
    }
    return isCollide;
}

bool resolveCollisionInCell(std::vector<std::vector<Cell>>& grid, std::vector<Particle>& particles, int radius, int xCellId, int yCellId, std::vector<CollideList>& collideGrid) {
    bool isCollide = false;
    auto &cell = grid[xCellId][yCellId];
    int N = cell.particlesId.size();
    for(int id = 0; id < N; id++) {
        int pId1 = cell.particlesId[id];
        auto &p1 = particles[pId1];
        auto &collideList = collideGrid[pId1].particlesId;
        for(int pId2 : collideList) {
            auto &p2 = particles[pId2];
            if(is_particle_collision(p1.loc, p1.vel, p2.loc, p2.vel, radius)) {
                isCollide = true;
                resolve_particle_collision(p1.loc, p1.vel, p2.loc, p2.vel);
            }
        }
    }
    return isCollide;
}

void resetCollideList(std::vector<CollideList>& collideGrid, std::vector<int>& collideWallList) {
    #pragma omp parallel for
    for(auto &collideList : collideGrid) {
        collideList.particlesId.clear();
    }
    collideWallList.clear();
}

bool resolveCollideInCell(std::vector<std::vector<Cell>>& grid,  std::vector<Particle>& particles, int squareSize, int numOfCells, int radius, int xCellId, int yCellId) {
    bool isCollide = false;
    auto &cell = grid[xCellId][yCellId];
            int N = cell.particlesId.size();

            // #pragma omp parallel for
            for(int id = 0; id < N; id++) {
                auto &p = particles[cell.particlesId[id]];
                //resolve wall collision
                if(is_wall_collision(p.loc, p.vel, squareSize, radius)) {
                    isCollide = true;
                    resolve_wall_collision(p.loc, p.vel, squareSize, radius);
                }
            }

            //refolve collide within this cell
            for(int id1 = 0; id1 < N; id1++) {
                for(int id2 = 0; id2 < N; id2++) {
                    auto &p1 = particles[cell.particlesId[id1]];
                    auto &p2 = particles[cell.particlesId[id2]];
                    if(is_particle_collision(p1.loc, p1.vel, p2.loc, p2.vel, radius)) {
                        isCollide = true;
                        resolve_particle_collision(p1.loc, p1.vel, p2.loc, p2.vel);
                    }
                }
            }

            //resolve collide with adjacent cell
            for(int xCellId1 = xCellId - 1; xCellId1 <= xCellId + 1; xCellId1++) {
                for(int yCellId1 = yCellId; yCellId1 <= yCellId + 1; yCellId1++) {
                    bool invalid = (xCellId1<0) | (yCellId1<0) | (xCellId1>=numOfCells) | (yCellId1>=numOfCells) | ((yCellId1 == yCellId) & (xCellId1 <= xCellId));
                    if(invalid) {
                        continue;
                    }
                    auto &cell1 = grid[xCellId1][yCellId1];
                    int N1 = cell1.particlesId.size();
                    for(int id1 = 0; id1 < N; id1++) {
                        for(int id2 = 0; id2 < N1; id2++) {
                            auto &p1 = particles[cell.particlesId[id1]];
                            auto &p2 = particles[cell1.particlesId[id2]];
                            if(is_particle_collision(p1.loc, p1.vel, p2.loc, p2.vel, radius)) {
                                isCollide = true;
                                resolve_particle_collision(p1.loc, p1.vel, p2.loc, p2.vel);
                            }
                        }
                    }
                }
            }
    return isCollide;
}

bool resolveCollide(std::vector<std::vector<Cell>>& grid, std::vector<Particle>& particles, int squareSize, int radius, int xStartId, int yStartId, int numOfCells) {
    bool isCollide = false;
    #pragma omp parallel for collapse(2)
    for(int xCellId = xStartId; xCellId < numOfCells; xCellId+=2) {
        for(int yCellId = yStartId; yCellId < numOfCells; yCellId+=2) {
            bool isCollideInCell = resolveCollideInCell(grid, particles, squareSize, numOfCells, radius, xCellId, yCellId);
            #pragma omp atomic
            isCollide |= isCollideInCell;
        }
    }
    return isCollide;
}

bool resolveCollision(std::vector<std::vector<Cell>>& grid, std::vector<Particle>& particles, int squareSize, int radius, int xStartId, int yStartId, int numOfCells, std::vector<int>& collideWallList, std::vector<CollideList>& collideGrid) {
    bool isCollide = false;
    isCollide |= resolveWallCollide(particles, squareSize, collideWallList, radius);
    #pragma omp parallel for collapse(2)
    for(int xCellId = xStartId; xCellId < numOfCells; xCellId+=2) {
        for(int yCellId = yStartId; yCellId < numOfCells; yCellId+=2) {
            bool isCollideInCell = resolveCollisionInCell(grid, particles, radius, xCellId, yCellId, collideGrid);
            #pragma omp atomic
            isCollide |= isCollideInCell;
        }
    }
    return isCollide;
}

int main(int argc, char* argv[]) {
    // Read arguments and input file
    Params params{};
    std::vector<Particle> particles;
    read_args(argc, argv, params, particles);

    // Set number of threads
    omp_set_num_threads(params.param_threads);

#if CHECK == 1
    // Initialize collision checker
    SimulationValidator validator(params.param_particles, params.square_size, params.param_radius, params.param_steps);
    // Initialize with starting positions
    validator.initialize(particles);
    // Uncomment the line below to enable visualization (makes program much slower)
    // validator.enable_viz_output("test.out");
#endif

    // TODO: this is the part where you simulate particle behavior.
    int N = params.param_particles;
    int L = params.square_size;
    int r = params.param_radius;
    int S = params.param_steps;
    int cellSize = 4*r;
    int numOfCells = static_cast<int>(std::ceil(L/cellSize));
    std::vector<CollideList> collideGrid(N);
    std::vector<int> collideWallList;
    //initialize grid
    std::vector<std::vector<Cell>> grid(numOfCells, std::vector<Cell>(numOfCells));
    for(int i = 0; i < S; i++) { 
        updatePositions(particles);
        allocateParticlesInCell(grid, particles, cellSize, numOfCells);
        resetCollideList(collideGrid, collideWallList);
        // bool isCollide = true;
        // //divide grid into 4 regions to process
        // while(isCollide) {
        //     isCollide = resolveCollide(grid, particles, L, r, 0, 0, numOfCells) |
        //     resolveCollide(grid, particles, L, r, 0, 1, numOfCells) |
        //     resolveCollide(grid, particles, L, r, 1, 0, numOfCells) |
        //     resolveCollide(grid, particles, L, r, 1, 1, numOfCells);
        // }

        bool isCollide = getCollideWallList(N, particles, collideWallList, L, r) |
        getCollideList(grid, particles, r, 0, 0, numOfCells, collideGrid) |
        getCollideList(grid, particles, r, 0, 1, numOfCells, collideGrid) |
        getCollideList(grid, particles, r, 1, 0, numOfCells, collideGrid) |
        getCollideList(grid, particles, r, 1, 1, numOfCells, collideGrid);
        while(isCollide) {
            isCollide = resolveCollision(grid, particles, L, r, 0, 0, numOfCells, collideWallList, collideGrid) |
            resolveCollision(grid, particles, L, r, 0, 1, numOfCells, collideWallList, collideGrid) |
            resolveCollision(grid, particles, L, r, 1, 0, numOfCells, collideWallList, collideGrid) |
            resolveCollision(grid, particles, L, r, 1, 1, numOfCells, collideWallList, collideGrid);
        }
        #if CHECK == 1
            validator.validate_step(particles);
        #endif
    }

    /*
    After simulating each timestep, you must call this exact block below.
    Make sure that your final submission has both the validation logic above and below included, within the #if

    #if CHECK == 1
        validator.validate_step(particles);
    #endif
    */
   std::printf("successfully");
   return 0;
}

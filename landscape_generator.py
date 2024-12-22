# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import copy
from collections import deque

# Constants
ROWS = 40
COLS = 50
TILE_SIZE = 64  # pixels
TILES_DIR = '/content/drive/My Drive/fish/Landscape/data'
POPULATION_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.05
ELITISM_COUNT = 5

# Define tile types and assign unique numbers
TILE_TYPES = {
    'desert': 0,
    'empty': 1,
    'forest': 2,
    'grass': 3,
    'lake': 4,
    'mountain': 5,
    'path': 6,
    'river': 7,
    'riverstone': 8,
    'rock': 9,
    'tree': 10
}

TILE_PROPORTIONS = {
    'desert': 3,
    'empty': 3,
    'forest': 6,    # Each 2x2 cluster represents 4 forest tiles
    'grass': 20,
    'lake': 3,
    'mountain': 6,
    'path': 3,
    'river': 40,
    'riverstone': 3,
    'rock': 2,
    'tree': 10      # Increased to surround forest clusters
}

# Load tile images
def load_tiles(tiles_dir, tile_types):
    tiles = {}
    for tile_name, tile_num in tile_types.items():
        path = os.path.join(tiles_dir, f"{tile_name}.png")
        if os.path.exists(path):
            tiles[tile_num] = Image.open(path).resize((TILE_SIZE, TILE_SIZE))
        else:
            print(f"Tile image not found: {path}")
            # Create a placeholder image if not found
            tiles[tile_num] = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (255, 0, 0, 255))
    return tiles

tiles = load_tiles(TILES_DIR, TILE_TYPES)

def create_dynamic_river(rows, cols, target_river_tiles):
    river_map = np.full((rows, cols), TILE_TYPES['grass'], dtype=int)
    x, y = random.randint(0, rows // 3), random.randint(0, cols // 3)  # Random start position
    river_tiles_placed = 0
    river_map[x, y] = TILE_TYPES['river']
    river_tiles_placed += 1
    
    while river_tiles_placed < target_river_tiles:
        move_options = []
        # Allow movement in all directions but prefer flowing down or sideways
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (-1, -1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and river_map[nx, ny] != TILE_TYPES['river']:
                move_options.append((nx, ny))
        
        if not move_options:  # Stop if there are no valid moves
            break
        
        x, y = random.choice(move_options)
        river_map[x, y] = TILE_TYPES['river']
        river_tiles_placed += 1
        
        if random.random() < 0.3 and river_tiles_placed < target_river_tiles:
            for dx, dy in random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], k=2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and river_map[nx, ny] != TILE_TYPES['river']:
                    river_map[nx, ny] = TILE_TYPES['river']
                    river_tiles_placed += 1
    return river_map

def adjust_grass_near_river(map_grid, rows, cols, max_distance=3):
    grass_num = TILE_TYPES['grass']
    stone_num = TILE_TYPES['rock']  # Replace grass with stones if too far
    river_num = TILE_TYPES['river']
    empty_num = TILE_TYPES['empty'] 
    for x in range(rows):
        for y in range(cols):
            if map_grid[x, y] == grass_num:
                is_near_river = False
                for dx in range(-max_distance, max_distance + 1):
                    for dy in range(-max_distance, max_distance + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and map_grid[nx, ny] == river_num:
                            is_near_river = True
                            break
                    if is_near_river:
                        break
                
                # Convert grass to stone or empty land if too far from river
                if not is_near_river:
                    map_grid[x, y] = random.choice([stone_num, empty_num])
    return map_grid


def initialize_population(pop_size, rows, cols, tile_proportions):
    population = []
    total = rows * cols
    tile_counts = {TILE_TYPES[k]: int(v/100 * total) for k, v in tile_proportions.items()}
    
    # Adjust to ensure total tiles match
    current_total = sum(tile_counts.values())
    while current_total < total:
        for tile, count in tile_counts.items():
            if current_total < total:
                tile_counts[tile] += 1
                current_total += 1
            else:
                break
    while current_total > total:
        for tile, count in tile_counts.items():
            if current_total > total and count > 0:
                tile_counts[tile] -= 1
                current_total -= 1
            else:
                break
    
    # Calculate number of forest clusters (each cluster is 4 forest tiles)
    forest_tiles = tile_counts[TILE_TYPES['forest']]
    if forest_tiles % 4 != 0:
        # Adjust to make it divisible by 4
        forest_tiles = (forest_tiles // 4) * 4
        tile_counts[TILE_TYPES['forest']] = forest_tiles
    num_forest_clusters = forest_tiles // 4
    
    target_river_tiles = tile_counts[TILE_TYPES['river']]
    river_template = create_dynamic_river(rows, cols, target_river_tiles)
    
    # Function to place forest clusters
    def place_forest_clusters(map_grid, num_clusters, tile_counts, rows, cols):
        available_positions = [(x, y) for x in range(rows-1) for y in range(cols-1)]
        random.shuffle(available_positions)
        clusters_placed = 0
        for pos in available_positions:
            x, y = pos
            # Check if 2x2 area is available (not already river or forest)
            area = map_grid[x:x+2, y:y+2]
            if np.all(area == TILE_TYPES['grass']) and clusters_placed < num_clusters:
                map_grid[x:x+2, y:y+2] = TILE_TYPES['forest']
                clusters_placed += 1
                # Surround the cluster with trees
                for dx in range(-1, 3):
                    for dy in range(-1, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if map_grid[nx, ny] == TILE_TYPES['grass']:
                                map_grid[nx, ny] = TILE_TYPES['tree']
                # Decrement tree count based on trees added
                # Calculate number of trees added
                trees_added = 0
                for dx in range(-1, 3):
                    for dy in range(-1, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if map_grid[nx, ny] == TILE_TYPES['tree']:
                                trees_added +=1
                tile_counts[TILE_TYPES['tree']] -= trees_added
                if tile_counts[TILE_TYPES['tree']] < 0:
                    tile_counts[TILE_TYPES['tree']] = 0
        return map_grid
    
    # Initialize population maps by placing the river and forest clusters, then filling the rest randomly
    for _ in range(pop_size):
        map_grid = river_template.copy()
        
        # Place forest clusters
        map_grid = place_forest_clusters(map_grid, num_forest_clusters, tile_counts, rows, cols)
        map_grid = adjust_grass_near_river(map_grid, rows, cols, max_distance=3)
        # Calculate remaining tiles
        remaining_tiles = []
        for tile, count in tile_counts.items():
            if tile in [TILE_TYPES['river'], TILE_TYPES['forest'], TILE_TYPES['tree']]:
                continue  # River, Forest, and Tree already placed
            remaining_tiles += [tile] * count
        # Remove tiles already placed as river and forest
        river_tile_count = np.sum(river_template == TILE_TYPES['river'])
        forest_tile_count = np.sum(map_grid == TILE_TYPES['forest'])
        remaining_tiles = remaining_tiles[:rows * cols - river_tile_count - forest_tile_count]
        random.shuffle(remaining_tiles)
        # Fill in the remaining tiles
        it = iter(remaining_tiles)
        for x in range(rows):
            for y in range(cols):
                if map_grid[x, y] in [TILE_TYPES['river'], TILE_TYPES['forest'], TILE_TYPES['tree']]:
                    continue
                map_grid[x, y] = next(it, TILE_TYPES['grass'])  # Default to grass if out of tiles
        population.append(map_grid)
    return population

# Fitness function
def calculate_fitness(map_grid):
    fitness = 0
    rows, cols = map_grid.shape
    total = rows * cols
    
    # 1. Tile proportions
    unique, counts = np.unique(map_grid, return_counts=True)
    tile_count = dict(zip(unique, counts))
    for tile, desired_percent in TILE_PROPORTIONS.items():
        tile_num = TILE_TYPES[tile]
        desired_count = desired_percent / 100 * total
        actual_count = tile_count.get(tile_num, 0)
        proportion_error = abs(desired_count - actual_count) / desired_count
        # Fitness is better when proportion error is smaller
        fitness += max(0, 1 - proportion_error)  # Reward between 0 and 1 for each tile type
    
    # 2. River connectivity and prominence
    river_num = TILE_TYPES['river']
    riverstone_num = TILE_TYPES['riverstone']
    river_cells = set(zip(*np.where(map_grid == river_num)))
    actual_river_tiles = len(river_cells)
    desired_river_tiles = TILE_PROPORTIONS['river'] / 100 * total
    river_coverage = actual_river_tiles / desired_river_tiles
    # Reward for coverage up to desired
    fitness += min(river_coverage, 1) * 50  # Up to 50 points for river coverage
    
    if not river_cells:
        fitness -= 100  # Large penalty if no river
    else:
        # Check if river connects top-left to bottom-right
        start = (0,0)
        end = (rows-1, cols-1)
        if map_grid[start] != river_num or map_grid[end] != river_num:
            fitness -= 50  # Penalty if river doesn't start/end correctly
        else:
            # BFS to check connectivity
            visited = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                current = queue.popleft()
                if current == end:
                    break
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nx, ny = current[0] + dx, current[1] + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if map_grid[nx, ny] == river_num and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            if end not in visited:
                fitness -= 50  # Penalty if river is not fully connected
            else:
                fitness += 100  # Reward for successful river connectivity
    
    # 3. Riverstone inside river
    riverstone_count = tile_count.get(riverstone_num, 0)
    if riverstone_count > 0:
        riverstone_positions = np.where(map_grid == riverstone_num)
        valid_riverstones = 0
        for x, y in zip(*riverstone_positions):
            if map_grid[x, y] == river_num:
                valid_riverstones +=1
            else:
                fitness -= 5  # Penalize if riverstone not on river
        fitness += valid_riverstones * 5  # Reward for correct riverstones
    else:
        fitness -= 20  # Penalty if no riverstone
    
    # 4. Mountains on river
    mountain_num = TILE_TYPES['mountain']
    mountains = set(zip(*np.where(map_grid == mountain_num)))
    adjacent_mountains = 0
    for x, y in mountains:
        # Check adjacent cells for river
        adjacent = False
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_grid[nx, ny] == river_num:
                    adjacent = True
                    break
        if adjacent:
            adjacent_mountains +=1
        else:
            fitness -= 2  # Penalize mountains not adjacent to river
    fitness += adjacent_mountains * 2  # Reward for mountains adjacent to river
    
    # 5. Grass surrounding river
    grass_num = TILE_TYPES['grass']
    for x, y in river_cells:
        # Check adjacent cells
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_grid[nx, ny] not in [grass_num, river_num, TILE_TYPES['riverstone']]:
                    fitness -= 1  # Penalize if river not surrounded by grass, riverstone, or river
                elif map_grid[nx, ny] == grass_num:
                    fitness += 0.5  # Slight reward for proper grass placement
    
    # 6. Lakes distribution (small lakes)
    lake_num = TILE_TYPES['lake']
    lake_cells = set(zip(*np.where(map_grid == lake_num)))
    lakes = []
    visited_lakes = set()
    for cell in lake_cells:
        if cell in visited_lakes:
            continue
        # BFS to find connected lake cells
        queue = deque([cell])
        current_lake = set()
        while queue:
            current = queue.popleft()
            if current in visited_lakes:
                continue
            visited_lakes.add(current)
            current_lake.add(current)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols and map_grid[nx, ny] == lake_num:
                    queue.append((nx, ny))
        lakes.append(current_lake)
    
    for lake in lakes:
        lake_size = len(lake)
        if lake_size > 5:
            fitness -= (lake_size - 5) * 2  # Penalize large lakes
        else:
            fitness += lake_size  # Reward small lakes
    
    # 7. Trees and grass mixing
    tree_num = TILE_TYPES['tree']
    trees = set(zip(*np.where(map_grid == tree_num)))
    for x, y in trees:
        adjacent_grass = 0
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_grid[nx, ny] == grass_num:
                    adjacent_grass +=1
        fitness += adjacent_grass * 1  # Reward trees near grass
    
    # 8. Forest Clustering and Surrounding Trees
    forest_num = TILE_TYPES['forest']
    forest_cells = set(zip(*np.where(map_grid == forest_num)))
    
    # Function to check 2x2 forest clusters
    def check_forest_clusters(map_grid, forest_num, rows, cols):
        clusters = 0
        for x in range(rows-1):
            for y in range(cols-1):
                if (map_grid[x, y] == forest_num and
                    map_grid[x+1, y] == forest_num and
                    map_grid[x, y+1] == forest_num and
                    map_grid[x+1, y+1] == forest_num):
                    clusters +=1
        return clusters
    
    actual_forest_clusters = check_forest_clusters(map_grid, forest_num, rows, cols)
    desired_forest_clusters = TILE_PROPORTIONS['forest'] / 100 * total / 4  # Each cluster is 4 tiles
    cluster_error = abs(desired_forest_clusters - actual_forest_clusters)
    fitness -= cluster_error * 5  # Penalize deviation from desired number of clusters
    fitness += actual_forest_clusters * 10  # Reward for each proper cluster
    
    # Ensure forests are surrounded by trees
    def check_forest_surrounding_trees(map_grid, forest_num, tree_num, rows, cols):
        penalties = 0
        for x, y in forest_cells:
            # Check adjacent cells for trees
            has_tree = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if map_grid[nx, ny] == tree_num:
                            has_tree = True
            if not has_tree:
                penalties +=1  # Penalize forests not surrounded by trees
        return penalties
    
    penalties = check_forest_surrounding_trees(map_grid, forest_num, TILE_TYPES['tree'], rows, cols)
    fitness -= penalties * 2  # Penalize each un-surrounded forest tile
    
    # 9. Avoid clustering for non-forest tiles
    # Reward for larger contiguous regions and penalize excessive scattering
    def get_contiguous_regions(map_grid, tile_num, rows, cols):
        visited = set()
        regions = []
        for x in range(rows):
            for y in range(cols):
                if map_grid[x, y] == tile_num and (x, y) not in visited:
                    queue = deque([(x, y)])
                    region = set()
                    while queue:
                        current = queue.popleft()
                        if current in visited:
                            continue
                        visited.add(current)
                        region.add(current)
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nx, ny = current[0] + dx, current[1] + dy
                            if 0 <= nx < rows and 0 <= ny < cols:
                                if map_grid[nx, ny] == tile_num and (nx, ny) not in visited:
                                    queue.append((nx, ny))
                    regions.append(region)
        return regions
    
    for tile, tile_num in TILE_TYPES.items():
        if tile in ['river', 'forest', 'tree']:
            continue  # Already handled
        regions = get_contiguous_regions(map_grid, tile_num, rows, cols)
        for region in regions:
            if len(region) >= 5:
                fitness += len(region) * 0.1  # Reward for larger regions
            elif len(region) < 2:
                fitness -= len(region) * 2  # Penalize tiny scattered tiles
    
    return fitness

# Selection: Tournament Selection
def tournament_selection(population, fitnesses, k=3):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(list(zip(population, fitnesses)), k)
        selected.append(max(participants, key=lambda x: x[1])[0])
    return selected

# Crossover: Single-point crossover with preservation of river
def crossover(parent1, parent2):
    rows, cols = parent1.shape
    crossover_point = random.randint(1, rows * cols -1)
    parent1_flat = parent1.flatten()
    parent2_flat = parent2.flatten()
    child1_flat = np.concatenate([parent1_flat[:crossover_point], parent2_flat[crossover_point:]])
    child2_flat = np.concatenate([parent2_flat[:crossover_point], parent1_flat[crossover_point:]])
    child1 = child1_flat.reshape((rows, cols))
    child2 = child2_flat.reshape((rows, cols))
    
    # Ensure river is preserved from parent1
    river_template = np.where(parent1 == TILE_TYPES['river'], TILE_TYPES['river'], child1)
    child1 = river_template
    
    # Ensure river is preserved from parent2
    river_template = np.where(parent2 == TILE_TYPES['river'], TILE_TYPES['river'], child2)
    child2 = river_template
    
    return child1, child2

# Mutation: Swap two tiles, preserving river and forest clusters
def mutate(map_grid, mutation_rate):
    rows, cols = map_grid.shape
    for x in range(rows):
        for y in range(cols):
            if map_grid[x, y] in [TILE_TYPES['river'], TILE_TYPES['forest'], TILE_TYPES['tree']]:
                continue  # Do not mutate river, forest, or tree tiles
            if random.random() < mutation_rate:
                # Swap with a random non-river, non-forest, non-tree position
                attempts = 0
                while attempts < 10:
                    nx, ny = random.randint(0, rows-1), random.randint(0, cols-1)
                    if map_grid[nx, ny] not in [TILE_TYPES['river'], TILE_TYPES['forest'], TILE_TYPES['tree']]:
                        break
                    attempts += 1
                map_grid[x, y], map_grid[nx, ny] = map_grid[nx, ny], map_grid[x, y]
    return map_grid

# Evolutionary Algorithm
def evolve_population(population, generations, mutation_rate):
    for gen in range(generations):
        fitnesses = [calculate_fitness(individual) for individual in population]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {gen+1}, Best Fitness: {best_fitness:.2f}, Avg Fitness: {avg_fitness:.2f}")
        
        # Elitism: Keep the top ELITISM_COUNT individuals
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        new_population = sorted_population[:ELITISM_COUNT]
        
        # Selection
        selected = tournament_selection(population, fitnesses)
        
        # Crossover
        children = []
        for i in range(0, len(selected)-1, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1, child2 = crossover(parent1, parent2)
            children.extend([child1, child2])
        
        # Mutation
        children = [mutate(child, mutation_rate) for child in children]
        
        # Fill the rest of the population
        new_population += children[:POPULATION_SIZE - ELITISM_COUNT]
        population = new_population
        
        # Optional: Early stopping if best fitness reaches a threshold
        # if best_fitness > 500:  # Example threshold
        #     print("Early stopping as fitness threshold is met.")
        #     break
    
    # Final fitness evaluation
    fitnesses = [calculate_fitness(individual) for individual in population]
    best_index = np.argmax(fitnesses)
    best_map = population[best_index]
    return best_map

# Assemble map into image
def assemble_map(map_grid, tiles):
    rows, cols = map_grid.shape
    map_image = Image.new('RGBA', (cols * TILE_SIZE, rows * TILE_SIZE))
    for x in range(rows):
        for y in range(cols):
            tile_num = map_grid[x, y]
            tile_img = tiles.get(tile_num, Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (255, 0, 0, 255)))
            map_image.paste(tile_img, (y * TILE_SIZE, x * TILE_SIZE))
    return map_image

# Main execution
if __name__ == "__main__":
    # Directory to save output maps
    output_dir = '/content/drive/My Drive/fish/Landscape/output'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):  # Generate 10 different maps
        # Introduce small random variation in tile proportions for diversity
        varied_proportions = {
            tile: max(1, min(100, TILE_PROPORTIONS[tile] + random.uniform(-0.5, 0.5)))
            for tile in TILE_PROPORTIONS
        }
        
        # Reinitialize population with varied proportions
        population = initialize_population(POPULATION_SIZE, ROWS, COLS, varied_proportions)
        
        # Generate a dynamic river for each map
        target_river_tiles = int(varied_proportions['river'] / 100 * ROWS * COLS)
        river_template = create_dynamic_river(ROWS, COLS, target_river_tiles)
        
        # Evolve population
        best_map = evolve_population(population, GENERATIONS, MUTATION_RATE)
        
        # Assemble the map into an image
        final_map_image = assemble_map(best_map, tiles)
        
        # Save the map
        save_path = os.path.join(output_dir, f'generated_map_{i+1}.png')
        final_map_image.save(save_path)
        print(f"Map {i+1} saved to {save_path}")
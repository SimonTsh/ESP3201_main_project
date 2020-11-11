from cv2 import floodFill
import numpy as np
from ImageTo3D import castTo3D

ROAD = 1
NO_IMG = -1
OBSTACLE = 0

def makeGrid(xs, zs, labels, d=1.0):
    # grid is nu by nv and is centered at u = nu/2, v = 0
    if xs.size < 100:
        print("makeGrid - WARNING: no image points detected")
        return np.zeros((10,10))

    xmax = np.max(xs); xmin = np.min(xs)
    zmax = np.max(zs); zmin = np.min(zs)
    nu = int((xmax - xmin) // d) + 1
    nv = int((zmax - zmin) // d) + 1
    # print("grid shape: ", nu, nv)
    # if (nu < 10) or (nv < 10):
    #     return np.zeros((10,10))
    grid = NO_IMG * np.ones((nu, nv), dtype=np.int8)    # fill with NO_IMG

    for i in range(len(xs)):                # iterate through each point
        u = int((xs[i] - xmin) // d)
        v = int((zs[i] - zmin) // d)

        # HOTFIX: discard points outside the allowed range
        # if u >= nu or u < 0 or v >= nv or v < 0:
        #     continue
        if labels[i] == 'CAR':
            grid[u,v] = OBSTACLE
        elif grid[u,v] == ROAD:
            continue
        elif labels[i] == 'ROAD':       # or (pos[3] == 'TRAF_SIGN'):
            grid[u,v] = ROAD            # can accumulate totals, but need to use a different 'invalid' flag (currently -1)
        else:
            grid[u,v] = OBSTACLE        # 1 for road cells, 0 for non-road cells

    # perform flood-fill and get the main road's mask
    # TODO: can try flood-filling no-img CCs that are surrounded by main road CC
    # idea: 1 round of averaging
    mask = np.array(grid, copy=True)
    mask[mask < 0] = 0
    mask = mask.astype('uint8')
    MASK_VALUE = 15

    TRIAL_DIST = 3
    for i in range(-TRIAL_DIST, TRIAL_DIST):
        for j in range(TRIAL_DIST):
            if mask[nu // 2 + i,j] == 0 or mask[nu // 2 + i,j] == MASK_VALUE:
                continue
            else:
                floodFill(mask, None, (j, nu // 2 + i), MASK_VALUE)
    mask = (mask == MASK_VALUE)
    grid[grid == ROAD] = mask[grid == ROAD]
    return grid

def drawGrid(grid):
    nu, nv = grid.shape
    us = np.arange(nu); vs = np.arange(nv)
    uu,vv = np.meshgrid(vs,us)
    c = plt.pcolormesh(uu, vv, grid, cmap ='Greens', vmin = np.min(grid), vmax = np.max(grid))
    plt.colorbar(c)
    plt.title('Discrete map', fontweight ="bold")
    plt.show()

def computeWalls(grid):
    # TODO: walls near no-img gaps are not added
    walls = np.zeros(grid.shape,dtype=np.uint8)

    count_nb_roads = np.zeros(grid.shape, dtype=np.uint8)
    count_nb_roads[:-1,:] += (grid[1:,:] == ROAD)
    count_nb_roads[1:,:] += (grid[:-1,:] == ROAD)
    count_nb_roads[:,:-1] += (grid[:,1:] == ROAD)
    count_nb_roads[:,1:] += (grid[:,:-1] == ROAD)
    # if road cell has < 4 adj road cells, it's on the perimeter. put up a wall.
    walls[(grid == ROAD) & (count_nb_roads < 4)] = 1

    count_noimg = np.zeros(grid.shape, dtype=np.uint8)
    count_noimg[:-1,:] += (grid[1:,:] == NO_IMG)
    count_noimg[1:,:] += (grid[:-1,:] == NO_IMG)
    count_noimg[:,:-1] += (grid[:,1:] == NO_IMG)
    count_noimg[:,1:] += (grid[:,:-1] == NO_IMG)
    count_noimg[-1,:] += 1
    count_noimg[:,-1] += 1

    # if the road cell is at the image border, there may not actually be an obstacle there.
    # so don't put a wall.
    walls[(grid == ROAD) & (count_noimg > 0)] = 0
    return walls

def computeGoal(grid):
    nu, nv = grid.shape
    us = np.arange(nu) - nu // 2
    vs = np.arange(nv)
    uu, vv = np.meshgrid(vs,us)
    dists = np.zeros(grid.shape, dtype=np.uint16)
    tmp = uu**2 + vv**2
    dists[grid == ROAD] = tmp[grid == ROAD]
    return np.unravel_index(dists.argmax(), dists.shape)

def getPotentialField(grid, walls, goal):
    # TODO: can set no-img cells around the road CC to be 'gradient-free' - take same value as adj valid cells
    NUM_ITER = 100
    U_WALL = 1.0
    U_GOAL = -1.0
    CONSTANT_WALL = False
    nu, nv = grid.shape
    us = np.abs(np.arange(nu) - nu // 2)
    vs = np.arange(nv)
    us, vs = np.meshgrid(vs,us)
    u_walls = 5 - 0.05*(us + vs)
    u_grid = np.zeros(grid.shape) # grid.copy().astype('float32')
    u_grid[walls == 1] = U_WALL if CONSTANT_WALL else u_walls[walls==1]
    u_grid[goal] = U_GOAL
    u_grid_dt = np.zeros(u_grid.shape)
    alpha = 0.99
    # print("solving laplacian:")
    for i in range(NUM_ITER):
        del_u = np.zeros(u_grid.shape)
        del_u[1:,:] += u_grid[:-1,:]
        del_u[:-1,:] += u_grid[1:,:]
        del_u[:,1:] += u_grid[:,:-1]
        del_u[:,:-1] += u_grid[:,1:]
        del_u = 0.25 * del_u
        u_grid = alpha * del_u + (1.0 - alpha) * u_grid
        u_grid = (grid > 0) * u_grid
        u_grid[walls == 1] = U_WALL if CONSTANT_WALL else u_walls[walls==1]
        u_grid[goal] = U_GOAL
        if np.max(u_grid) > 100:
            print("WARNING: laplacian solver diverging")
            break
    # print("laplacian solved")

    u_grid[walls == 1] = U_WALL if CONSTANT_WALL else u_walls[walls==1]
    u_grid[goal] = U_GOAL

    # z_eval = 4       # trick? evaluate gradient ~5-10m ahead to avoid upslope at very start of map frustrum
    z_eval = 2       # trick? evaluate gradient ~5-10m ahead to avoid upslope at very start of map frustrum
    dx = -1 * (u_grid[nu // 2 + 1, z_eval] - u_grid[nu // 2 - 1, z_eval])
    dz = u_grid[nu // 2, 0 ] - u_grid[nu // 2, int(max(2, 0))]
    # dz = -0.5 * (u_grid[nu // 2, z_eval + 1] - u_grid[nu // 2, max(z_eval - 1, 0)])

    # if dz < 0:
    #     dx = -dx
    #     dz = -dz
    heading = -np.arctan2(dx, dz)
    return heading, u_grid

def getHeadingInput(img, pitch=0.0, roll=0.0):
    # print("casting to 3d:")
    # xs, zs, labels = castTo3D(img, flat_approx=True)
    if np.abs(pitch) <= np.spacing(1) and np.abs(roll) <= np.spacing(1):
        xs, zs, labels = castTo3D(img, flat_approx=False, pitch=pitch, roll=roll)
    else:
        xs, zs, labels = castTo3D(img, flat_approx=True)
    # print("making grid:")
    grid = makeGrid(xs, zs, labels, d=1.5)
    walls = computeWalls(grid)
    goal = computeGoal(grid)
    return getPotentialField(grid, walls, goal)

import pygame
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH = 800
HEIGHT = 800
SIZE = 800
THETA = 0.7
G = 1.0
dt = 0.5

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(BLACK)


class Body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.force = np.zeros(2, dtype=float)

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, self.pos, np.min([self.mass / 5, 5]))


class QuadTreeNode:
    def __init__(self, centre, size, show):
        self.centre = np.array(centre, dtype=float)
        self.size = size
        self.mass = None
        self.centreOfMass = None
        self.children = None
        self.body = None
        self.show = show

    def draw(self, screen):
        pygame.draw.rect(
            screen,
            WHITE,
            pygame.Rect(
                self.centre[0] - self.size / 2,
                self.centre[1] - self.size / 2,
                self.size,
                self.size,
            ),
            1,
        )


def insertBody(body, root):
    if root.body is None:
        root.mass = body.mass
        root.centreOfMass = np.copy(body.pos)
        root.body = body

    elif root.children is None:
        # Add a minimum size to not max out recursion depth
        if root.size <= 1e-20:
            return

        centreX = root.centre[0]
        centreY = root.centre[1]
        root.children = [
            QuadTreeNode(
                (centreX - root.size / 4, centreY - root.size / 4),
                root.size / 2,
                root.show,
            ),
            QuadTreeNode(
                (centreX + root.size / 4, centreY - root.size / 4),
                root.size / 2,
                root.show,
            ),
            QuadTreeNode(
                (centreX - root.size / 4, centreY + root.size / 4),
                root.size / 2,
                root.show,
            ),
            QuadTreeNode(
                (centreX + root.size / 4, centreY + root.size / 4),
                root.size / 2,
                root.show,
            ),
        ]
        if root.show:
            for child in root.children:
                child.draw(screen)

        existingNodeQuadIndex = findQuadIndex(root.centre, root.body)
        insertBody(
            root.body,
            root.children[existingNodeQuadIndex],
        )
        root.centreOfMass[0] = (
            root.centreOfMass[0] * root.mass + body.pos[0] * body.mass
        ) / (root.mass + body.mass)
        root.centreOfMass[1] = (
            root.centreOfMass[1] * root.mass + body.pos[1] * body.mass
        ) / (root.mass + body.mass)
        root.mass += body.mass
        newNodeQuadIndex = findQuadIndex(root.centre, body)
        insertBody(body, root.children[newNodeQuadIndex])
    else:
        root.centreOfMass[0] = (
            root.centreOfMass[0] * root.mass + body.pos[0] * body.mass
        ) / (root.mass + body.mass)
        root.centreOfMass[1] = (
            root.centreOfMass[1] * root.mass + body.pos[1] * body.mass
        ) / (root.mass + body.mass)
        root.mass += body.mass
        newNodeQuadIndex = findQuadIndex(root.centre, body)
        insertBody(body, root.children[newNodeQuadIndex])


def findQuadIndex(centre, node):
    """
    Indexing of Quadrants:
    0 | 1
    --+--
    2 | 3
    """
    index = 0
    if node.pos[0] > centre[0]:
        index += 1
    if node.pos[1] > centre[1]:
        index += 2
    return index


def calculateForce(body, root):
    if root.centreOfMass is not None:
        deltaDist = root.centreOfMass - body.pos
        r = np.sqrt(deltaDist[0] ** 2 + deltaDist[1] ** 2)
        if r == 0:
            return

        if root.size / r < THETA:
            force = (
                G * body.mass * root.mass * deltaDist[0] / r**3,
                G * body.mass * root.mass * deltaDist[1] / r**3,
            )
            body.force += force
        else:
            if root.children is not None:
                for child in root.children:
                    if child is not None:
                        calculateForce(body, child)


def barnesHut(bodies, toggleQuads):
    root = QuadTreeNode((SIZE / 2, SIZE / 2), SIZE, toggleQuads)

    for body in bodies:
        insertBody(body, root)

    for body in bodies:
        calculateForce(body, root)

    for body in bodies:
        body.vel += body.force / body.mass * dt
        body.pos += body.vel * dt
        body.force = np.zeros(2, dtype=float)


def generateSpiralBodies(numOfPoints, arms, spiralDensity):
    theta = np.linspace(5, arms * 2 * np.pi, numOfPoints)
    r = np.sqrt(theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x += np.random.normal(0, spiralDensity, numOfPoints)
    y += np.random.normal(0, spiralDensity, numOfPoints)

    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * SIZE
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * SIZE

    bodies = []
    for i in range(numOfPoints):
        centreVector = np.array([SIZE / 2, SIZE / 2]) - np.array([x[i], y[i]])
        perpendicularVelVector = np.array([-centreVector[1], centreVector[0]])
        normalizedVel = (
            5 * perpendicularVelVector / np.linalg.norm(perpendicularVelVector)
        )
        bodies.append(
            Body(
                mass=10,
                pos=[x[i], y[i]],
                vel=normalizedVel,
            )
        )

    return bodies


bodies = generateSpiralBodies(200, 6, 0.1)
bodies.insert(0, Body(mass=10000, pos=[SIZE / 2, SIZE / 2], vel=[0, 0]))

toggleQuads = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_t:
                toggleQuads = not toggleQuads

    screen.fill(BLACK)
    barnesHut(bodies, toggleQuads)
    for body in bodies:
        body.draw(screen)

    pygame.display.update()

import Foundation

/// Snake game environment.
/// The snake moves on a grid, eats food to grow, and dies if it hits a wall or itself.
///
/// Observation (11 features):
///   - danger straight, danger right, danger left (3)
///   - direction: up, right, down, left (4)
///   - food: up, right, down, left relative to head (4)
///
/// Actions: 0 = go straight, 1 = turn right, 2 = turn left
public struct Snake: Environment {
    public let observationSize: Int = 11
    public let actionCount: Int = 3
    public let gridSize: Int
    public let maxStepsWithoutFood: Int

    // State
    private var body: [(Int, Int)] = []
    private var direction: Direction = .right
    private var food: (Int, Int) = (0, 0)
    private var stepsSinceFood: Int = 0
    private var score: Int = 0

    private enum Direction: Int {
        case up = 0, right = 1, down = 2, left = 3

        var dx: Int {
            switch self { case .up: 0; case .right: 1; case .down: 0; case .left: -1 }
        }
        var dy: Int {
            switch self { case .up: -1; case .right: 0; case .down: 1; case .left: 0 }
        }

        var turnRight: Direction {
            Direction(rawValue: (rawValue + 1) % 4)!
        }
        var turnLeft: Direction {
            Direction(rawValue: (rawValue + 3) % 4)!
        }
    }

    public init(gridSize: Int = 10, maxStepsWithoutFood: Int? = nil) {
        self.gridSize = gridSize
        self.maxStepsWithoutFood = maxStepsWithoutFood ?? gridSize * gridSize
    }

    public mutating func reset() -> [Double] {
        let mid = gridSize / 2
        body = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        direction = .right
        score = 0
        stepsSinceFood = 0
        placeFood()
        return observation
    }

    public mutating func step(action: Int) -> StepResult {
        // Turn
        switch action {
        case 1: direction = direction.turnRight
        case 2: direction = direction.turnLeft
        default: break // go straight
        }

        // Move head
        let head = body[0]
        let newHead = (head.0 + direction.dx, head.1 + direction.dy)

        // Check death
        let hitWall = newHead.0 < 0 || newHead.0 >= gridSize || newHead.1 < 0 || newHead.1 >= gridSize
        let hitSelf = body.contains(where: { $0.0 == newHead.0 && $0.1 == newHead.1 })

        if hitWall || hitSelf {
            return StepResult(observation: observation, reward: -10.0, done: true)
        }

        body.insert(newHead, at: 0)
        stepsSinceFood += 1

        // Check food
        var reward: Double
        if newHead.0 == food.0 && newHead.1 == food.1 {
            score += 1
            reward = 10.0
            stepsSinceFood = 0
            placeFood()
        } else {
            body.removeLast()
            reward = -0.01 // small step penalty
        }

        // Timeout
        if stepsSinceFood >= maxStepsWithoutFood {
            return StepResult(observation: observation, reward: -5.0, done: true)
        }

        return StepResult(observation: observation, reward: reward, done: false)
    }

    // MARK: - Public accessors for rendering

    public var snakeBody: [(Int, Int)] { body }
    public var foodPosition: (Int, Int) { food }
    public var currentScore: Int { score }
    public var headDirection: Int { direction.rawValue }

    // MARK: - Private

    private mutating func placeFood() {
        let bodySet = Set(body.map { "\($0.0),\($0.1)" })
        var attempts = 0
        repeat {
            food = (Int.random(in: 0..<gridSize), Int.random(in: 0..<gridSize))
            attempts += 1
        } while bodySet.contains("\(food.0),\(food.1)") && attempts < 100
    }

    private var observation: [Double] {
        let head = body[0]

        // Danger detection
        func isDanger(_ dir: Direction) -> Double {
            let nx = head.0 + dir.dx
            let ny = head.1 + dir.dy
            if nx < 0 || nx >= gridSize || ny < 0 || ny >= gridSize { return 1.0 }
            if body.contains(where: { $0.0 == nx && $0.1 == ny }) { return 1.0 }
            return 0.0
        }

        let dangerStraight = isDanger(direction)
        let dangerRight = isDanger(direction.turnRight)
        let dangerLeft = isDanger(direction.turnLeft)

        // Direction one-hot
        let dirUp: Double = direction == .up ? 1.0 : 0.0
        let dirRight: Double = direction == .right ? 1.0 : 0.0
        let dirDown: Double = direction == .down ? 1.0 : 0.0
        let dirLeft: Double = direction == .left ? 1.0 : 0.0

        // Food direction
        let foodUp: Double = food.1 < head.1 ? 1.0 : 0.0
        let foodRight: Double = food.0 > head.0 ? 1.0 : 0.0
        let foodDown: Double = food.1 > head.1 ? 1.0 : 0.0
        let foodLeft: Double = food.0 < head.0 ? 1.0 : 0.0

        return [dangerStraight, dangerRight, dangerLeft,
                dirUp, dirRight, dirDown, dirLeft,
                foodUp, foodRight, foodDown, foodLeft]
    }
}

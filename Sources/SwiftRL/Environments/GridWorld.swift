import Foundation

/// A simple grid navigation environment.
/// Agent starts at (0,0), goal is at (size-1, size-1).
/// Actions: 0=up, 1=right, 2=down, 3=left.
public struct GridWorld: Environment {
    public let size: Int
    public let observationSize: Int  // row + col (normalized)
    public let actionCount: Int = 4
    public let maxSteps: Int

    private var agentRow: Int = 0
    private var agentCol: Int = 0
    private var stepCount: Int = 0

    public init(size: Int = 8, maxSteps: Int? = nil) {
        self.size = size
        self.observationSize = 2
        self.maxSteps = maxSteps ?? size * size * 2
    }

    public mutating func reset() -> [Double] {
        agentRow = 0
        agentCol = 0
        stepCount = 0
        return observation
    }

    public mutating func step(action: Int) -> StepResult {
        stepCount += 1

        switch action {
        case 0: agentRow = max(0, agentRow - 1)           // up
        case 1: agentCol = min(size - 1, agentCol + 1)    // right
        case 2: agentRow = min(size - 1, agentRow + 1)    // down
        case 3: agentCol = max(0, agentCol - 1)            // left
        default: break
        }

        let atGoal = agentRow == size - 1 && agentCol == size - 1
        let timeout = stepCount >= maxSteps

        let reward: Double
        if atGoal {
            reward = 1.0
        } else if timeout {
            reward = -0.1
        } else {
            reward = -0.01  // small step penalty to encourage efficiency
        }

        return StepResult(
            observation: observation,
            reward: reward,
            done: atGoal || timeout
        )
    }

    private var observation: [Double] {
        [Double(agentRow) / Double(size - 1), Double(agentCol) / Double(size - 1)]
    }
}

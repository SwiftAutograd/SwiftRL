import Foundation

/// A single transition in the environment.
public struct Experience: Sendable {
    public let state: [Double]
    public let action: Int
    public let reward: Double
    public let nextState: [Double]
    public let done: Bool

    public init(state: [Double], action: Int, reward: Double, nextState: [Double], done: Bool) {
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.done = done
    }
}

/// A reinforcement learning agent that can act and learn.
public protocol Agent {
    /// Choose an action given an observation.
    mutating func act(observation: [Double]) -> Int
    /// Run one training step. Returns the loss (or 0 if no update occurred).
    mutating func update() -> Double
}

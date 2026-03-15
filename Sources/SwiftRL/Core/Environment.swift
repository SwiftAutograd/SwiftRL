import Foundation

/// The result of taking one step in an environment.
public struct StepResult: Sendable {
    /// The observation after the action was taken.
    public let observation: [Double]
    /// The reward received for this transition.
    public let reward: Double
    /// Whether the episode has ended.
    public let done: Bool

    public init(observation: [Double], reward: Double, done: Bool) {
        self.observation = observation
        self.reward = reward
        self.done = done
    }
}

/// A reinforcement learning environment with discrete actions.
///
/// Follows the Gymnasium convention: reset() → observe, step(action) → (observe, reward, done).
public protocol Environment {
    /// The size of the observation vector.
    var observationSize: Int { get }
    /// The number of discrete actions available.
    var actionCount: Int { get }
    /// Reset the environment and return the initial observation.
    mutating func reset() -> [Double]
    /// Take an action and return the result.
    mutating func step(action: Int) -> StepResult
}

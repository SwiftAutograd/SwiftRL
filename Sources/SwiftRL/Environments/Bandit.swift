import Foundation

/// A k-armed bandit environment. Each arm has a fixed mean reward (Gaussian).
/// The agent must learn which arm gives the highest reward.
public struct Bandit: Environment {
    public let observationSize: Int = 1  // dummy observation (bandits are stateless)
    public let actionCount: Int
    private let means: [Double]
    private let stddev: Double

    /// Create a bandit with k arms. Means are sampled uniformly from [0, 1].
    public init(arms: Int = 10, stddev: Double = 1.0) {
        self.actionCount = arms
        self.means = (0..<arms).map { _ in Double.random(in: 0...1) }
        self.stddev = stddev
    }

    /// Create a bandit with explicit reward means.
    public init(means: [Double], stddev: Double = 1.0) {
        self.actionCount = means.count
        self.means = means
        self.stddev = stddev
    }

    public mutating func reset() -> [Double] {
        [0.0]  // stateless — observation is always the same
    }

    public mutating func step(action: Int) -> StepResult {
        precondition(action >= 0 && action < actionCount)
        // Gaussian reward centered on the arm's mean
        let u1 = Double.random(in: 0..<1)
        let u2 = Double.random(in: 0..<1)
        let normal = sqrt(-2.0 * Darwin.log(u1)) * cos(2.0 * .pi * u2)
        let reward = means[action] + stddev * normal
        return StepResult(observation: [0.0], reward: reward, done: true)
    }

    /// The index of the optimal arm (for evaluation).
    public var optimalArm: Int {
        means.enumerated().max(by: { $0.element < $1.element })!.offset
    }
}

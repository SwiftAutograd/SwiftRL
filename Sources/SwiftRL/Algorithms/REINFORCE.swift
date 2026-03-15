import SwiftGrad
import Foundation

/// REINFORCE (Monte Carlo Policy Gradient).
///
/// The simplest policy gradient algorithm. Collects full episodes,
/// computes discounted returns, and updates the policy using:
///   loss = -Σ log(π(a|s)) * G_t
///
/// Uses an MLP with softmax output as the policy network.
public struct REINFORCE: Agent {
    public let policy: MLP
    public let optimizer: Adam
    public let gamma: Double

    // Episode buffer
    private var logProbs: [Value] = []
    private var rewards: [Double] = []

    public init(
        observationSize: Int,
        hiddenSizes: [Int] = [32],
        actionCount: Int,
        learningRate: Double = 0.001,
        gamma: Double = 0.99
    ) {
        self.policy = MLP(inputSize: observationSize, layerSizes: hiddenSizes + [actionCount])
        self.optimizer = Adam(parameters: policy.parameters(), learningRate: learningRate)
        self.gamma = gamma
    }

    /// Choose an action by sampling from the policy's softmax distribution.
    public mutating func act(observation: [Double]) -> Int {
        let inputs = observation.map { Value($0) }
        let logits = policy(inputs)
        let probs = softmax(logits)

        // Sample action
        let probValues = probs.map(\.data)
        let action = sampleCategorical(probs: probValues)

        // Store log probability for the chosen action
        logProbs.append(probs[action].log())

        return action
    }

    /// Record a reward for the current timestep.
    public mutating func recordReward(_ reward: Double) {
        rewards.append(reward)
    }

    /// Compute discounted returns and update the policy. Call at end of episode.
    /// Returns the loss value.
    public mutating func update() -> Double {
        guard !rewards.isEmpty else { return 0 }

        // Compute discounted returns (G_t)
        var returns: [Double] = Array(repeating: 0, count: rewards.count)
        var g = 0.0
        for t in stride(from: rewards.count - 1, through: 0, by: -1) {
            g = rewards[t] + gamma * g
            returns[t] = g
        }

        // Normalize returns for stability
        let mean = returns.reduce(0, +) / Double(returns.count)
        let variance = returns.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(returns.count)
        let std = sqrt(variance + 1e-8)
        let normalizedReturns = returns.map { ($0 - mean) / std }

        // Compute policy gradient loss: -Σ log(π(a|s)) * G_t
        var loss = Value(0.0)
        for (logProb, g) in zip(logProbs, normalizedReturns) {
            loss = loss + logProb * (-g)
        }

        // Backpropagate and update
        policy.zeroGrad()
        loss.backward()
        optimizer.step()

        let lossValue = loss.data

        // Clear episode buffer
        logProbs.removeAll()
        rewards.removeAll()

        return lossValue
    }
}

// MARK: - Training Helper

extension REINFORCE {
    /// Train for a number of episodes on an environment. Returns reward history.
    public mutating func train(
        environment: inout some Environment,
        episodes: Int,
        onEpisode: ((Int, Double) -> Void)? = nil
    ) -> [Double] {
        var rewardHistory: [Double] = []

        for episode in 0..<episodes {
            var obs = environment.reset()
            var totalReward = 0.0

            while true {
                let action = act(observation: obs)
                let result = environment.step(action: action)
                recordReward(result.reward)
                totalReward += result.reward
                obs = result.observation

                if result.done { break }
            }

            _ = update()
            rewardHistory.append(totalReward)
            onEpisode?(episode, totalReward)
        }

        return rewardHistory
    }
}

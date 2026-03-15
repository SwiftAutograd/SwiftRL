import SwiftGrad
import Foundation

/// Deep Q-Network (DQN) with experience replay and target network.
///
/// The Q-network outputs Q(s, a) for all actions. Training minimizes:
///   loss = MSE(Q(s,a), r + γ * max_a' Q_target(s', a'))
public struct DQN: Agent {
    public let qNetwork: MLP
    public let targetNetwork: MLP
    public let optimizer: Adam
    public let gamma: Double
    public let batchSize: Int
    public let targetUpdateFrequency: Int
    public let epsilonSchedule: any EpsilonSchedule

    public private(set) var replayBuffer: ReplayBuffer
    private var totalSteps: Int = 0
    private let actionCount: Int

    public init(
        observationSize: Int,
        hiddenSizes: [Int] = [64, 32],
        actionCount: Int,
        learningRate: Double = 0.001,
        gamma: Double = 0.99,
        bufferCapacity: Int = 10_000,
        batchSize: Int = 32,
        targetUpdateFrequency: Int = 100,
        epsilonSchedule: any EpsilonSchedule = ExponentialDecay()
    ) {
        self.qNetwork = MLP(inputSize: observationSize, layerSizes: hiddenSizes + [actionCount])
        self.targetNetwork = MLP(inputSize: observationSize, layerSizes: hiddenSizes + [actionCount])
        self.optimizer = Adam(parameters: qNetwork.parameters(), learningRate: learningRate)
        self.gamma = gamma
        self.batchSize = batchSize
        self.targetUpdateFrequency = targetUpdateFrequency
        self.epsilonSchedule = epsilonSchedule
        self.replayBuffer = ReplayBuffer(capacity: bufferCapacity)
        self.actionCount = actionCount

        // Initialize target network with same weights
        syncTargetNetwork()
    }

    /// Choose an action using epsilon-greedy policy.
    public mutating func act(observation: [Double]) -> Int {
        let eps = epsilonSchedule.epsilon(step: totalSteps)

        if Double.random(in: 0..<1) < eps {
            // Random exploration
            return Int.random(in: 0..<actionCount)
        } else {
            // Greedy: pick action with highest Q-value
            let inputs = observation.map { Value($0) }
            let qValues = qNetwork(inputs)
            return qValues.enumerated().max(by: { $0.element.data < $1.element.data })!.offset
        }
    }

    /// Store an experience in the replay buffer.
    public mutating func store(_ experience: Experience) {
        replayBuffer.append(experience)
    }

    /// Sample a minibatch and perform one gradient update. Returns the loss.
    public mutating func update() -> Double {
        totalSteps += 1

        guard replayBuffer.count >= batchSize else { return 0 }

        let batch = replayBuffer.sample(batchSize: batchSize)

        var totalLoss = Value(0.0)

        for exp in batch {
            // Current Q-value for the taken action
            let stateInputs = exp.state.map { Value($0) }
            let qValues = qNetwork(stateInputs)
            let qValue = qValues[exp.action]

            // Target Q-value
            let target: Double
            if exp.done {
                target = exp.reward
            } else {
                let nextInputs = exp.nextState.map { Value($0) }
                let nextQValues = targetNetwork(nextInputs)
                let maxNextQ = nextQValues.map(\.data).max() ?? 0.0
                target = exp.reward + gamma * maxNextQ
            }

            // TD error
            let diff = qValue - target
            totalLoss = totalLoss + diff * diff
        }

        let loss = totalLoss * (1.0 / Double(batchSize))

        // Backpropagate and update
        qNetwork.zeroGrad()
        loss.backward()
        optimizer.step()

        // Periodically sync target network
        if totalSteps % targetUpdateFrequency == 0 {
            syncTargetNetwork()
        }

        return loss.data
    }

    /// Copy weights from Q-network to target network.
    private func syncTargetNetwork() {
        let qParams = qNetwork.parameters()
        let tParams = targetNetwork.parameters()
        for (tp, qp) in zip(tParams, qParams) {
            tp.data = qp.data
        }
    }
}

// MARK: - Training Helper

extension DQN {
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

                store(Experience(
                    state: obs,
                    action: action,
                    reward: result.reward,
                    nextState: result.observation,
                    done: result.done
                ))

                _ = update()
                totalReward += result.reward
                obs = result.observation

                if result.done { break }
            }

            rewardHistory.append(totalReward)
            onEpisode?(episode, totalReward)
        }

        return rewardHistory
    }
}

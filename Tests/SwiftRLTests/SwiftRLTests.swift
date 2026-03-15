import Testing
@testable import SwiftRL
import SwiftGrad

// MARK: - ReplayBuffer Tests

@Test func testReplayBufferAppendAndSample() {
    var buffer = ReplayBuffer(capacity: 100)
    #expect(buffer.count == 0)

    for i in 0..<50 {
        buffer.append(Experience(
            state: [Double(i)], action: 0, reward: 1.0,
            nextState: [Double(i + 1)], done: false
        ))
    }

    #expect(buffer.count == 50)

    let batch = buffer.sample(batchSize: 10)
    #expect(batch.count == 10)
}

@Test func testReplayBufferOverwrite() {
    var buffer = ReplayBuffer(capacity: 5)
    for i in 0..<10 {
        buffer.append(Experience(
            state: [Double(i)], action: 0, reward: Double(i),
            nextState: [Double(i + 1)], done: false
        ))
    }
    #expect(buffer.count == 5)
}

// MARK: - Softmax Tests

@Test func testSoftmaxSumsToOne() {
    let logits = [Value(1.0), Value(2.0), Value(3.0)]
    let probs = softmax(logits)
    let sum = probs.map(\.data).reduce(0, +)
    #expect(abs(sum - 1.0) < 1e-6)
}

@Test func testSoftmaxGradients() {
    let logits = [Value(1.0), Value(2.0), Value(3.0)]
    let probs = softmax(logits)
    // Take log of first prob and backprop
    let logP = probs[0].log()
    logP.backward()
    // Gradients should be non-zero
    #expect(logits[0].grad != 0)
    #expect(logits[1].grad != 0)
}

@Test func testCategoricalSampling() {
    // Deterministic-ish: with probs [0, 0, 1], should always pick action 2
    var counts = [0, 0, 0]
    for _ in 0..<100 {
        let action = sampleCategorical(probs: [0.0, 0.0, 1.0])
        counts[action] += 1
    }
    #expect(counts[2] == 100)
}

// MARK: - Adam Tests

@Test func testAdamConverges() {
    // Minimize (x - 3)^2 using Adam
    let x = Value(10.0)
    let adam = Adam(parameters: [x], learningRate: 0.1)

    for _ in 0..<200 {
        x.grad = 0
        let loss = (x - 3.0).power(2)
        loss.backward()
        adam.step()
    }

    #expect(abs(x.data - 3.0) < 0.1)
}

// MARK: - Environment Tests

@Test func testGridWorldBasic() {
    var env = GridWorld(size: 4)
    let obs = env.reset()
    #expect(obs.count == 2)
    #expect(obs[0] == 0.0)
    #expect(obs[1] == 0.0)

    // Move right
    let result = env.step(action: 1)
    #expect(result.observation[1] > 0)
    #expect(!result.done || result.reward != 0)
}

@Test func testGridWorldReachesGoal() {
    var env = GridWorld(size: 3)
    _ = env.reset()
    // Navigate to (2,2): right, right, down, down
    _ = env.step(action: 1)
    _ = env.step(action: 1)
    _ = env.step(action: 2)
    let result = env.step(action: 2)
    #expect(result.done)
    #expect(result.reward == 1.0)
}

@Test func testCartPoleBasic() {
    var env = CartPole()
    let obs = env.reset()
    #expect(obs.count == 4)

    let result = env.step(action: 1)
    #expect(result.observation.count == 4)
}

@Test func testBanditBasic() {
    var env = Bandit(means: [0.0, 0.5, 1.0], stddev: 0.1)
    #expect(env.actionCount == 3)
    #expect(env.optimalArm == 2)

    _ = env.reset()
    let result = env.step(action: 2)
    #expect(result.done) // bandits are single-step
}

// MARK: - Epsilon Schedule Tests

@Test func testLinearDecay() {
    let schedule = LinearDecay(start: 1.0, end: 0.1, steps: 100)
    #expect(schedule.epsilon(step: 0) == 1.0)
    #expect(abs(schedule.epsilon(step: 50) - 0.55) < 0.01)
    #expect(abs(schedule.epsilon(step: 100) - 0.1) < 0.01)
    #expect(abs(schedule.epsilon(step: 200) - 0.1) < 0.01)
}

@Test func testExponentialDecay() {
    let schedule = ExponentialDecay(start: 1.0, decayRate: 0.99, minimum: 0.01)
    #expect(schedule.epsilon(step: 0) == 1.0)
    #expect(schedule.epsilon(step: 100) > 0.01)  // not yet at minimum
    #expect(schedule.epsilon(step: 100000) <= 0.01)  // at or below minimum
}

// MARK: - REINFORCE Integration Test

@Test func testREINFORCELearnsBandit() {
    var agent = REINFORCE(
        observationSize: 1,
        hiddenSizes: [16],
        actionCount: 3,
        learningRate: 0.01,
        gamma: 0.99
    )
    var env = Bandit(means: [0.1, 0.5, 0.9], stddev: 0.1)

    let rewards = agent.train(environment: &env, episodes: 500)

    // Average reward over last 100 episodes should improve
    let earlyAvg = rewards.prefix(100).reduce(0, +) / 100.0
    let lateAvg = rewards.suffix(100).reduce(0, +) / 100.0
    #expect(lateAvg > earlyAvg)
}

// MARK: - DQN Integration Test

@Test func testDQNLearnsGridWorld() {
    var agent = DQN(
        observationSize: 2,
        hiddenSizes: [32],
        actionCount: 4,
        learningRate: 0.001,
        gamma: 0.99,
        bufferCapacity: 5000,
        batchSize: 16,
        targetUpdateFrequency: 50,
        epsilonSchedule: LinearDecay(start: 1.0, end: 0.05, steps: 300)
    )
    var env = GridWorld(size: 4)

    let rewards = agent.train(environment: &env, episodes: 400)

    // Average reward should improve over time
    let earlyAvg = rewards.prefix(50).reduce(0, +) / 50.0
    let lateAvg = rewards.suffix(50).reduce(0, +) / 50.0
    #expect(lateAvg > earlyAvg)
}

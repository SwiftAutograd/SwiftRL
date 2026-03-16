import Testing
import Foundation
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

@Test func testReplayBufferSampleWhenNearlyEmpty() {
    var buffer = ReplayBuffer(capacity: 100)
    buffer.append(Experience(
        state: [1.0], action: 0, reward: 1.0,
        nextState: [2.0], done: false
    ))
    #expect(buffer.count == 1)

    let batch = buffer.sample(batchSize: 1)
    #expect(batch.count == 1)
    #expect(batch[0].state == [1.0])
}

@Test func testReplayBufferClear() {
    var buffer = ReplayBuffer(capacity: 10)
    for i in 0..<5 {
        buffer.append(Experience(
            state: [Double(i)], action: 0, reward: 1.0,
            nextState: [Double(i + 1)], done: false
        ))
    }
    #expect(buffer.count == 5)

    buffer.clear()
    #expect(buffer.count == 0)

    // Buffer should still be usable after clearing
    buffer.append(Experience(
        state: [99.0], action: 1, reward: 5.0,
        nextState: [100.0], done: true
    ))
    #expect(buffer.count == 1)
    let batch = buffer.sample(batchSize: 1)
    #expect(batch[0].reward == 5.0)
}

@Test func testReplayBufferIsFull() {
    var buffer = ReplayBuffer(capacity: 3)
    #expect(!buffer.isFull)
    for i in 0..<3 {
        buffer.append(Experience(
            state: [Double(i)], action: 0, reward: 1.0,
            nextState: [Double(i + 1)], done: false
        ))
    }
    #expect(buffer.isFull)
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

@Test func testLogSoftmax() {
    let logits = [Value(1.0), Value(2.0), Value(3.0)]
    let logProbs = logSoftmax(logits)

    // log-softmax values should be negative (log of probabilities < 1)
    for lp in logProbs {
        #expect(lp.data < 0)
    }

    // exp(log-softmax) should sum to 1
    let probSum = logProbs.map { exp($0.data) }.reduce(0, +)
    #expect(abs(probSum - 1.0) < 1e-6)

    // Should match log(softmax(x)) numerically
    let logits2 = [Value(1.0), Value(2.0), Value(3.0)]
    let naiveLogProbs = softmax(logits2).map { Foundation.log($0.data) }
    for (a, b) in zip(logProbs.map(\.data), naiveLogProbs) {
        #expect(abs(a - b) < 1e-6)
    }
}

@Test func testLogSoftmaxGradients() {
    let logits = [Value(2.0), Value(1.0), Value(0.5)]
    let logProbs = logSoftmax(logits)
    // Backprop through the first element
    logProbs[0].backward()
    // All logits should receive gradients
    for logit in logits {
        #expect(logit.grad != 0)
    }
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

@Test func testAdamMultipleParameters() {
    // Minimize (x - 2)^2 + (y - 5)^2
    let x = Value(0.0)
    let y = Value(0.0)
    let adam = Adam(parameters: [x, y], learningRate: 0.1)

    for _ in 0..<300 {
        x.grad = 0
        y.grad = 0
        let loss = (x - 2.0).power(2) + (y - 5.0).power(2)
        loss.backward()
        adam.step()
    }

    #expect(abs(x.data - 2.0) < 0.2)
    #expect(abs(y.data - 5.0) < 0.2)
}

// MARK: - RewardTracker Tests

@Test func testRewardTrackerBasic() {
    var tracker = RewardTracker(windowSize: 3)
    #expect(tracker.totalEpisodes == 0)
    #expect(tracker.lastReward == nil)
    #expect(tracker.runningAverage == 0)
    #expect(tracker.bestReward == 0)

    tracker.record(10.0)
    #expect(tracker.totalEpisodes == 1)
    #expect(tracker.lastReward == 10.0)
    #expect(tracker.runningAverage == 10.0)
    #expect(tracker.bestReward == 10.0)

    tracker.record(20.0)
    tracker.record(30.0)
    #expect(tracker.totalEpisodes == 3)
    #expect(tracker.runningAverage == 20.0) // (10+20+30)/3

    // After window overflows, running average uses only the last windowSize entries
    tracker.record(100.0)
    #expect(tracker.totalEpisodes == 4)
    // Window of 3: [20, 30, 100] -> avg = 50
    #expect(abs(tracker.runningAverage - 50.0) < 1e-6)
    #expect(tracker.bestReward == 100.0)
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

// MARK: - Snake Environment Tests

@Test func testSnakeReset() {
    var env = Snake(gridSize: 10)
    let obs = env.reset()
    #expect(obs.count == 11)
    #expect(env.snakeBody.count == 3)
    #expect(env.currentScore == 0)
}

@Test func testSnakeDeathByWall() {
    var env = Snake(gridSize: 10)
    _ = env.reset()
    // Snake starts at (5,5) heading right. Move straight until hitting the right wall.
    var done = false
    var stepsTaken = 0
    while !done && stepsTaken < 20 {
        let result = env.step(action: 0) // go straight (right)
        done = result.done
        if done {
            #expect(result.reward == -10.0)
        }
        stepsTaken += 1
    }
    #expect(done)
}

@Test func testSnakeTurning() {
    var env = Snake(gridSize: 10)
    _ = env.reset()
    // Initially heading right (direction = 1)
    #expect(env.headDirection == 1) // right

    // Turn right -> heading down
    _ = env.step(action: 1)
    #expect(env.headDirection == 2) // down

    // Turn left -> heading right again
    _ = env.step(action: 2)
    #expect(env.headDirection == 1) // right
}

@Test func testSnakeObservationSize() {
    var env = Snake(gridSize: 6)
    #expect(env.observationSize == 11)
    #expect(env.actionCount == 3)
    let obs = env.reset()
    #expect(obs.count == env.observationSize)

    let result = env.step(action: 0)
    #expect(result.observation.count == env.observationSize)
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

    let rewards = agent.train(environment: &env, episodes: 1000)

    // Average reward over last 100 episodes should be close to the best arm (0.9)
    let lateAvg = rewards.suffix(100).reduce(0, +) / 100.0
    #expect(lateAvg > 0.5)
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

# SwiftRL

On-device **reinforcement learning** for Swift. Built on [SwiftGrad](https://github.com/SwiftAutograd/SwiftGrad)'s autograd engine.

SwiftRL brings reinforcement learning to iOS, macOS, and visionOS - no Python, no server, no cloud. Train RL agents directly on Apple devices with real-time gradient computation through SwiftGrad's `backward()`.

## Status

SwiftRL is in active development. The autograd foundation ([SwiftGrad](https://github.com/SwiftAutograd/SwiftGrad)) is complete and tested.

## The Problem

There is **no reinforcement learning library for Swift**. The only attempt ([swift-rl](https://github.com/eaplatanios/swift-rl)) died in 2021 when Swift for TensorFlow was archived. Every RL tool today - Stable-Baselines3, CleanRL, RLlib, Unity ML-Agents - requires Python and cannot run on iOS.

Meanwhile:
- AI in gaming is a **$5.85B market** growing to $38B by 2034
- Mobile holds **52% of the AI gaming market**
- Games using adaptive AI see **~30% higher engagement**
- Apple has **28 million registered developers** with zero RL tools

## Why Swift?

| Advantage | Why It Matters for RL |
|---|---|
| **Real-time performance** | Policy updates within 16ms frame budgets. No GIL, no GC pauses. |
| **Privacy by default** | RL agents learn from user behavior that never leaves the device. |
| **Native game integration** | Direct access to SpriteKit, RealityKit, GameplayKit game loops. |
| **Unified memory** | Apple Silicon shares CPU/GPU memory - no data copies for training. |
| **visionOS exclusive** | Spatial computing is Swift-only. Adaptive spatial agents require Swift. |

## Planned Architecture

```
SwiftRL
├── Core
│   ├── Environment        - Protocol: step(action) → (state, reward, done)
│   ├── ReplayBuffer       - Uniform and prioritized experience replay
│   ├── Policy             - Protocol for policy networks
│   └── Trainer            - Training loop orchestration
├── Algorithms
│   ├── REINFORCE          - Simplest policy gradient
│   ├── DQN               - Deep Q-Network with target network
│   ├── A2C               - Advantage Actor-Critic
│   └── PPO               - Proximal Policy Optimization
├── Environments
│   ├── GridWorld          - Navigation with obstacles
│   ├── CartPole           - Classic control benchmark
│   └── Bandit             - Multi-armed bandit
└── Optimizers
    ├── SGD               - (from SwiftGrad)
    └── Adam              - Adaptive moment estimation
```

## Planned Usage

```swift
import SwiftRL

// Define an environment
let env = GridWorld(size: 8)

// Create a policy network (powered by SwiftGrad)
let policy = MLP(inputSize: env.observationSize, layerSizes: [64, 32, env.actionCount])

// Train with DQN
let agent = DQN(
    policy: policy,
    learningRate: 0.001,
    gamma: 0.99,
    epsilon: DecayingEpsilon(start: 1.0, end: 0.01, decay: 0.995)
)

// Training loop
for episode in 0..<1000 {
    let reward = agent.train(environment: env)
    if episode % 100 == 0 {
        print("Episode \(episode): reward = \(reward)")
    }
}

// Use the trained agent
let action = agent.act(observation: env.reset())
```

## Target Use Cases

| Use Case | RL Algorithm | Platform |
|---|---|---|
| **Adaptive game NPCs** | PPO / DQN | iOS, visionOS |
| **Dynamic difficulty** | Contextual bandits → PPO | iOS |
| **Smart notifications** | Multi-armed bandit | iOS, watchOS |
| **Spatial agents** | PPO with continuous actions | visionOS |
| **Automated playtesting** | DQN / A2C | macOS |
| **Personalized fitness** | Contextual bandits | watchOS, iOS |

## Demo Apps

See [SwiftRLDemos](https://github.com/SwiftAutograd/SwiftRLDemos) for playable iOS apps showcasing SwiftRL:
- Snake - DQN learns to hunt food in real-time
- 2048 - Policy gradient discovers tile-merging strategies
- Connect Four - Self-play TD learning
- Blackjack - Monte Carlo policy evaluation

## Part of the SwiftAutograd Organization

| Repository | Description | Status |
|---|---|---|
| [SwiftGrad](https://github.com/SwiftAutograd/SwiftGrad) | Autograd engine | Working |
| **[SwiftRL](https://github.com/SwiftAutograd/SwiftRL)** | Reinforcement learning (you are here) | In development |
| [SwiftRLDemos](https://github.com/SwiftAutograd/SwiftRLDemos) | Demo apps | Planned |

## Research & Inspiration

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy - the autograd engine SwiftGrad is built on
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - single-file RL implementations we aim to match in clarity
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) - the closest analog (but Python-dependent, desktop-only training)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - the API design standard for RL libraries
- Apple ML Research - [8+ RL papers](https://machinelearning.apple.com) published 2023-2025

## Contributing

SwiftRL is in early development. If you're interested in contributing, open an issue to discuss before submitting a PR.

## License

MIT - see [LICENSE](LICENSE).

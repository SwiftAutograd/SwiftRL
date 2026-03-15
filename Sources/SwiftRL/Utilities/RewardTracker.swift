import Foundation

/// Tracks episode rewards and computes running statistics.
public struct RewardTracker: Sendable {
    private var rewards: [Double] = []
    private let windowSize: Int

    public var totalEpisodes: Int { rewards.count }
    public var lastReward: Double? { rewards.last }

    public init(windowSize: Int = 100) {
        self.windowSize = windowSize
    }

    /// Record the total reward from a completed episode.
    public mutating func record(_ reward: Double) {
        rewards.append(reward)
    }

    /// Running average over the last `windowSize` episodes.
    public var runningAverage: Double {
        guard !rewards.isEmpty else { return 0 }
        let window = rewards.suffix(windowSize)
        return window.reduce(0, +) / Double(window.count)
    }

    /// Best single-episode reward seen so far.
    public var bestReward: Double {
        rewards.max() ?? 0
    }
}

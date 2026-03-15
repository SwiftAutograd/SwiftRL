import Foundation

/// A schedule that produces an epsilon value for epsilon-greedy exploration.
public protocol EpsilonSchedule: Sendable {
    func epsilon(step: Int) -> Double
}

/// Linear decay from start to end over a fixed number of steps.
public struct LinearDecay: EpsilonSchedule {
    public let start: Double
    public let end: Double
    public let steps: Int

    public init(start: Double = 1.0, end: Double = 0.01, steps: Int = 1000) {
        self.start = start
        self.end = end
        self.steps = steps
    }

    public func epsilon(step: Int) -> Double {
        if step >= steps { return end }
        return start + (end - start) * Double(step) / Double(steps)
    }
}

/// Exponential decay: start * decayRate^step, clamped to a minimum.
public struct ExponentialDecay: EpsilonSchedule {
    public let start: Double
    public let decayRate: Double
    public let minimum: Double

    public init(start: Double = 1.0, decayRate: Double = 0.995, minimum: Double = 0.01) {
        self.start = start
        self.decayRate = decayRate
        self.minimum = minimum
    }

    public func epsilon(step: Int) -> Double {
        max(minimum, start * pow(decayRate, Double(step)))
    }
}

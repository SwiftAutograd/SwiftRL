import SwiftGrad
import Foundation

/// Compute softmax over an array of Value logits (numerically stable).
/// Returns an array of Value probabilities that sum to 1.
public func softmax(_ logits: [Value]) -> [Value] {
    // Subtract max for numerical stability (max is treated as a constant)
    let maxLogit = logits.map(\.data).max() ?? 0.0
    let exps = logits.map { ($0 - maxLogit).exp() }
    let sumExp = exps.dropFirst().reduce(exps[0], +)
    return exps.map { $0 / sumExp }
}

/// Sample an action index from a categorical distribution defined by probabilities.
public func sampleCategorical(probs: [Double]) -> Int {
    let r = Double.random(in: 0..<1)
    var cumulative = 0.0
    for (i, p) in probs.enumerated() {
        cumulative += p
        if r < cumulative { return i }
    }
    return probs.count - 1
}

/// Compute log-softmax (more numerically stable than log(softmax(x))).
public func logSoftmax(_ logits: [Value]) -> [Value] {
    let maxLogit = logits.map(\.data).max() ?? 0.0
    let shifted = logits.map { $0 - maxLogit }
    let exps = shifted.map { $0.exp() }
    let logSumExp = exps.dropFirst().reduce(exps[0], +).log()
    return shifted.map { $0 - logSumExp }
}

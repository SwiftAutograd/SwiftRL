import SwiftGrad
import Foundation

/// Adam optimizer (Kingma & Ba, 2014).
public final class Adam {
    public let parameters: [Value]
    public var learningRate: Double
    public let beta1: Double
    public let beta2: Double
    public let epsilon: Double

    private var m: [Double]  // first moment
    private var v: [Double]  // second moment
    private var t: Int = 0   // timestep

    public init(
        parameters: [Value],
        learningRate: Double = 0.001,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8
    ) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = Array(repeating: 0.0, count: parameters.count)
        self.v = Array(repeating: 0.0, count: parameters.count)
    }

    /// Perform one optimization step.
    public func step() {
        t += 1
        for i in 0..<parameters.count {
            let g = parameters[i].grad
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            let mHat = m[i] / (1 - pow(beta1, Double(t)))
            let vHat = v[i] / (1 - pow(beta2, Double(t)))
            parameters[i].data -= learningRate * mHat / (sqrt(vHat) + epsilon)
        }
    }
}

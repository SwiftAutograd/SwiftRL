import Foundation

/// Classic CartPole-v1 environment.
/// A pole is attached to a cart on a frictionless track. The goal is to keep the pole
/// balanced by applying forces to the cart.
///
/// Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
/// Actions: 0 = push left, 1 = push right
public struct CartPole: Environment {
    public let observationSize: Int = 4
    public let actionCount: Int = 2
    public let maxSteps: Int

    // Physics constants
    private let gravity: Double = 9.8
    private let massCart: Double = 1.0
    private let massPole: Double = 0.1
    private let length: Double = 0.5  // half-pole length
    private let forceMag: Double = 10.0
    private let tau: Double = 0.02  // time step

    // Termination thresholds
    private let xThreshold: Double = 2.4
    private let thetaThreshold: Double = 12.0 * .pi / 180.0  // 12 degrees

    // State
    private var x: Double = 0
    private var xDot: Double = 0
    private var theta: Double = 0
    private var thetaDot: Double = 0
    private var stepCount: Int = 0

    public init(maxSteps: Int = 500) {
        self.maxSteps = maxSteps
    }

    public mutating func reset() -> [Double] {
        x = Double.random(in: -0.05...0.05)
        xDot = Double.random(in: -0.05...0.05)
        theta = Double.random(in: -0.05...0.05)
        thetaDot = Double.random(in: -0.05...0.05)
        stepCount = 0
        return observation
    }

    public mutating func step(action: Int) -> StepResult {
        stepCount += 1

        let force = action == 1 ? forceMag : -forceMag
        let totalMass = massCart + massPole
        let poleMassLength = massPole * length

        let cosTheta = cos(theta)
        let sinTheta = sin(theta)

        let temp = (force + poleMassLength * thetaDot * thetaDot * sinTheta) / totalMass
        let thetaAcc = (gravity * sinTheta - cosTheta * temp) /
            (length * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass))
        let xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass

        // Euler integration
        x += tau * xDot
        xDot += tau * xAcc
        theta += tau * thetaDot
        thetaDot += tau * thetaAcc

        let done = abs(x) > xThreshold
            || abs(theta) > thetaThreshold
            || stepCount >= maxSteps

        let reward = done && stepCount < maxSteps ? 0.0 : 1.0

        return StepResult(observation: observation, reward: reward, done: done)
    }

    private var observation: [Double] {
        [x, xDot, theta, thetaDot]
    }
}

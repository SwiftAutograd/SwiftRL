import Foundation

/// A fixed-capacity ring buffer that stores experience tuples for off-policy learning.
public struct ReplayBuffer: Sendable {
    private var buffer: [Experience]
    private var position: Int = 0
    public let capacity: Int
    public var count: Int { min(buffer.count, capacity) }
    public var isFull: Bool { buffer.count >= capacity }

    public init(capacity: Int) {
        self.capacity = capacity
        self.buffer = []
        self.buffer.reserveCapacity(capacity)
    }

    /// Add an experience to the buffer, overwriting the oldest if full.
    public mutating func append(_ experience: Experience) {
        if buffer.count < capacity {
            buffer.append(experience)
        } else {
            buffer[position] = experience
        }
        position = (position + 1) % capacity
    }

    /// Sample a random minibatch of experiences.
    public func sample(batchSize: Int) -> [Experience] {
        precondition(batchSize <= count, "Batch size exceeds buffer count")
        var indices = Set<Int>()
        while indices.count < batchSize {
            indices.insert(Int.random(in: 0..<count))
        }
        return indices.map { buffer[$0] }
    }

    /// Remove all stored experiences.
    public mutating func clear() {
        buffer.removeAll(keepingCapacity: true)
        position = 0
    }
}

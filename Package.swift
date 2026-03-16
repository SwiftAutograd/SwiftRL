// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftRL",
    products: [
        .library(name: "SwiftRL", targets: ["SwiftRL"]),
    ],
    dependencies: [
        .package(url: "https://github.com/SwiftAutograd/SwiftGrad.git", from: "0.2.0"),
    ],
    targets: [
        .target(
            name: "SwiftRL",
            dependencies: ["SwiftGrad"]
        ),
        .testTarget(
            name: "SwiftRLTests",
            dependencies: ["SwiftRL"]
        ),
    ]
)

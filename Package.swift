// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftAudioPrimitives",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(
            name: "SwiftAudioPrimitives",
            targets: ["SwiftAudioPrimitives"]
        ),
    ],
    targets: [
        .target(
            name: "SwiftAudioPrimitives",
            dependencies: [],
            path: "Sources/SwiftAudioPrimitives",
            resources: [
                .copy("Metal/default.metallib"),
                .copy("Metal/Shaders.metal")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ]
        ),
        .testTarget(
            name: "SwiftAudioPrimitivesTests",
            dependencies: ["SwiftAudioPrimitives"],
            path: "Tests/SwiftAudioPrimitivesTests"
        ),
    ]
)

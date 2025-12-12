// swift-tools-version: 5.9
// SwiftRosa - A modular Swift/Metal port of librosa

import PackageDescription

let package = Package(
    name: "SwiftRosa",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        // Individual packages
        .library(name: "SwiftRosaCore", targets: ["SwiftRosaCore"]),
        .library(name: "SwiftRosaAnalysis", targets: ["SwiftRosaAnalysis"]),
        .library(name: "SwiftRosaEffects", targets: ["SwiftRosaEffects"]),
        .library(name: "SwiftRosaStreaming", targets: ["SwiftRosaStreaming"]),
        .library(name: "SwiftRosaML", targets: ["SwiftRosaML"]),
        // Umbrella package (imports all)
        .library(name: "SwiftRosa", targets: ["SwiftRosa"]),
    ],
    targets: [
        // MARK: - Core (Tier 1 - no internal dependencies)
        .target(
            name: "SwiftRosaCore",
            dependencies: [],
            path: "Sources/SwiftRosaCore",
            resources: [
                .process("Metal/Shaders.metal")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ]
        ),

        // MARK: - Tier 2 packages (depend on Core)
        .target(
            name: "SwiftRosaAnalysis",
            dependencies: ["SwiftRosaCore"],
            path: "Sources/SwiftRosaAnalysis",
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "SwiftRosaEffects",
            dependencies: ["SwiftRosaCore"],
            path: "Sources/SwiftRosaEffects",
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "SwiftRosaStreaming",
            dependencies: ["SwiftRosaCore", "SwiftRosaAnalysis", "SwiftRosaEffects"],
            path: "Sources/SwiftRosaStreaming",
            linkerSettings: [
                .linkedFramework("Accelerate"),
            ]
        ),

        // MARK: - Tier 3 (depends on Core + Effects)
        .target(
            name: "SwiftRosaML",
            dependencies: ["SwiftRosaCore", "SwiftRosaEffects"],
            path: "Sources/SwiftRosaML",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreML"),
                .linkedFramework("AVFoundation"),
            ]
        ),

        // MARK: - Umbrella (re-exports all)
        .target(
            name: "SwiftRosa",
            dependencies: [
                "SwiftRosaCore",
                "SwiftRosaAnalysis",
                "SwiftRosaEffects",
                "SwiftRosaStreaming",
                "SwiftRosaML",
            ],
            path: "Sources/SwiftRosa"
        ),

        // MARK: - Tests
        // Core tests need other packages for cross-cutting validation tests
        .testTarget(
            name: "SwiftRosaCoreTests",
            dependencies: ["SwiftRosaCore", "SwiftRosaEffects", "SwiftRosaAnalysis", "SwiftRosaStreaming"],
            path: "Tests/SwiftRosaCoreTests",
            resources: [.copy("ReferenceData")]
        ),
        .testTarget(
            name: "SwiftRosaAnalysisTests",
            dependencies: ["SwiftRosaAnalysis", "SwiftRosaCore"],
            path: "Tests/SwiftRosaAnalysisTests"
        ),
        .testTarget(
            name: "SwiftRosaEffectsTests",
            dependencies: ["SwiftRosaEffects", "SwiftRosaCore"],
            path: "Tests/SwiftRosaEffectsTests"
        ),
        .testTarget(
            name: "SwiftRosaStreamingTests",
            dependencies: ["SwiftRosaStreaming", "SwiftRosaCore"],
            path: "Tests/SwiftRosaStreamingTests"
        ),
        .testTarget(
            name: "SwiftRosaMLTests",
            dependencies: ["SwiftRosaML", "SwiftRosaCore", "SwiftRosaEffects", "SwiftRosaStreaming"],
            path: "Tests/SwiftRosaMLTests"
        ),
    ]
)

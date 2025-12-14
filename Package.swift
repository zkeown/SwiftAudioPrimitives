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
        .library(name: "SwiftRosaNN", targets: ["SwiftRosaNN"]),
        // Umbrella package (imports all)
        .library(name: "SwiftRosa", targets: ["SwiftRosa"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        // MARK: - Build Plugins
        .plugin(
            name: "MetalCompilerPlugin",
            capability: .buildTool(),
            path: "Plugins/MetalCompilerPlugin"
        ),

        // MARK: - Core (Tier 1 - no internal dependencies)
        .target(
            name: "SwiftRosaCore",
            dependencies: [],
            path: "Sources/SwiftRosaCore",
            resources: [
                .process("Resources/PrivacyInfo.xcprivacy"),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ],
            plugins: ["MetalCompilerPlugin"]
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

        // MARK: - Neural Network Primitives (Tier 2 - depends on Core)
        .target(
            name: "SwiftRosaNN",
            dependencies: ["SwiftRosaCore"],
            path: "Sources/SwiftRosaNN",
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ],
            plugins: ["MetalCompilerPlugin"]
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
                "SwiftRosaNN",
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
        .testTarget(
            name: "SwiftRosaNNTests",
            dependencies: ["SwiftRosaNN", "SwiftRosaCore", "SwiftRosaML"],
            path: "Tests/SwiftRosaNNTests",
            resources: [.copy("ReferenceData")]
        ),

        // MARK: - Benchmark Tests (separate target - run with: swift test --filter SwiftRosaBenchmarks)
        .testTarget(
            name: "SwiftRosaBenchmarks",
            dependencies: [
                "SwiftRosaCore",
                "SwiftRosaAnalysis",
                "SwiftRosaEffects",
                "SwiftRosaStreaming",
                "SwiftRosaML",
                "SwiftRosaNN",
            ],
            path: "Tests/SwiftRosaBenchmarks",
            resources: [.copy("ReferenceTimings")]
        ),
    ]
)

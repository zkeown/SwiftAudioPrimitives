import PackagePlugin
import Foundation

@main
struct MetalCompilerPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
        guard let target = target as? SourceModuleTarget else { return [] }

        // Find all .metal files in the target
        let metalFiles = target.sourceFiles.filter { $0.path.extension == "metal" }
        guard !metalFiles.isEmpty else { return [] }

        let airPath = context.pluginWorkDirectory.appending("Shaders.air")
        let metallibPath = context.pluginWorkDirectory.appending("default.metallib")

        // Build arguments for metal compiler
        var metalArgs = ["-c"]
        for file in metalFiles {
            metalArgs.append(file.path.string)
        }
        metalArgs.append("-o")
        metalArgs.append(airPath.string)

        return [
            // Step 1: Compile .metal to .air (Metal IR)
            .buildCommand(
                displayName: "Compile Metal Shaders",
                executable: Path("/usr/bin/xcrun"),
                arguments: ["metal"] + metalArgs,
                inputFiles: metalFiles.map { $0.path },
                outputFiles: [airPath]
            ),
            // Step 2: Link .air to .metallib
            .buildCommand(
                displayName: "Link Metal Library",
                executable: Path("/usr/bin/xcrun"),
                arguments: [
                    "metallib",
                    airPath.string,
                    "-o",
                    metallibPath.string
                ],
                inputFiles: [airPath],
                outputFiles: [metallibPath]
            )
        ]
    }
}

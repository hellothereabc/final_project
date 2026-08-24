import CoreML
import SwiftUI
import UIKit

/// On-device half of the phone-vs-laptop comparison.
///
/// runtime_bench/bench.py measures the same .mlpackage on the Mac across
/// CoreML compute units; nothing on the Mac can measure the phone's Neural
/// Engine, or what the phone's thermal state does to a sustained run. This
/// screen collects that half: median latency per compute-unit setting over the
/// bundled samples, plus the detections themselves so the two sides can be
/// compared on identical inputs rather than on similar-looking ones.
///
/// The detections matter as much as the timings. The Mac letterboxes with grey
/// 114 the way ultralytics does at training time; the app asks Vision for
/// `.scaleFit`, which pads with black. If the boxes agree, that difference is
/// cosmetic. If they do not, the app is not running the model that was measured.
///
/// `.cpuAndGPU` / `.all` are behind a toggle because both have crashed on this
/// model before - a `MLIR pass manager failed` assertion inside
/// `handler.perform`, which traps rather than throwing, so the app dies rather
/// than reporting an error. Off by default; turn it on knowing the run may not
/// come back.
struct BenchmarkView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var running = false
    @State private var progress = ""
    @State private var rows: [Row] = []
    @State private var report = ""
    @State private var includeGPU = false
    @State private var repeats = 10

    private let samples = ["sample1", "sample2", "sample3"]

    struct Row: Identifiable {
        let id = UUID()
        let units: String
        let medianMs: Double
        let p90Ms: Double
        let loadMs: Double
        let detections: Int
        let failed: String?
    }

    var body: some View {
        NavigationStack {
            List {
                Section("run") {
                    Stepper("Repeats per image: \(repeats)", value: $repeats, in: 1...50)
                    Toggle("Include GPU paths (may crash)", isOn: $includeGPU)
                    Button(running ? "Running…" : "Start") { start() }
                        .disabled(running)
                    if !progress.isEmpty {
                        Text(progress).font(.caption).foregroundStyle(.secondary)
                    }
                }

                if !rows.isEmpty {
                    Section("median ms per 640x640 frame") {
                        ForEach(rows) { row in
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(row.units).font(.body.monospaced())
                                    Spacer()
                                    if let failed = row.failed {
                                        Text(failed).foregroundStyle(.red).font(.caption)
                                    } else {
                                        Text(String(format: "%.1f ms · %.0f fps",
                                                    row.medianMs, 1000 / row.medianMs))
                                    }
                                }
                                if row.failed == nil {
                                    Text(String(format: "p90 %.1f ms · load %.0f ms · %d detections",
                                                row.p90Ms, row.loadMs, row.detections))
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }

                if !report.isEmpty {
                    Section("report") {
                        Button {
                            UIPasteboard.general.string = report
                            progress = "JSON copied — paste it into runtime_bench/"
                        } label: {
                            Label("Copy JSON", systemImage: "doc.on.doc")
                        }
                        Text(report)
                            .font(.system(size: 9).monospaced())
                            .lineLimit(12)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Benchmark")
            .toolbar { ToolbarItem(placement: .cancellationAction) {
                Button("Done") { dismiss() }.disabled(running)
            } }
        }
    }

    // MARK: - work

    private func start() {
        running = true
        rows = []
        report = ""

        var configs: [(String, MLComputeUnits)] = [
            ("cpuOnly", .cpuOnly),
            ("cpuAndNeuralEngine", .cpuAndNeuralEngine),
        ]
        if includeGPU {
            configs.append(("cpuAndGPU", .cpuAndGPU))
            configs.append(("all", .all))
        }

        let images = samples.compactMap { name -> (String, UIImage)? in
            guard let image = Self.bundleImage(name) else { return nil }
            return (name, image)
        }
        let passes = repeats

        Task.detached(priority: .userInitiated) {
            var payload: [[String: Any]] = []

            for (name, units) in configs {
                await MainActor.run { progress = "\(name): loading…" }

                let loadStarted = Date()
                let detector: Detector
                do {
                    detector = try Detector(computeUnits: units)
                } catch {
                    await MainActor.run {
                        rows.append(Row(units: name, medianMs: 0, p90Ms: 0, loadMs: 0,
                                        detections: 0, failed: error.localizedDescription))
                    }
                    continue
                }
                let loadMs = Date().timeIntervalSince(loadStarted) * 1000

                // First frames pay for lazy graph compilation and, on the ANE,
                // for the model being loaded onto it. Not part of the number.
                for (_, image) in images.prefix(1) {
                    _ = try? detector.detect(image)
                }

                var timings: [Double] = []
                var lastDetections: [String: [Detection]] = [:]
                for pass in 0..<passes {
                    for (imageName, image) in images {
                        let started = Date()
                        let found = (try? detector.detect(image)) ?? []
                        timings.append(Date().timeIntervalSince(started) * 1000)
                        if pass == passes - 1 { lastDetections[imageName] = found }
                    }
                    await MainActor.run {
                        progress = "\(name): pass \(pass + 1)/\(passes)"
                    }
                }

                let sorted = timings.sorted()
                let median = sorted[sorted.count / 2]
                let p90 = sorted[min(sorted.count - 1, Int(0.9 * Double(sorted.count)))]
                let total = lastDetections.values.reduce(0) { $0 + $1.count }

                await MainActor.run {
                    rows.append(Row(units: name, medianMs: median, p90Ms: p90,
                                    loadMs: loadMs, detections: total, failed: nil))
                }

                payload.append([
                    "computeUnits": name,
                    "median_ms": median,
                    "p90_ms": p90,
                    "mean_ms": timings.reduce(0, +) / Double(timings.count),
                    "load_ms": loadMs,
                    "frames": timings.count,
                    "thermal_state": Self.thermalState(),
                    "images": images.map { imageName, image in
                        [
                            "name": imageName,
                            "width": Int(image.size.width),
                            "height": Int(image.size.height),
                            // Vision's normalised rects, origin bottom-left -
                            // runtime_bench/compare_device.py flips them.
                            "detections": (lastDetections[imageName] ?? []).map {
                                [
                                    "label": $0.label,
                                    "conf": Double($0.confidence),
                                    "x": $0.rect.origin.x, "y": $0.rect.origin.y,
                                    "w": $0.rect.width, "h": $0.rect.height,
                                ] as [String: Any]
                            },
                        ] as [String: Any]
                    },
                ])
            }

            let document: [String: Any] = [
                "device": Self.machine(),
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "repeats": passes,
                "configs": payload,
            ]
            let json = (try? JSONSerialization.data(withJSONObject: document,
                                                    options: [.prettyPrinted, .sortedKeys]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"

            await MainActor.run {
                report = json
                progress = "done — copy the JSON"
                running = false
            }
        }
    }

    private static func bundleImage(_ name: String) -> UIImage? {
        if let asset = UIImage(named: name) { return asset }
        guard let url = Bundle.main.url(forResource: name, withExtension: "jpg"),
              let data = try? Data(contentsOf: url)
        else { return nil }
        return UIImage(data: data)
    }

    /// "iPhone15,2" and the like: which chip ran this matters more than the
    /// marketing name when the number is a latency.
    ///
    /// Read through a Mirror rather than by taking a pointer into `utsname`:
    /// `withUnsafePointer(to: &info.machine)` needs exclusive access to the
    /// field while `MemoryLayout.size(ofValue: info.machine)` is still reading
    /// it, which the compiler rejects as overlapping access.
    private nonisolated static func machine() -> String {
        var info = utsname()
        uname(&info)
        return Mirror(reflecting: info.machine).children.reduce(into: "") { name, field in
            guard let byte = field.value as? CChar, byte != 0 else { return }
            name.append(Character(UnicodeScalar(UInt8(byte))))
        }
    }

    /// A phone that has heated up quietly halves its own clocks, so a latency
    /// without this next to it is not reproducible.
    private nonisolated static func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

#Preview { BenchmarkView() }

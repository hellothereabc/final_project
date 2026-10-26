import CoreML
import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var detector: Detector?
    @State private var status = "Loading model…"
    @State private var image: UIImage?
    @State private var detections: [Detection] = []
    @State private var busy = false
    @State private var pickerItem: PhotosPickerItem?
    @State private var showCamera = false
    @State private var showBenchmark = false
    /// Display-time cut-off, above the model's own baked-in 0.25. Fixed rather
    /// than user-adjustable: on in-domain frames moving it changes almost nothing,
    /// and the errors that matter (ripe fruit labelled half) are confident, so a
    /// threshold does not touch them. `-conf` still overrides it for sweeps.
    @State private var minConfidence: Float = 0.45
    @State private var lastMs: Double = 0
    /// Which bundled model runs, and on what. Persisted so the choice survives a
    /// relaunch; the size ladder is the point of having several models on the
    /// phone at all, so switching is a menu on the main screen rather than a
    /// setting buried elsewhere.
    @AppStorage("modelName") private var modelName = ""
    @AppStorage("useNeuralEngine") private var useNeuralEngine = false

    private let samples = ["sample1", "sample2", "sample3"]

    var body: some View {
        // Deliberately not gated behind a loading branch: a view that renders
        // nothing until the model arrives is indistinguishable from a crash,
        // which is exactly the black screen seen on device before.
        VStack(spacing: 12) {
            header
            preview
            counters
            controls
        }
        .padding()
        .task { await loadDetector() }
        .onChange(of: pickerItem) { _, item in Task { await loadPicked(item) } }
        .onChange(of: modelName) { _, _ in Task { await loadDetector() } }
        .onChange(of: useNeuralEngine) { _, _ in Task { await loadDetector() } }
        .fullScreenCover(isPresented: $showCamera) {
            if let detector { CameraScreen(detector: detector) }
        }
        .sheet(isPresented: $showBenchmark) { BenchmarkView(modelName: currentModelName) }
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("Tomato Ripeness").font(.headline)
            modelMenu
            Text(status)
                .font(.caption)
                .foregroundStyle(detector == nil ? .red : .secondary)
                .multilineTextAlignment(.center)
        }
    }

    /// Model and compute-unit choice. Every entry is a compiled model in the
    /// bundle (see Detector.availableModels); the Neural Engine toggle is the
    /// setting BenchmarkView measured as 2-4x faster than the CPU default.
    private var modelMenu: some View {
        HStack(spacing: 8) {
            Menu {
                Picker("Model", selection: $modelName) {
                    ForEach(Detector.availableModels(), id: \.self) { name in
                        Text(name).tag(name)
                    }
                }
            } label: {
                Label(currentModelName, systemImage: "cpu")
                    .font(.subheadline)
            }
            .buttonStyle(.bordered)
            .disabled(busy)

            Toggle("Neural Engine", isOn: $useNeuralEngine)
                .toggleStyle(.button)
                .font(.subheadline)
                .disabled(busy)
        }
    }


    @ViewBuilder
    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(.quaternary)

            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .overlay { BoxOverlay(detections: visible, imageSize: image.size) }
            } else {
                Text("Pick a sample or a photo").foregroundStyle(.secondary)
            }

            if busy { ProgressView().controlSize(.large) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var counters: some View {
        StageCounters(detections: visible)
    }

    private var visible: [Detection] {
        detections.filter { $0.confidence >= minConfidence }
    }

    private func refreshStatus() {
        guard !detections.isEmpty || lastMs > 0 else { return }
        status = String(format: "%d tomatoes · %.0f ms", visible.count, lastMs)
    }

    private var controls: some View {
        HStack(spacing: 8) {
            ForEach(samples, id: \.self) { name in
                Button(name.replacingOccurrences(of: "sample", with: "#")) {
                    if let img = Self.bundleImage(name) { run(on: img) }
                    else { status = "Sample \(name) missing from the bundle" }
                }
                .buttonStyle(.bordered)
            }
            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Photo", systemImage: "photo").labelStyle(.iconOnly)
            }
            .buttonStyle(.bordered)

            // Phone half of runtime_bench: the laptop cannot time this model
            // on the Neural Engine, and no dataset number says what the export
            // costs once Vision does the letterboxing.
            Button {
                showBenchmark = true
            } label: {
                Label("Benchmark", systemImage: "speedometer").labelStyle(.iconOnly)
            }
            .buttonStyle(.bordered)

            Button {
                showCamera = true
            } label: {
                Label("Camera", systemImage: "camera")
            }
            .buttonStyle(.borderedProminent)
        }
        .disabled(detector == nil || busy)
    }

    // MARK: - work

    /// The samples are loose files in the bundle, not asset-catalog entries, and
    /// `UIImage(named:)` does not reliably find those — it returned nil for every
    /// sample even though the files were in the built .app. Read them by URL.
    private static func bundleImage(_ name: String) -> UIImage? {
        if let asset = UIImage(named: name) { return asset }
        guard let url = Bundle.main.url(forResource: name, withExtension: "jpg"),
              let data = try? Data(contentsOf: url)
        else { return nil }
        return UIImage(data: data)
    }

    /// A stored choice that no longer exists in the bundle (a model removed
    /// from the build) falls back to the default instead of a stale name.
    private var currentModelName: String {
        Detector.availableModels().contains(modelName) ? modelName : Detector.defaultModelName
    }

    private func loadDetector() async {
        // Off the main actor: compiling a model can take seconds on first launch
        // and would otherwise block the first frame from ever being drawn.
        let name = currentModelName
        let units: MLComputeUnits = useNeuralEngine ? .cpuAndNeuralEngine : .cpuOnly
        detector = nil
        detections = []
        status = "Loading \(name)…"
        let started = Date()
        let result = await Task.detached(priority: .userInitiated) {
            Result { try Detector(modelName: name, computeUnits: units) }
        }.value

        switch result {
        case .success(let loaded):
            detector = loaded
            let loadMs = Date().timeIntervalSince(started) * 1000
            status = String(format: "%@ ready on %@ · loaded in %.0f ms",
                            loaded.modelName, useNeuralEngine ? "Neural Engine" : "CPU", loadMs)

            // Lets an automated run exercise the whole detect path without a tap:
            // build_run_sim(launchArgs: ["-autorun-sample1"]).
            let args = ProcessInfo.processInfo.arguments
            // -conf lets an automated run sweep the threshold without touching the UI.
            if let i = args.firstIndex(of: "-conf"), i + 1 < args.count,
               let parsed = Float(args[i + 1]) {
                minConfidence = min(max(parsed, 0.25), 0.9)
            }
            if let i = args.firstIndex(of: "-model"), i + 1 < args.count,
               Detector.availableModels().contains(args[i + 1]), args[i + 1] != modelName {
                modelName = args[i + 1]   // triggers a reload through onChange
                return
            }
            if args.contains("-autorun-sample1"), let sample = Self.bundleImage("sample1") {
                run(on: sample)
            }
            if args.contains("-autorun-camera") { showCamera = true }
            if args.contains("-autorun-benchmark") { showBenchmark = true }
        case .failure(let error):
            status = error.localizedDescription
        }
    }

    private func loadPicked(_ item: PhotosPickerItem?) async {
        guard let item,
              let data = try? await item.loadTransferable(type: Data.self),
              let picked = UIImage(data: data)
        else { return }
        run(on: picked)
    }

    private func run(on picked: UIImage) {
        guard let detector else { return }
        image = picked
        detections = []
        busy = true
        status = "Detecting…"

        let started = Date()
        Task.detached(priority: .userInitiated) {
            print(String(format: "[TR] detect start, image %.0fx%.0f", picked.size.width, picked.size.height))
            let outcome = Result { try detector.detect(picked) }
            let elapsedMs = Date().timeIntervalSince(started) * 1000
            print(String(format: "[TR] detect finished in %.0f ms", elapsedMs))

            await MainActor.run {
                busy = false
                switch outcome {
                case .success(let found):
                    detections = found
                    lastMs = elapsedMs
                    refreshStatus()
                case .failure(let error):
                    status = error.localizedDescription
                }
            }
        }

        // Without this the spinner can run forever with nothing on screen to say
        // why — the same silent-failure mode the model loading was hardened against.
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(20))
            if busy {
                status = "Still running after 20 s — inference is stuck, not slow"
                print("[TR] watchdog: still busy after 20s")
            }
        }
    }
}

#Preview { ContentView() }

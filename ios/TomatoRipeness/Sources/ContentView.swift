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
        .fullScreenCover(isPresented: $showCamera) {
            if let detector { CameraScreen(detector: detector) }
        }
        .sheet(isPresented: $showBenchmark) { BenchmarkView() }
    }

    private var header: some View {
        VStack(spacing: 2) {
            Text("Tomato Ripeness").font(.headline)
            Text(status)
                .font(.caption)
                .foregroundStyle(detector == nil ? .red : .secondary)
                .multilineTextAlignment(.center)
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

    private func loadDetector() async {
        // Off the main actor: compiling a model can take seconds on first launch
        // and would otherwise block the first frame from ever being drawn.
        let result = await Task.detached(priority: .userInitiated) {
            Result { try Detector() }
        }.value

        switch result {
        case .success(let loaded):
            detector = loaded
            let names = loaded.labels.isEmpty ? "labels from model metadata unavailable"
                                              : loaded.labels.joined(separator: ", ")
            status = "Model ready · \(names)"

            // Lets an automated run exercise the whole detect path without a tap:
            // build_run_sim(launchArgs: ["-autorun-sample1"]).
            let args = ProcessInfo.processInfo.arguments
            // -conf lets an automated run sweep the threshold without touching the UI.
            if let i = args.firstIndex(of: "-conf"), i + 1 < args.count,
               let parsed = Float(args[i + 1]) {
                minConfidence = min(max(parsed, 0.25), 0.9)
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

import CoreML
import UIKit
import Vision

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    /// Normalised, Vision coordinates: origin bottom-left.
    let rect: CGRect
}

enum DetectorError: LocalizedError {
    case modelMissing
    case badImage

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "No .mlmodelc in the app bundle — the model was not compiled into this build."
        case .badImage:
            return "Could not read that image."
        }
    }
}

/// Wraps the CoreML detector exported by ../../export_coreml.py.
///
/// That export bakes NMS into the model, so Vision hands back finished
/// `VNRecognizedObjectObservation`s and there is no box decoding here.
final class Detector {
    private let model: VNCoreMLModel

    /// Class names in the order the model declares them, for stable UI ordering.
    let labels: [String]

    /// Which compute units this instance was built with, for the benchmark screen.
    let computeUnits: MLComputeUnits

    /// The compiled model's name, as `availableModels()` lists it.
    let modelName: String

    /// Every model compiled into the bundle, by name: the Laboro size ladder
    /// `YOLO11n`..`YOLO11x` copied in by sync_models.sh, smallest first, then
    /// anything else alphabetically.
    static func availableModels() -> [String] {
        guard let root = Bundle.main.resourceURL,
              let items = try? FileManager.default.contentsOfDirectory(
                  at: root, includingPropertiesForKeys: nil)
        else { return [] }
        let names = items.filter { $0.pathExtension == "mlmodelc" }
            .map { $0.deletingPathExtension().lastPathComponent }
        let ladder = ["YOLO11n", "YOLO11s", "YOLO11m", "YOLO11l", "YOLO11x"]
        return ladder.filter(names.contains) + names.filter { !ladder.contains($0) }.sorted()
    }

    /// The model used when nothing is chosen: the smallest, so the CPU default
    /// still keeps the camera fluid; the picker is there for the rest.
    static var defaultModelName: String {
        availableModels().first ?? ""
    }

    /// `computeUnits` defaults to the only setting known to work on both targets;
    /// BenchmarkView passes the others deliberately, one detector each, because
    /// the choice is fixed at load time and cannot be changed per request.
    /// `modelName` picks one of `availableModels()`; nil means the default.
    init(modelName: String? = nil, computeUnits: MLComputeUnits = .cpuOnly) throws {
        let name = modelName ?? Self.defaultModelName
        guard let url = Self.findCompiledModel(named: name) else { throw DetectorError.modelMissing }
        self.modelName = name

        let config = MLModelConfiguration()
        // The GPU path is unusable for this model on both targets, in two different
        // shapes, so it is excluded outright rather than per-environment:
        //   simulator — `Espresso exception: "Invalid state": MpsGraph backend
        //               validation on incompatible OS`, then inference hangs forever;
        //   device    — `MPSGraphExecutable.mm: failed assertion 'MLIR pass manager
        //               failed'`, which traps as EXC_BREAKPOINT inside handler.perform.
        // CPU runs a 640x640 frame in ~0.1 s, so nothing is lost by staying off it.
        // BenchmarkView is that retry: it builds one detector per setting and
        // times them, so the default here can be revisited with numbers.
        config.computeUnits = computeUnits
        self.computeUnits = computeUnits

        let started = Date()
        let mlModel = try MLModel(contentsOf: url, configuration: config)
        model = try VNCoreMLModel(for: mlModel)
        labels = Self.readLabels(from: mlModel)
        print(String(format: "[TR] %@ loaded in %.0f ms, computeUnits=%d, labels=%d",
                     name, Date().timeIntervalSince(started) * 1000,
                     config.computeUnits.rawValue, labels.count))
    }

    func detect(_ image: UIImage, confidence: Float = 0.25) throws -> [Detection] {
        guard let cgImage = image.cgImage else { throw DetectorError.badImage }

        let request = VNCoreMLRequest(model: model)
        // Matches the letterboxing ultralytics uses at training time.
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cgImage: cgImage,
                                            orientation: image.cgImageOrientation)
        print("[TR] handler.perform entering")
        try handler.perform([request])
        print(String(format: "[TR] handler.perform returned, %d raw results", request.results?.count ?? -1))

        return Self.decode(request, confidence: confidence)
    }

    /// Camera path: Vision reads the capture buffer directly, with no UIImage or
    /// CGImage copy per frame.
    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation = .right,
                confidence: Float = 0.25) throws -> [Detection] {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation)
        try handler.perform([request])

        return Self.decode(request, confidence: confidence)
    }

    private static func decode(_ request: VNCoreMLRequest, confidence: Float) -> [Detection] {
        let observations = request.results as? [VNRecognizedObjectObservation] ?? []
        return observations.compactMap { observation in
            guard let top = observation.labels.first, top.confidence >= confidence else { return nil }
            return Detection(label: top.identifier,
                             confidence: top.confidence,
                             rect: observation.boundingBox)
        }
    }

    /// Counts per class, in model-declared order, including classes with zero hits.
    func counts(for detections: [Detection]) -> [(label: String, count: Int)] {
        let tally = Dictionary(grouping: detections, by: \.label).mapValues(\.count)
        let ordered = labels.isEmpty ? tally.keys.sorted() : labels
        return ordered.map { ($0, tally[$0] ?? 0) }
    }

    // MARK: - bundle plumbing

    /// Xcode compiles every .mlpackage in Compile Sources to `<name>.mlmodelc`
    /// at the bundle root; a missing name falls back to the first one found, so
    /// a build with a single renamed export still runs.
    private static func findCompiledModel(named name: String) -> URL? {
        guard let root = Bundle.main.resourceURL,
              let items = try? FileManager.default.contentsOfDirectory(
                  at: root, includingPropertiesForKeys: nil)
        else { return nil }
        let compiled = items.filter { $0.pathExtension == "mlmodelc" }
        return compiled.first { $0.deletingPathExtension().lastPathComponent == name }
            ?? compiled.first
    }

    /// ultralytics writes the class list into the model's user-defined metadata.
    private static func readLabels(from model: MLModel) -> [String] {
        let userDefined = model.modelDescription
            .metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]

        guard let raw = userDefined?["names"] ?? userDefined?["classes"] else { return [] }

        // Either a python dict "{0: 'b_green', ...}" or a newline/comma list.
        let separators = CharacterSet(charactersIn: ",\n")
        return raw
            .components(separatedBy: separators)
            .map { part -> String in
                let value = part.contains(":") ? part.components(separatedBy: ":").last ?? part : part
                return value.trimmingCharacters(in: CharacterSet(charactersIn: " '\"{}[]"))
            }
            .filter { !$0.isEmpty }
    }
}

private extension UIImage {
    var cgImageOrientation: CGImagePropertyOrientation {
        CGImagePropertyOrientation(rawValue: UInt32(imageOrientation.exifValue)) ?? .up
    }
}

private extension UIImage.Orientation {
    var exifValue: Int {
        switch self {
        case .up: return 1
        case .down: return 3
        case .left: return 8
        case .right: return 6
        case .upMirrored: return 2
        case .downMirrored: return 4
        case .leftMirrored: return 5
        case .rightMirrored: return 7
        @unknown default: return 1
        }
    }
}

import AVFoundation
import SwiftUI

struct CameraScreen: View {
    @StateObject private var camera: CameraController
    @Environment(\.dismiss) private var dismiss

    /// Same fixed cut-off as the photo screen, above the model's baked-in 0.25.
    private let minConfidence: Float = 0.45

    init(detector: Detector) {
        _camera = StateObject(wrappedValue: CameraController(detector: detector))
    }

    private var visible: [Detection] {
        camera.detections.filter { $0.confidence >= minConfidence }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraPreview(session: camera.session)
                .overlay {
                    BoxOverlay(detections: visible, imageSize: camera.videoSize)
                }
                .ignoresSafeArea()

            VStack {
                header
                Spacer()
                StageCounters(detections: visible, compact: true)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
            }
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }

    private var header: some View {
        HStack {
            Button("Close") { dismiss() }
                .buttonStyle(.borderedProminent)
                .tint(.black.opacity(0.5))

            Spacer()

            Text(camera.problem ?? String(format: "%d found · %.0f ms",
                                          visible.count, camera.inferenceMs))
                .font(.caption.monospacedDigit())
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.black.opacity(0.5), in: Capsule())
                .foregroundStyle(camera.problem == nil ? .white : .orange)
        }
        .padding(.horizontal)
    }
}

/// Hosts the capture preview. `.resizeAspect` letterboxes the frame instead of
/// cropping it, which keeps the box geometry identical to the photo screen and
/// lets both reuse BoxOverlay unchanged.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            // Portrait. Must match the .right orientation Vision is given, or the
            // boxes land rotated relative to what is on screen.
            previewLayer.connection?.videoRotationAngle = 90
        }
    }
}

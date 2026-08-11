import AVFoundation
import SwiftUI
import Vision

/// Live detection over the camera feed.
///
/// Frames are processed synchronously on one serial queue and the output drops
/// late frames, so a slow inference skips frames rather than building a backlog
/// that would draw boxes belonging to an already-stale frame.
final class CameraController: NSObject, ObservableObject {
    @Published private(set) var detections: [Detection] = []
    @Published private(set) var inferenceMs: Double = 0
    /// Displayed frame size (buffer axes swapped for the portrait preview).
    @Published private(set) var videoSize: CGSize = .zero
    @Published private(set) var problem: String?

    let session = AVCaptureSession()

    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "TomatoRipeness.frames")
    private let detector: Detector

    init(detector: Detector) {
        self.detector = detector
        super.init()
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self else { return }
            guard granted else {
                DispatchQueue.main.async {
                    self.problem = "Camera access denied — enable it in Settings."
                }
                return
            }
            self.queue.async { self.configureAndRun() }
        }
    }

    func stop() {
        queue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configureAndRun() {
        guard session.inputs.isEmpty else {
            if !session.isRunning { session.startRunning() }
            return
        }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input)
        else {
            session.commitConfiguration()
            DispatchQueue.main.async {
                // The simulator has no camera at all, so this is the expected
                // message there rather than a fault.
                self.problem = "No usable back camera on this device."
            }
            return
        }
        session.addInput(input)

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }

        session.commitConfiguration()
        session.startRunning()
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let started = Date()
        let found = (try? detector.detect(pixelBuffer: pixels)) ?? []
        let elapsedMs = Date().timeIntervalSince(started) * 1000

        // Vision reads the buffer with .right, which turns the landscape capture
        // into the portrait frame on screen — hence the swapped axes here.
        let displayed = CGSize(width: CVPixelBufferGetHeight(pixels),
                               height: CVPixelBufferGetWidth(pixels))

        DispatchQueue.main.async {
            self.detections = found
            self.inferenceMs = elapsedMs
            self.videoSize = displayed
        }
    }
}

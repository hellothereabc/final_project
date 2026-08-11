import SwiftUI

/// Draws detection boxes over an aspect-fit image.
///
/// Vision returns normalised rects in the *image* frame with the origin at the
/// bottom-left; SwiftUI draws top-left in the *view* frame. The image is letterboxed
/// inside the view, so boxes have to be mapped through that letterbox or they drift
/// on any photo whose aspect ratio differs from the view's.
struct BoxOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let fitted = Self.aspectFit(image: imageSize, in: geo.size)

            ForEach(detections) { detection in
                let box = Self.place(detection.rect, in: fitted)

                ZStack(alignment: .topLeading) {
                    Rectangle()
                        .stroke(color(for: detection.label), lineWidth: 2)
                        .frame(width: box.width, height: box.height)

                    Text(detection.label.shortRipenessName)
                        .font(.system(size: 9, weight: .semibold))
                        .padding(.horizontal, 3)
                        .padding(.vertical, 1)
                        .background(color(for: detection.label))
                        .foregroundStyle(.white)
                        .offset(y: -13)
                }
                .position(x: box.midX, y: box.midY)
            }
        }
    }

    private static func aspectFit(image: CGSize, in container: CGSize) -> CGRect {
        guard image.width > 0, image.height > 0 else { return .zero }
        let scale = min(container.width / image.width, container.height / image.height)
        let size = CGSize(width: image.width * scale, height: image.height * scale)
        return CGRect(x: (container.width - size.width) / 2,
                      y: (container.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    private static func place(_ normalised: CGRect, in frame: CGRect) -> CGRect {
        CGRect(x: frame.minX + normalised.minX * frame.width,
               // flip: Vision's y grows upward
               y: frame.minY + (1 - normalised.maxY) * frame.height,
               width: normalised.width * frame.width,
               height: normalised.height * frame.height)
    }

    private func color(for label: String) -> Color {
        switch label.ripenessStage {
        case .green: return Color(red: 0.16, green: 0.62, blue: 0.56)
        case .half: return Color(red: 0.96, green: 0.64, blue: 0.38)
        case .fully: return Color(red: 0.90, green: 0.22, blue: 0.27)
        case .unknown: return .blue
        }
    }
}

enum RipenessStage {
    case green, half, fully, unknown
}

struct StageTally: Identifiable {
    let stage: String
    let count: Int
    let color: Color
    var id: String { stage }
}

/// Per-stage counts in a fixed ripening order, shared by the photo and camera
/// screens so both always show the same three tiles even at zero.
func stageTally(_ detections: [Detection]) -> [StageTally] {
    let byStage = Dictionary(grouping: detections, by: \.label.ripenessStage).mapValues(\.count)
    return [
        StageTally(stage: "green", count: byStage[.green] ?? 0,
                   color: Color(red: 0.16, green: 0.62, blue: 0.56)),
        StageTally(stage: "half", count: byStage[.half] ?? 0,
                   color: Color(red: 0.96, green: 0.64, blue: 0.38)),
        StageTally(stage: "ripe", count: byStage[.fully] ?? 0,
                   color: Color(red: 0.90, green: 0.22, blue: 0.27)),
    ]
}

/// The three counter tiles, reused by both screens.
struct StageCounters: View {
    let detections: [Detection]
    var compact = false

    var body: some View {
        HStack(spacing: 10) {
            ForEach(stageTally(detections)) { entry in
                VStack(spacing: 2) {
                    Text("\(entry.count)").font(compact ? .title3.bold() : .title2.bold())
                    Text(entry.stage).font(.caption2)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, compact ? 5 : 8)
                .background(entry.color.opacity(compact ? 0.85 : 0.15),
                            in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(compact ? .white : .primary)
            }
        }
    }
}

extension String {
    /// Class names carry a size prefix (`b_` normal, `l_` cherry) that the UI does
    /// not distinguish; collapse to the ripeness stage.
    var ripenessStage: RipenessStage {
        let name = lowercased()
        if name.contains("half") { return .half }        // before "ripened"
        if name.contains("green") || name.contains("unripe") { return .green }
        if name.contains("full") || name.contains("ripe") { return .fully }
        return .unknown
    }

    var shortRipenessName: String {
        switch ripenessStage {
        case .green: return "green"
        case .half: return "half"
        case .fully: return "ripe"
        case .unknown: return self
        }
    }
}

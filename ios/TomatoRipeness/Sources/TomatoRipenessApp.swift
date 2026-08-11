import SwiftUI

@main
struct TomatoRipenessApp: App {
    init() {
        // stdout is block-buffered when the app is not attached to a terminal, so
        // diagnostic prints would sit in the buffer instead of reaching the
        // console captured over the device connection.
        setvbuf(stdout, nil, _IONBF, 0)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

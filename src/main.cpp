#include "app.h"
#include <iostream>

/**
 * Program entry point.
 *
 * The main function is intentionally thin:
 *  - Delegates all argument parsing to app::parse_args
 *  - Delegates the actual solver / application logic to app::run
 *  - Catches std::exception and prints a short error message to stderr
 *    so that failures are reported cleanly (with exit code 1).
 */
int main(int argc, char **argv) {
    try {
        // Parse command-line arguments into an AppConfig object.
        auto cfg = app::parse_args(argc, argv);

        // Run the application with the parsed configuration.
        // app::run returns the desired process exit code.
        return app::run(cfg);
    } catch (const std::exception &ex) {
        // Any unexpected error is reported here in a uniform way.
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}

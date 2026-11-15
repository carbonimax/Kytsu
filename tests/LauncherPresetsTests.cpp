#include "presets.h"

#include <QByteArray>
#include <QProcessEnvironment>
#include <QStringList>

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LauncherPresetsTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	const QByteArray json = R"({
		"_comment": "unknown keys are ignored",
		"version": 1,
		"global": {"env": {"KYTY_A": "global", "KYTY_B": "global", "KYTY_USER": "preset"},
		           "args": ["--flag-g", "--value", "1"]},
		"titles": {"TITLE_A": {"env": {"KYTY_B": "title"}, "args": ["--value", "2", "--flag-t"]},
		           "TITLE_B": {"env": {"KYTY_C": "other"}}}
	})";

	Presets::Preset preset;
	QString         error;
	Check(Presets::Parse(json, "TITLE_A", preset, error) && error.isEmpty(), "valid file parses");
	Check(preset.env.value("KYTY_B") == "title", "title env wins over global");
	Check(preset.env.value("KYTY_A") == "global" && !preset.env.contains("KYTY_C"),
	      "only global and the chosen title are merged");
	Check(preset.args == QStringList {"--value", "2", "--flag-t", "--flag-g"},
	      "title option replaces the global one with its value");

	QProcessEnvironment env;
	env.insert("KYTY_USER", "user");
	QStringList args {"--value", "9", "--flag-t", "--game", "eboot.bin"};
	const auto  applied = Presets::Apply(preset, env, args);
	Check(env.value("KYTY_USER") == "user", "process environment wins");
	Check(env.value("KYTY_A") == "global" && env.value("KYTY_B") == "title", "preset env applied");
	Check(args == QStringList {"--value", "9", "--flag-t", "--game", "eboot.bin", "--flag-g"},
	      "dialog options win and a skipped option drops its value");
	Check(applied == QStringList {"KYTY_A=global", "KYTY_B=title", "--flag-g"}, "applied summary");

	Check(Presets::Parse(json, "", preset, error) && preset.env.size() == 3 &&
	          preset.args == QStringList {"--flag-g", "--value", "1"},
	      "no title id uses the global entry only");
	Check(Presets::Parse(R"({"version": 1})", "TITLE_A", preset, error) && preset.env.isEmpty() &&
	          preset.args.isEmpty(),
	      "empty sections are allowed");

	const char* bad[] = {
	    "",
	    "not json",
	    "[]",
	    R"({"version": 2})",
	    R"({"version": "1"})",
	    R"({"version": 1, "global": []})",
	    R"({"version": 1, "global": {"env": {"KYTY_A": 1}}})",
	    R"({"version": 1, "global": {"env": {"BAD KEY": "1"}}})",
	    R"({"version": 1, "global": {"args": ["--x", 2]}})",
	    R"({"version": 1, "global": {"args": ["value-first"]}})",
	    R"({"version": 1, "titles": []})",
	    R"({"version": 1, "global": {"env": {"KYTY_A": "1"}}, "titles": {"TITLE_B": {"env": []}}})",
	};
	for (const auto* text: bad) {
		const bool ok = Presets::Parse(text, "TITLE_A", preset, error);
		Check(!ok && !error.isEmpty() && preset.env.isEmpty() && preset.args.isEmpty(),
		      "malformed input is rejected as a whole");
	}

	std::puts("LauncherPresetsTests: all cases passed");
	return 0;
}

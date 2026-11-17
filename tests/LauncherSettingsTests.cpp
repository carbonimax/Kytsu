#include "configuration.h"

#include <QCoreApplication>
#include <QSettings>
#include <QTemporaryDir>
#include <cstdio>
#include <cstdlib>

// Launcher settings migrations (Configuration::ReadSettings) on every platform, without widgets; MacLauncherTests
// covers the UI.

static void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "LauncherSettingsTests: %s\n", message);
		std::abort();
	}
}

int main(int argc, char* argv[]) {
	QCoreApplication app(argc, argv);
	QTemporaryDir    temporary;
	Check(temporary.isValid(), "temporary directory");
	// Reads an ini saved with both options off; `version` < 0: saved before settings_version existed.
	const auto read = [&temporary](const char* name, int version, Configuration& out) {
		QSettings settings(temporary.filePath(QString::fromLatin1(name)), QSettings::IniFormat);
		if (version >= 0) {
			settings.setValue("settings_version", version);
		}
		settings.setValue("shader_cache_enabled", false);
		settings.setValue("tessellation_enabled", false);
		out.ReadSettings(&settings);
	};
	Configuration defaults;
	constexpr bool tess = Config::DEFAULT_TESSELLATION; // off on macOS
	Check(defaults.shader_cache_enabled && defaults.tessellation_enabled == tess, "shader store and tessellation defaults");
	Configuration unversioned;
	read("unversioned.ini", -1, unversioned);
	Check(unversioned.shader_cache_enabled && unversioned.tessellation_enabled == tess, "no settings_version: both migrations");
	Configuration version1;
	read("version1.ini", 1, version1);
	Check(!version1.shader_cache_enabled && version1.tessellation_enabled == tess, "settings_version=1: tessellation migrated, shader store kept off");
	Configuration version2;
	read("version2.ini", 2, version2);
	Check(!version2.shader_cache_enabled && !version2.tessellation_enabled, "settings_version=2: both choices kept");
	Configuration changed;
	changed.shader_cache_enabled = false;
	changed.tessellation_enabled = false;
	{
		QSettings settings(temporary.filePath("saved.ini"), QSettings::IniFormat);
		changed.WriteSettings(&settings);
		settings.sync();
		Check(settings.value("settings_version").toInt() == Configuration::SETTINGS_VERSION, "settings_version written");
		Configuration restored;
		restored.ReadSettings(&settings);
		Check(restored.GameSettings() == changed.GameSettings(), "options saved off by this launcher stay off");
	}
	std::puts("LauncherSettingsTests: ok");
	return 0;
}

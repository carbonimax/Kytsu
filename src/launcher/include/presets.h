#ifndef PRESETS_H
#define PRESETS_H

#include <QMap>
#include <QString>
#include <QStringList>

class QByteArray;
class QProcessEnvironment;

// Recommended options from a presets.json file:
// { "version": 1, "global": { "env": {...}, "args": [...] }, "titles": { "<TITLE_ID>": {...} } }
namespace Presets {

struct Preset {
	QMap<QString, QString> env;
	QStringList            args;
};

// Merges the global entry with the entry of title_id; the title wins.
// Malformed input returns false with error set and leaves out empty.
bool Parse(const QByteArray& json, const QString& title_id, Preset& out, QString& error);

// Adds only what is missing: variables already in env and options already in args win.
// Returns what was added, as KEY=VALUE and argument tokens.
QStringList Apply(const Preset& preset, QProcessEnvironment& env, QStringList& args);

// KYTY_PRESETS if set, else presets.json in dir.
QString FilePath(const QString& dir);

} // namespace Presets

#endif // PRESETS_H

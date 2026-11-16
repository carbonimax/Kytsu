#include "presets.h"

#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcessEnvironment>
#include <QRegularExpression>

namespace Presets {

static bool IsOption(const QString& arg) {
	return arg.startsWith(QStringLiteral("--"));
}

// Appends each option of src (with its value tokens) whose name is not in dst yet.
// shortcut: a value token starting with "--" reads as an option; no current option takes one.
static QStringList AppendMissing(QStringList& dst, const QStringList& src) {
	QStringList added;
	for (qsizetype i = 0; i < src.size();) {
		qsizetype end = i + 1;
		while (end < src.size() && !IsOption(src.at(end))) {
			++end;
		}
		if (!dst.contains(src.at(i))) {
			const auto group = src.mid(i, end - i);
			dst << group;
			added << group;
		}
		i = end;
	}
	return added;
}

// The Windows launcher runs the emulator through cmd.exe, which expands %VAR% before it reads quotes:
// a preset value or argument holding these characters could break out of CmdCommand's quoting.
static bool IsSafeText(const QString& text) {
	for (const QChar c: text) {
		const char16_t u = c.unicode();
		if (u < 0x20 || u == u'%' || u == u'"' || u == u'^' || u == u'!') {
			return false;
		}
	}
	return true;
}

static bool ReadEntry(const QJsonValue& value, const QString& where, Preset& out, QString& error) {
	if (value.isUndefined()) {
		return true;
	}
	if (!value.isObject()) {
		error = where + QStringLiteral(" must be an object");
		return false;
	}
	const auto entry = value.toObject();
	const auto env   = entry.value(QStringLiteral("env"));
	if (!env.isUndefined() && !env.isObject()) {
		error = where + QStringLiteral(".env must be an object");
		return false;
	}
	// Only emulator and MoltenVK variables: a preset must not reach loader paths such as VK_LAYER_PATH.
	static const QRegularExpression key_pattern(QStringLiteral("^(KYTY|MVK)_[A-Za-z0-9_]*$"));
	const auto                      vars = env.toObject();
	for (auto it = vars.constBegin(); it != vars.constEnd(); ++it) {
		if (!key_pattern.match(it.key()).hasMatch() || !it.value().isString() ||
		    !IsSafeText(it.value().toString())) {
			error = QStringLiteral("%1.env.%2 must be a KYTY_ or MVK_ variable with a plain string value")
			            .arg(where, it.key());
			return false;
		}
		out.env.insert(it.key(), it.value().toString());
	}
	const auto args = entry.value(QStringLiteral("args"));
	if (!args.isUndefined() && !args.isArray()) {
		error = where + QStringLiteral(".args must be an array");
		return false;
	}
	for (const auto& arg: args.toArray()) {
		if (!arg.isString() || (out.args.isEmpty() && !IsOption(arg.toString())) || !IsSafeText(arg.toString())) {
			error = where + QStringLiteral(".args must be plain strings starting with an --option");
			return false;
		}
		out.args << arg.toString();
	}
	return true;
}

bool Parse(const QByteArray& json, const QString& title_id, Preset& out, QString& error) {
	out = {};
	QJsonParseError parse_error {};
	const auto      doc = QJsonDocument::fromJson(json, &parse_error);
	if (!doc.isObject()) {
		error = parse_error.error != QJsonParseError::NoError ? parse_error.errorString()
		                                                       : QStringLiteral("root must be an object");
		return false;
	}
	const auto root = doc.object();
	if (root.value(QStringLiteral("version")).toInt(-1) != 1) {
		error = QStringLiteral("version must be 1");
		return false;
	}
	Preset global;
	if (!ReadEntry(root.value(QStringLiteral("global")), QStringLiteral("global"), global, error)) {
		return false;
	}
	const auto titles = root.value(QStringLiteral("titles"));
	if (!titles.isUndefined() && !titles.isObject()) {
		error = QStringLiteral("titles must be an object");
		return false;
	}
	// Every title is checked so a broken file fails the same way for all games.
	Preset     title;
	const auto entries = titles.toObject();
	for (auto it = entries.constBegin(); it != entries.constEnd(); ++it) {
		Preset entry;
		if (!ReadEntry(it.value(), QStringLiteral("titles.") + it.key(), entry, error)) {
			return false;
		}
		if (it.key() == title_id) {
			title = entry;
		}
	}
	out.env = global.env;
	out.env.insert(title.env);
	out.args = title.args;
	AppendMissing(out.args, global.args);
	error.clear();
	return true;
}

QStringList Apply(const Preset& preset, QProcessEnvironment& env, QStringList& args) {
	QStringList applied;
	for (auto it = preset.env.constBegin(); it != preset.env.constEnd(); ++it) {
		if (!env.contains(it.key())) {
			env.insert(it.key(), it.value());
			applied << it.key() + '=' + it.value();
		}
	}
	return applied + AppendMissing(args, preset.args);
}

QString FilePath(const QString& dir) {
	const auto path = qEnvironmentVariable("KYTY_PRESETS");
	return path.isEmpty() ? QDir(dir).filePath(QStringLiteral("presets.json")) : path;
}

} // namespace Presets

#ifndef LAUNCHER_INPUT_ICONS_H
#define LAUNCHER_INPUT_ICONS_H

#include <QIcon>
#include <QString>

// Pad glyphs and key/mouse prompts for the input mapping dialog, all drawn by code (no image files).
namespace InputIcons {

// "pad/<name>", "key/<name>" or "mouse/<name>". Null when that name has no drawing.
QIcon Icon(const QString& alias);
// The alias for a dialog control id ("Cross", "LeftStickUp"...), or empty.
QString PadControlAlias(const QString& id);
// The alias for a captured keyboard/mouse binding ("A", "F5", "Left Shift", "Mouse:Left"...), or empty.
QString HostInputAlias(const QString& binding);

} // namespace InputIcons

#endif // LAUNCHER_INPUT_ICONS_H

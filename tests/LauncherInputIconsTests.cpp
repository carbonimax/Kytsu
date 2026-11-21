#include "inputIcons.h"

#include <QGuiApplication>
#include <QImage>
#include <QPalette>
#include <QPixmap>
#include <QSet>
#include <QStringList>

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char *message, const QString &name = {}) {
  if (!value) {
    std::fprintf(stderr, "LauncherInputIconsTests: failed: %s %s\n", message,
                 qPrintable(name));
    std::abort();
  }
}

// The darkest/lightest painted pixel, to tell which text colour the icon used.
int PaintedLightness(const QIcon &icon, bool darkest) {
  const auto image = icon.pixmap(QSize(24, 24), 1.0).toImage();
  int best = darkest ? 256 : -1;
  for (int y = 0; y < image.height(); ++y) {
    for (int x = 0; x < image.width(); ++x) {
      const auto color = image.pixelColor(x, y);
      if (color.alpha() > 200) {
        best = darkest ? qMin(best, color.lightness()) : qMax(best, color.lightness());
      }
    }
  }
  return best;
}

} // namespace

int main(int argc, char **argv) {
  qputenv("QT_QPA_PLATFORM", "offscreen");
  QGuiApplication application(argc, argv);

  // Every name the input mapping dialog looks up: the pad control ids and the captured bindings.
  const QStringList pad_ids = {
      "Cross",          "Circle",        "Square",          "Triangle",      "Up",
      "Down",           "Left",          "Right",           "L1",            "R1",
      "L2",             "R2",            "L3",              "R3",            "Options",
      "TouchPad",       "TouchPadRight", "LeftStickUp",     "LeftStickDown", "LeftStickLeft",
      "LeftStickRight", "RightStickUp",  "RightStickDown",  "RightStickLeft", "RightStickRight"};
  QStringList bindings = {"Return",   "Backspace", "Tab",        "Space",      "Left Shift",
                          "Left Ctrl", "Left Alt",  "Insert",     "Delete",     "Home",
                          "End",      "PageUp",    "PageDown",   "CapsLock",   "Numlock",
                          "PrintScreen", "Left",   "Right",      "Up",         "Down",
                          "Mouse:Left", "Mouse:Right", "Mouse:Middle", "Mouse:X1", "Mouse:X2"};
  for (char c = 'A'; c <= 'Z'; ++c) {
    bindings << QString(QLatin1Char(c));
  }
  for (char c = '0'; c <= '9'; ++c) {
    bindings << QString(QLatin1Char(c));
  }
  for (int n = 1; n <= 12; ++n) {
    bindings << QStringLiteral("F%1").arg(n);
  }

  QSet<QString> aliases;
  for (const auto &id : pad_ids) {
    const auto alias = InputIcons::PadControlAlias(id);
    Check(!alias.isEmpty(), "no alias for pad control", id);
    aliases.insert(alias);
  }
  for (const auto &binding : bindings) {
    const auto alias = InputIcons::HostInputAlias(binding);
    Check(!alias.isEmpty(), "no alias for binding", binding);
    aliases.insert(alias);
  }
  Check(aliases.size() == 97, "expected 97 distinct names");

  for (const auto &alias : aliases) {
    const auto icon = InputIcons::Icon(alias);
    Check(!icon.isNull(), "null icon", alias);
    // 24 logical px at 2x is a 48 px pixmap, drawn at that size (crisp at any DPI).
    const auto pixmap = icon.pixmap(QSize(24, 24), 2.0);
    Check(pixmap.size() == QSize(48, 48) && pixmap.devicePixelRatio() == 2.0,
          "pixmap not at the device pixel ratio", alias);
    const auto image = pixmap.toImage();
    int painted = 0;
    for (int y = 0; y < image.height(); ++y) {
      for (int x = 0; x < image.width(); ++x) {
        painted += image.pixelColor(x, y).alpha() > 128 ? 1 : 0;
      }
    }
    Check(painted > 40, "icon paints almost nothing", alias);
  }

  // Unknown names keep the text fallback.
  Check(InputIcons::Icon(QString()).isNull(), "empty alias has an icon");
  Check(InputIcons::Icon(QStringLiteral("pad/unknown")).isNull(), "unknown alias has an icon");
  Check(InputIcons::Icon(QStringLiteral("key/esc")).isNull(), "esc is never looked up");
  for (const char *binding : {"Keypad 1", "F13", "Left GUI", "ScrollLock", "Mouse:Wheel"}) {
    Check(InputIcons::HostInputAlias(QString::fromLatin1(binding)).isEmpty(),
          "binding without a drawing got an alias", QString::fromLatin1(binding));
  }
  Check(InputIcons::Icon(InputIcons::HostInputAlias(QString(QChar(0xE9)))).isNull(),
        "a non-ASCII key has an icon");
  Check(InputIcons::PadControlAlias(QStringLiteral("GyroToggle")).isEmpty(),
        "control without a drawing got an alias");

  // The colour follows the current palette (light and dark themes), not the first one used.
  const auto icon = InputIcons::Icon(QStringLiteral("pad/cross"));
  QPalette light;
  light.setColor(QPalette::WindowText, Qt::black);
  QGuiApplication::setPalette(light);
  Check(PaintedLightness(icon, true) < 60, "light theme icon is not dark");
  QPalette dark;
  dark.setColor(QPalette::WindowText, Qt::white);
  QGuiApplication::setPalette(dark);
  Check(PaintedLightness(icon, false) > 200, "dark theme icon is not light");

  std::puts("LauncherInputIconsTests: OK");
  return 0;
}

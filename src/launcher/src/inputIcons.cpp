#include "inputIcons.h"

#include <QFontMetricsF>
#include <QGuiApplication>
#include <QHash>
#include <QIconEngine>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPixmap>

#include <functional>
#include <initializer_list>
#include <utility>

namespace {

// Every drawing works on a 24x24 grid, with the pen and colour set by the engine.
using Draw = std::function<void(QPainter&)>;

constexpr QPointF CENTER(12, 12);

class PaintedIconEngine final: public QIconEngine {
public:
	explicit PaintedIconEngine(Draw draw): m_draw(std::move(draw)) {}

	void paint(QPainter* painter, const QRect& rect, QIcon::Mode mode, QIcon::State /*state*/) override {
		// The colour is read at paint time, so a theme switch repaints with the new text colour.
		const auto group = mode == QIcon::Disabled ? QPalette::Disabled : QPalette::Active;
		const auto color = QGuiApplication::palette().color(group, QPalette::WindowText);
		painter->save();
		painter->setRenderHint(QPainter::Antialiasing);
		painter->translate(rect.topLeft());
		painter->scale(rect.width() / 24.0, rect.height() / 24.0);
		painter->setPen(QPen(color, 1.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
		painter->setBrush(Qt::NoBrush);
		m_draw(*painter);
		painter->restore();
	}

	QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override {
		return scaledPixmap(size, mode, state, 1.0);
	}

	// size is in device-independent pixels; the test checks the 24 px at 2x -> 48 px contract.
	QPixmap scaledPixmap(const QSize& size, QIcon::Mode mode, QIcon::State state, qreal scale) override {
		QPixmap pixmap(size * scale);
		pixmap.fill(Qt::transparent);
		{
			QPainter painter(&pixmap);
			paint(&painter, pixmap.rect(), mode, state);
		}
		pixmap.setDevicePixelRatio(scale);
		return pixmap;
	}

	[[nodiscard]] QIconEngine* clone() const override { return new PaintedIconEngine(m_draw); }

private:
	Draw m_draw;
};

void Fill(QPainter& p, const QPainterPath& path) {
	p.fillPath(path, p.pen().color());
}

// Text as a filled outline, fitted into box: crisp at any size and free of font hinting.
void Label(QPainter& p, const QString& text, const QRectF& box) {
	QFont font = QGuiApplication::font();
	font.setPixelSize(100);
	font.setBold(true);
	QPainterPath path;
	path.addText(0, 0, font, text);
	const auto bounds = path.boundingRect();
	const auto cap    = QFontMetricsF(font).capHeight();
	if (bounds.isEmpty() || cap <= 0) {
		return;
	}
	const auto scale = qMin(box.width() / bounds.width(), box.height() / cap);
	QTransform transform;
	transform.translate(box.center().x(), box.center().y());
	transform.scale(scale, scale);
	transform.translate(-bounds.center().x(), cap / 2);
	Fill(p, transform.map(path));
}

// Runs draw rotated around the centre: 0 up, 90 right, 180 down, 270 left.
void Rotated(QPainter& p, int degrees, const std::function<void()>& draw) {
	p.save();
	p.translate(CENTER);
	p.rotate(degrees);
	p.translate(-CENTER);
	draw();
	p.restore();
}

QPainterPath Polygon(std::initializer_list<QPointF> points) {
	QPainterPath path;
	path.addPolygon(QPolygonF(points));
	path.closeSubpath();
	return path;
}

// Pad.

void FaceButton(QPainter& p, char symbol) {
	p.drawEllipse(CENTER, 10.5, 10.5);
	auto pen = p.pen();
	pen.setWidthF(2);
	p.setPen(pen);
	switch (symbol) {
		case 'x':
			p.drawLine(QPointF(8, 8), QPointF(16, 16));
			p.drawLine(QPointF(16, 8), QPointF(8, 16));
			break;
		case 'o': p.drawEllipse(CENTER, 4.5, 4.5); break;
		case 's': p.drawRect(QRectF(7.75, 7.75, 8.5, 8.5)); break;
		default: p.drawPath(Polygon({{12, 6.8}, {17, 15.5}, {7, 15.5}})); break;
	}
}

void Dpad(QPainter& p, int degrees) {
	QPainterPath vertical;
	vertical.addRect(8.5, 2.5, 7, 19);
	QPainterPath horizontal;
	horizontal.addRect(2.5, 8.5, 19, 7);
	p.drawPath(vertical.united(horizontal));
	Rotated(p, degrees, [&] {
		QPainterPath arm;
		arm.addRect(8.5, 2.5, 7, 6);
		Fill(p, arm);
	});
}

void Shoulder(QPainter& p, const QString& text, bool trigger) {
	if (trigger) {
		QPainterPath shape;
		shape.moveTo(4, 21.5);
		shape.lineTo(4, 9);
		shape.quadTo(4, 2.5, 12, 2.5);
		shape.quadTo(20, 2.5, 20, 9);
		shape.lineTo(20, 21.5);
		shape.closeSubpath();
		p.drawPath(shape);
		Label(p, text, QRectF(6.5, 10, 11, 8));
	} else {
		p.drawRoundedRect(QRectF(2, 6, 20, 12), 4, 4);
		Label(p, text, QRectF(5.5, 8.5, 13, 7));
	}
}

void StickPress(QPainter& p, const QString& text) {
	p.drawEllipse(CENTER, 10.5, 10.5);
	p.drawEllipse(CENTER, 8, 8);
	Label(p, text, QRectF(7, 9, 10, 6));
}

void StickDirection(QPainter& p, const QString& letter, int degrees) {
	p.drawEllipse(CENTER, 7, 7);
	Label(p, letter, QRectF(8.5, 8.5, 7, 7));
	Rotated(p, degrees, [&] { Fill(p, Polygon({{12, 0.8}, {15.5, 4.3}, {8.5, 4.3}})); });
}

void Options(QPainter& p) {
	p.drawRoundedRect(QRectF(4.5, 6, 15, 12), 6, 6);
	for (const double y: {9.5, 12.0, 14.5}) {
		p.drawLine(QPointF(9, y), QPointF(15, y));
	}
}

void Touchpad(QPainter& p) {
	p.drawRoundedRect(QRectF(1.5, 5.5, 21, 13), 3, 3);
	QPainterPath finger;
	finger.addEllipse(CENTER, 2, 2);
	Fill(p, finger);
}

// Keyboard.

void KeyCap(QPainter& p) {
	p.drawRoundedRect(QRectF(1.75, 1.75, 20.5, 20.5), 4, 4);
}

void KeyText(QPainter& p, const QString& text) {
	KeyCap(p);
	// One character fills the cap; longer names get the full width.
	Label(p, text, text.size() == 1 ? QRectF(7, 7, 10, 10) : QRectF(4, 8, 16, 8));
}

void KeyArrow(QPainter& p, int degrees) {
	KeyCap(p);
	Rotated(p, degrees, [&] {
		p.drawLine(QPointF(12, 17), QPointF(12, 7));
		p.drawPolyline(QPolygonF({{8, 11}, {12, 7}, {16, 11}}));
	});
}

void KeyReturn(QPainter& p) {
	KeyCap(p);
	p.drawPolyline(QPolygonF({{17, 6.5}, {17, 13.5}, {7, 13.5}}));
	p.drawPolyline(QPolygonF({{10, 10.5}, {7, 13.5}, {10, 16.5}}));
}

void KeyBackspace(QPainter& p) {
	KeyCap(p);
	p.drawPath(Polygon({{9, 7.5}, {18, 7.5}, {18, 16.5}, {9, 16.5}, {5, 12}}));
	p.drawLine(QPointF(11.5, 10), QPointF(15.5, 14));
	p.drawLine(QPointF(15.5, 10), QPointF(11.5, 14));
}

void KeyTab(QPainter& p) {
	KeyCap(p);
	p.drawLine(QPointF(5.5, 12), QPointF(16, 12));
	p.drawPolyline(QPolygonF({{12.5, 8.5}, {16, 12}, {12.5, 15.5}}));
	p.drawLine(QPointF(18, 7.5), QPointF(18, 16.5));
}

void KeyShift(QPainter& p, bool lock) {
	KeyCap(p);
	if (lock) {
		p.drawPath(Polygon({{12, 5}, {17.5, 10.5}, {14.75, 10.5}, {14.75, 14.5}, {9.25, 14.5},
		                    {9.25, 10.5}, {6.5, 10.5}}));
		p.drawLine(QPointF(9.25, 17.5), QPointF(14.75, 17.5));
	} else {
		p.drawPath(Polygon(
		    {{12, 5.5}, {18, 12}, {15, 12}, {15, 18}, {9, 18}, {9, 12}, {6, 12}}));
	}
}

void KeySpace(QPainter& p) {
	KeyCap(p);
	p.drawPolyline(QPolygonF({{6, 11}, {6, 15}, {18, 15}, {18, 11}}));
}

// Mouse: a pill with two top buttons, a wheel and two side buttons; the named part is filled.

void Mouse(QPainter& p, const QString& part) {
	QPainterPath body;
	body.addRoundedRect(QRectF(6, 1.5, 14, 21), 7, 7);
	QPainterPath wheel;
	wheel.addRoundedRect(QRectF(11.5, 3.5, 3, 5), 1.5, 1.5);
	if (part == QLatin1String("left") || part == QLatin1String("right")) {
		QPainterPath button;
		button.addRect(part == QLatin1String("left") ? QRectF(0, 0, 13, 10) : QRectF(13, 0, 11, 10));
		QPainterPath gap;
		gap.addRoundedRect(QRectF(10.5, 2.5, 5, 7), 2.5, 2.5);
		Fill(p, button.intersected(body).subtracted(gap));
	}
	p.save();
	p.setClipPath(body);
	p.drawLine(QPointF(0, 10), QPointF(24, 10));
	p.drawLine(QPointF(13, 0), QPointF(13, 3.5));
	p.drawLine(QPointF(13, 8.5), QPointF(13, 10));
	p.restore();
	p.drawPath(body);
	part == QLatin1String("middle") ? Fill(p, wheel) : p.drawPath(wheel);

	// X2 (forward) above X1 (back), on the thumb side.
	for (const auto& [name, y]: {std::pair {"x2", 11.0}, std::pair {"x1", 15.5}}) {
		QPainterPath side;
		side.addRoundedRect(QRectF(2, y, 2.5, 3.5), 1, 1);
		part == QLatin1String(name) ? Fill(p, side) : p.drawPath(side);
	}
}

const QHash<QString, Draw>& Drawings() {
	static const QHash<QString, Draw> drawings = [] {
		QHash<QString, Draw> d;
		const auto add = [&d](const char* alias, Draw draw) { d.insert(QString::fromLatin1(alias), std::move(draw)); };

		add("pad/cross", [](QPainter& p) { FaceButton(p, 'x'); });
		add("pad/circle", [](QPainter& p) { FaceButton(p, 'o'); });
		add("pad/square", [](QPainter& p) { FaceButton(p, 's'); });
		add("pad/triangle", [](QPainter& p) { FaceButton(p, 't'); });
		const std::pair<const char*, int> directions[] = {{"up", 0}, {"right", 90}, {"down", 180}, {"left", 270}};
		for (const auto& direction: directions) {
			const auto dir     = QString::fromLatin1(direction.first);
			const int  degrees = direction.second;
			d.insert(QStringLiteral("pad/dpad_") + dir, [degrees](QPainter& p) { Dpad(p, degrees); });
			d.insert(QStringLiteral("pad/lstick_") + dir,
			         [degrees](QPainter& p) { StickDirection(p, QStringLiteral("L"), degrees); });
			d.insert(QStringLiteral("pad/rstick_") + dir,
			         [degrees](QPainter& p) { StickDirection(p, QStringLiteral("R"), degrees); });
			d.insert(QStringLiteral("key/arrow_") + dir, [degrees](QPainter& p) { KeyArrow(p, degrees); });
		}
		for (const char* name: {"L1", "R1", "L2", "R2"}) {
			const auto text    = QString::fromLatin1(name);
			const bool trigger = name[1] == '2';
			d.insert(QStringLiteral("pad/") + text.toLower(),
			         [text, trigger](QPainter& p) { Shoulder(p, text, trigger); });
		}
		add("pad/l3", [](QPainter& p) { StickPress(p, QStringLiteral("L3")); });
		add("pad/r3", [](QPainter& p) { StickPress(p, QStringLiteral("R3")); });
		add("pad/options", Options);
		add("pad/touchpad", Touchpad);

		for (char c = 'a'; c <= 'z'; ++c) {
			d.insert(QStringLiteral("key/") + QLatin1Char(c),
			         [c](QPainter& p) { KeyText(p, QString(QLatin1Char(c)).toUpper()); });
		}
		for (char c = '0'; c <= '9'; ++c) {
			d.insert(QStringLiteral("key/") + QLatin1Char(c), [c](QPainter& p) { KeyText(p, QString(QLatin1Char(c))); });
		}
		for (int n = 1; n <= 12; ++n) {
			d.insert(QStringLiteral("key/f%1").arg(n), [n](QPainter& p) { KeyText(p, QStringLiteral("F%1").arg(n)); });
		}
		const std::pair<const char*, const char*> texts[] = {
		    {"ctrl", "Ctrl"},   {"alt", "Alt"},     {"insert", "Ins"},      {"delete", "Del"},
		    {"home", "Home"},   {"end", "End"},     {"pageup", "PgUp"},     {"pagedown", "PgDn"},
		    {"numlock", "Num"}, {"printscreen", "PrtSc"}};
		for (const auto& [name, text]: texts) {
			const auto label = QString::fromLatin1(text);
			d.insert(QStringLiteral("key/") + QString::fromLatin1(name), [label](QPainter& p) { KeyText(p, label); });
		}
		add("key/return", KeyReturn);
		add("key/backspace", KeyBackspace);
		add("key/tab", KeyTab);
		add("key/shift", [](QPainter& p) { KeyShift(p, false); });
		add("key/capslock", [](QPainter& p) { KeyShift(p, true); });
		add("key/space", KeySpace);

		for (const char* name: {"left", "right", "middle", "x1", "x2"}) {
			const auto part = QString::fromLatin1(name);
			d.insert(QStringLiteral("mouse/") + part, [part](QPainter& p) { Mouse(p, part); });
		}
		return d;
	}();
	return drawings;
}

} // namespace

namespace InputIcons {

QIcon Icon(const QString& alias) {
	static QHash<QString, QIcon> cache;
	if (const auto it = cache.constFind(alias); it != cache.constEnd()) {
		return *it;
	}
	const auto& drawings = Drawings();
	const auto  draw     = drawings.constFind(alias);
	if (draw == drawings.constEnd()) {
		return {};
	}
	return *cache.insert(alias, QIcon(new PaintedIconEngine(*draw)));
}

QString PadControlAlias(const QString& id) {
	static const QHash<QString, QString> icons = {
	    {QStringLiteral("Cross"), QStringLiteral("pad/cross")},
	    {QStringLiteral("Circle"), QStringLiteral("pad/circle")},
	    {QStringLiteral("Square"), QStringLiteral("pad/square")},
	    {QStringLiteral("Triangle"), QStringLiteral("pad/triangle")},
	    {QStringLiteral("Up"), QStringLiteral("pad/dpad_up")},
	    {QStringLiteral("Down"), QStringLiteral("pad/dpad_down")},
	    {QStringLiteral("Left"), QStringLiteral("pad/dpad_left")},
	    {QStringLiteral("Right"), QStringLiteral("pad/dpad_right")},
	    {QStringLiteral("L1"), QStringLiteral("pad/l1")},
	    {QStringLiteral("R1"), QStringLiteral("pad/r1")},
	    {QStringLiteral("L2"), QStringLiteral("pad/l2")},
	    {QStringLiteral("R2"), QStringLiteral("pad/r2")},
	    {QStringLiteral("L3"), QStringLiteral("pad/l3")},
	    {QStringLiteral("R3"), QStringLiteral("pad/r3")},
	    {QStringLiteral("Options"), QStringLiteral("pad/options")},
	    {QStringLiteral("TouchPad"), QStringLiteral("pad/touchpad")},
	    {QStringLiteral("TouchPadRight"), QStringLiteral("pad/touchpad")},
	    {QStringLiteral("LeftStickUp"), QStringLiteral("pad/lstick_up")},
	    {QStringLiteral("LeftStickDown"), QStringLiteral("pad/lstick_down")},
	    {QStringLiteral("LeftStickLeft"), QStringLiteral("pad/lstick_left")},
	    {QStringLiteral("LeftStickRight"), QStringLiteral("pad/lstick_right")},
	    {QStringLiteral("RightStickUp"), QStringLiteral("pad/rstick_up")},
	    {QStringLiteral("RightStickDown"), QStringLiteral("pad/rstick_down")},
	    {QStringLiteral("RightStickLeft"), QStringLiteral("pad/rstick_left")},
	    {QStringLiteral("RightStickRight"), QStringLiteral("pad/rstick_right")},
	};
	return icons.value(id);
}

QString HostInputAlias(const QString& binding) {
	static const QHash<QString, QString> named = {
	    {QStringLiteral("Return"), QStringLiteral("key/return")},
	    {QStringLiteral("Backspace"), QStringLiteral("key/backspace")},
	    {QStringLiteral("Tab"), QStringLiteral("key/tab")},
	    {QStringLiteral("Space"), QStringLiteral("key/space")},
	    {QStringLiteral("Left Shift"), QStringLiteral("key/shift")},
	    {QStringLiteral("Left Ctrl"), QStringLiteral("key/ctrl")},
	    {QStringLiteral("Left Alt"), QStringLiteral("key/alt")},
	    {QStringLiteral("Insert"), QStringLiteral("key/insert")},
	    {QStringLiteral("Delete"), QStringLiteral("key/delete")},
	    {QStringLiteral("Home"), QStringLiteral("key/home")},
	    {QStringLiteral("End"), QStringLiteral("key/end")},
	    {QStringLiteral("PageUp"), QStringLiteral("key/pageup")},
	    {QStringLiteral("PageDown"), QStringLiteral("key/pagedown")},
	    {QStringLiteral("CapsLock"), QStringLiteral("key/capslock")},
	    {QStringLiteral("Numlock"), QStringLiteral("key/numlock")},
	    {QStringLiteral("PrintScreen"), QStringLiteral("key/printscreen")},
	    {QStringLiteral("Left"), QStringLiteral("key/arrow_left")},
	    {QStringLiteral("Right"), QStringLiteral("key/arrow_right")},
	    {QStringLiteral("Up"), QStringLiteral("key/arrow_up")},
	    {QStringLiteral("Down"), QStringLiteral("key/arrow_down")},
	    {QStringLiteral("Mouse:Left"), QStringLiteral("mouse/left")},
	    {QStringLiteral("Mouse:Right"), QStringLiteral("mouse/right")},
	    {QStringLiteral("Mouse:Middle"), QStringLiteral("mouse/middle")},
	    {QStringLiteral("Mouse:X1"), QStringLiteral("mouse/x1")},
	    {QStringLiteral("Mouse:X2"), QStringLiteral("mouse/x2")},
	};
	auto alias = named.value(binding);
	if (alias.isEmpty() && binding.size() == 1 && binding.at(0).isLetterOrNumber()) {
		alias = QStringLiteral("key/") + binding.at(0).toLower();
	}
	if (alias.isEmpty() && binding.size() >= 2 && binding.size() <= 3 &&
	    binding.at(0).toUpper() == QLatin1Char('F')) {
		bool      ok = false;
		const int n  = binding.mid(1).toInt(&ok);
		if (ok && n >= 1 && n <= 12) {
			alias = QStringLiteral("key/f%1").arg(n);
		}
	}
	return alias;
}

} // namespace InputIcons

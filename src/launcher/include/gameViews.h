#ifndef GAME_VIEWS_H
#define GAME_VIEWS_H

#include "gameContent.h"
#include "gameGridView.h"

#include <QGuiApplication>
#include <QIcon>
#include <QListView>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPixmap>
#include <QResizeEvent>
#include <QStackedWidget>
#include <QString>
#include <QStyledItemDelegate>
#include <QTreeWidget>

inline constexpr int GAME_NAME_COLUMN             = 0;
inline constexpr int GAME_SERIAL_COLUMN           = 1;
inline constexpr int GAME_VERSION_COLUMN          = 2;
inline constexpr int GAME_FIRMWARE_VERSION_COLUMN = 3;
inline constexpr int GAME_SIZE_COLUMN             = 4;
inline constexpr int GAME_PATH_COLUMN             = 5;
inline constexpr int GAME_STATUS_COLUMN           = 6;
inline constexpr int GAME_COMMENT_COLUMN          = 7;

// Holds the transparent list and grid views and paints the selected game's pic0 behind both.
class GameViews: public QStackedWidget {
public:
	using QStackedWidget::QStackedWidget;

	void SetBackgroundImage(const QString& key) {
		if (m_path == key) {
			return;
		}
		m_path   = key;
		m_source = QPixmap();
		if (!key.isEmpty()) {
			m_source.loadFromData(GameContent::ReadFile(key, QStringLiteral("sce_sys/pic0.png"),
			                                            GameContent::MaxImageSize));
		}
		UpdateScaledBackground();
		update();
	}

protected:
	void resizeEvent(QResizeEvent* event) override {
		QStackedWidget::resizeEvent(event);
		UpdateScaledBackground();
	}

	void paintEvent(QPaintEvent* event) override {
		QPainter painter(this);
		auto     overlay = palette().color(QPalette::Base);
		if (!m_scaled.isNull()) {
			painter.drawPixmap((width() - m_scaled.width()) / 2, (height() - m_scaled.height()) / 2,
			                   m_scaled);
			overlay.setAlpha(160);
		}
		painter.fillRect(event->rect(), overlay);
	}

private:
	void UpdateScaledBackground() {
		m_scaled = m_source.isNull() ? QPixmap()
		                             : m_source.scaled(size(), Qt::KeepAspectRatioByExpanding,
		                                               Qt::FastTransformation);
	}

	QString m_path;
	QPixmap m_source;
	QPixmap m_scaled;
};

// Marks a title that supports PS5 Pro mode with a pill in the top-right corner of its cover.
inline void PaintProBadge(QPainter* painter, const QRect& cover, QFont font, const QString& text) {
	font.setBold(true);
	const QFontMetrics metrics(font);
	const int          inset = metrics.height() / 3;
	QRect              badge(0, 0, metrics.horizontalAdvance(text) + 2 * inset, metrics.height());
	badge.moveTopRight(cover.topRight() + QPoint(-inset, inset));
	painter->save();
	painter->setRenderHint(QPainter::Antialiasing);
	painter->setPen(Qt::NoPen);
	painter->setBrush(QColor(0x534ab7));
	painter->drawRoundedRect(badge, 4, 4);
	painter->setFont(font);
	painter->setPen(QColor(0xeeedfe));
	painter->drawText(badge, Qt::AlignCenter, text);
	painter->restore();
}

// The carousel tile: the cover, enlarged, outlined and named when it is the selected game.
class GameGridDelegate: public QStyledItemDelegate {
public:
	using QStyledItemDelegate::QStyledItemDelegate;

	// PS5-style carousel: a single row of square tiles; only the selected one is enlarged and
	// outlined. The slider value is the size of the selected tile.
	// Cells are as wide as the largest tile, so they always cover the drawn (packed) tiles and
	// Qt never culls a visible one.
	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex&) const override {
		const int big = option.decorationSize.width() * 3 / 4;
		return QSize(big + GameGridView::TileGap, big + 12);
	}

	void paint(QPainter* painter, const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override {
		const auto* view = dynamic_cast<const GameGridView*>(option.widget);
		if (view == nullptr) {
			return;
		}
		const bool  selected = view->TileScale(index.row()) > 0.5;
		const QRect slot     = view->TileRect(index.row());
		const qreal pop      = view->PopScale(index.row());
		const QSize psize(qRound(slot.width() * qMax<qreal>(pop, 0.0)),
		                  qRound(slot.height() * qMax<qreal>(pop, 0.0)));
		QRect rect(QPoint(0, 0), psize);
		rect.moveCenter(slot.center());
		painter->save();
		painter->setOpacity(qBound<qreal>(0.0, pop * 1.5, 1.0));
		painter->setRenderHint(QPainter::Antialiasing);
		painter->setRenderHint(QPainter::SmoothPixmapTransform);

		QPainterPath clip;
		clip.addRoundedRect(rect, 12, 12);
		painter->setClipPath(clip);
		painter->fillRect(rect, QColor(30, 34, 44));
		const auto icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
		if (!icon.isNull()) {
			const auto pixmap = icon.pixmap(rect.size());
			painter->drawPixmap(rect, pixmap.scaled(rect.size(), Qt::KeepAspectRatioByExpanding,
			                                        Qt::SmoothTransformation));
		}
		painter->setClipping(false);

		if (selected) {
			const QRectF outline = QRectF(rect).adjusted(-2, -2, 2, 2);
			painter->setBrush(Qt::NoBrush);
			painter->setPen(QPen(QColor(255, 255, 255, 110), 3));
			painter->drawRoundedRect(outline, 14, 14);

			// A bright comet circling the outline, with a soft halo.
			QConicalGradient glow(outline.center(), view->GlowAngle());
			glow.setColorAt(0.0, QColor(255, 255, 255, 255));
			glow.setColorAt(0.22, QColor(120, 190, 255, 160));
			glow.setColorAt(0.45, QColor(120, 190, 255, 0));
			glow.setColorAt(1.0, QColor(255, 255, 255, 0));
			painter->setPen(QPen(QBrush(glow), 9));
			painter->setOpacity(painter->opacity() * 0.35);
			painter->drawRoundedRect(outline, 14, 14);
			painter->setOpacity(painter->opacity() / 0.35);
			painter->setPen(QPen(QBrush(glow), 3.5));
			painter->drawRoundedRect(outline, 14, 14);

			const auto name = index.data(Qt::DisplayRole).toString();
			if (!name.isEmpty()) {
				auto label_font = option.font;
				label_font.setPointSizeF(option.font.pointSizeF() * 1.1);
				label_font.setWeight(QFont::DemiBold);
				painter->setFont(label_font);
				painter->setPen(Qt::white);
				const QRect label(rect.center().x() - 180, rect.bottom() + 8, 360, 24);
				painter->drawText(label, Qt::AlignHCenter | Qt::AlignTop,
				                  QFontMetrics(label_font).elidedText(name, Qt::ElideRight, label.width()));
			}
		}
		painter->restore();
	}
};

// The list's name cell, with the PS5 Pro badge over its icon.
class GameListNameDelegate: public QStyledItemDelegate {
public:
	using QStyledItemDelegate::QStyledItemDelegate;

	void paint(QPainter* painter, const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override {
		QStyledItemDelegate::paint(painter, option, index);
		if (!index.data(Qt::UserRole).toBool()) {
			return;
		}
		QStyleOptionViewItem opt(option);
		initStyleOption(&opt, index);
		const auto icon = opt.widget->style()->subElementRect(QStyle::SE_ItemViewItemDecoration,
		                                                      &opt, opt.widget);
		// The list's small icon takes a smaller badge than the grid's covers.
		auto font = opt.font;
		font.setPixelSize(icon.height() / 5);
		PaintProBadge(painter, icon, font, QStringLiteral("PRO"));
	}
};

#endif // GAME_VIEWS_H

#include "gameViews.h"
#include "launcherTheme.h"
#include "ui_configuration_list_widget.h"

#include <QApplication>
#include <QColor>
#include <QDir>
#include <QElapsedTimer>
#include <QFontMetrics>
#include <QImage>
#include <QScrollBar>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "LauncherGameViewsTests: failed: %s\n", message);
    std::abort();
  }
}

// The red pic0 dominates a pixel only where nothing opaque is painted over the background.
bool ShowsBackground(const QColor &color) {
  return color.red() > color.green() + 40 && color.red() > color.blue() + 40;
}

} // namespace

int main(int argc, char **argv) {
  qputenv("QT_QPA_PLATFORM", "offscreen");
  QApplication application(argc, argv);
  // Breeze, where installed, paints its own color behind scroll bars; Fusion's are opaque.
  const bool breeze = QApplication::setStyle(QStringLiteral("Breeze")) != nullptr;
  // A fixed light palette, whatever the desktop colors: a black panel then stands out.
  QApplication::setPalette(QPalette(Qt::white));
  LauncherTheme::Initialize(application);

  QTemporaryDir game;
  Check(QDir(game.path()).mkpath(QStringLiteral("sce_sys")), "cannot create sce_sys");
  QImage pic0(64, 36, QImage::Format_RGB32);
  pic0.fill(Qt::red);
  Check(pic0.save(game.filePath(QStringLiteral("sce_sys/pic0.png"))), "cannot write pic0.png");

  QWidget                     host;
  Ui::ConfigurationListWidget ui;
  ui.setupUi(&host);
  auto *views = ui.game_views;
  auto *list  = ui.cfgs_list;
  auto *grid  = ui.cfgs_grid;
  // The launcher's carousel setup that the tiles depend on (ConfigureGameGrid).
  grid->setModel(list->model());
  grid->setSelectionModel(list->selectionModel());
  grid->setFlow(QListView::LeftToRight);
  grid->setWrapping(false);
  grid->setItemDelegate(new GameGridDelegate(grid));
  grid->setIconSize(QSize(128, 128));

  QPixmap cover(128, 128);
  cover.fill(Qt::blue);
  auto *item = new QTreeWidgetItem(list, {QStringLiteral("Game")});
  item->setIcon(GAME_NAME_COLUMN, QIcon(cover));

  host.resize(800, 600);
  views->SetBackgroundImage(game.path());
  host.show();
  QApplication::processEvents();
  const QPoint empty(views->width() - 60, views->height() - 30);

  Check(ShowsBackground(views->grab().toImage().pixelColor(empty)),
        "the game background is hidden behind the list view");

  if (breeze) {
    list->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    QApplication::processEvents();
    const auto *bar = list->verticalScrollBar();
    Check(ShowsBackground(views->grab().toImage().pixelColor(
              bar->mapTo(views, QPoint(2, bar->height() - 20)))),
          "Breeze hides the game background behind a scroll bar");
  }

  list->setCurrentItem(item);
  views->setCurrentWidget(grid);
  grid->SetBackgroundImage(game.path());
  // Let the pop-in and the selected tile's growth finish.
  QElapsedTimer clock;
  clock.start();
  while (clock.elapsed() < 1500) {
    QApplication::processEvents();
  }
  const auto image = grid->viewport()->grab().toImage();
  Check(ShowsBackground(image.pixelColor(grid->viewport()->width() - 10,
                                         grid->viewport()->height() - 10)),
        "the carousel does not paint the game background");
  const auto tile = grid->TileRect(0);
  Check(tile.width() == 128 * 3 / 4, "the selected tile is not three quarters of the slider size");
  Check(image.pixelColor(tile.center()) == QColor(Qt::blue), "the tile does not show the cover");

  // A title that supports PS5 Pro mode carries the badge in the tile's top-right corner.
  item->setData(GAME_NAME_COLUMN, Qt::UserRole, true);
  QApplication::processEvents();
  auto font = grid->font();
  font.setBold(true);
  const int    inset = QFontMetrics(font).height() / 3;
  const QPoint badge = tile.topRight() + QPoint(-inset - 2, inset + QFontMetrics(font).height() / 2);
  const auto   pro   = grid->viewport()->grab().toImage().pixelColor(badge);
  Check(qAbs(pro.red() - 0x53) < 24 && qAbs(pro.green() - 0x4a) < 24 && qAbs(pro.blue() - 0xb7) < 24,
        "the carousel tile does not show the PS5 Pro badge");

  std::printf("LauncherGameViewsTests: passed%s\n", breeze ? "" : " (Breeze checks skipped)");
  return 0;
}

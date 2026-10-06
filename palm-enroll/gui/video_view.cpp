#include "video_view.h"

#include <QPainter>
#include <QPainterPath>

namespace {

const int kBones[][2] = {{0, 1},   {1, 2},   {2, 3},   {3, 4},   {0, 5},   {5, 6},   {6, 7},
                         {7, 8},   {5, 9},   {9, 10},  {10, 11}, {11, 12}, {9, 13},  {13, 14},
                         {14, 15}, {15, 16}, {13, 17}, {0, 17},  {17, 18}, {18, 19}, {19, 20}};

QColor stateColor(palm::HandState s) {
  switch (s) {
    case palm::HandState::Ok: return QColor(46, 204, 113);
    case palm::HandState::LeftHand: return QColor(231, 76, 60);
    case palm::HandState::NoHand: return QColor(200, 200, 200);
    default: return QColor(243, 156, 18);
  }
}

void drawLabel(QPainter& p, const QRect& area, const QString& text, const QColor& bg, int px, int flags) {
  QFont f = p.font();
  f.setPixelSize(px);
  f.setBold(true);
  p.setFont(f);
  QRect r = p.fontMetrics().boundingRect(area, flags | Qt::TextWordWrap, text).adjusted(-14, -8, 14, 8);
  r.moveCenter(QPoint(area.center().x(), r.center().y()));
  QColor b = bg;
  b.setAlpha(210);
  p.setPen(Qt::NoPen);
  p.setBrush(b);
  p.drawRoundedRect(r, 8, 8);
  p.setPen(Qt::white);
  p.drawText(r, Qt::AlignCenter | Qt::TextWordWrap, text);
}

}  // namespace

VideoView::VideoView(bool mirror, QWidget* parent) : QWidget(parent), mirror_(mirror) {
  setMinimumSize(320, 240);
  setAttribute(Qt::WA_OpaquePaintEvent);
}

void VideoView::setFrame(const FrameResult& r) {
  r_ = r;
  update();
}

void VideoView::setBanner(const QString& text, const QColor& color) {
  banner_ = text;
  bannerColor_ = color;
  update();
}

void VideoView::paintEvent(QPaintEvent*) {
  QPainter p(this);
  p.fillRect(rect(), Qt::black);
  p.setRenderHint(QPainter::Antialiasing);
  if (r_.view.isNull()) {
    drawLabel(p, rect(), message_.isEmpty() ? QStringLiteral("Đang khởi động...") : message_, QColor(60, 60, 60), 22,
              Qt::AlignVCenter);
    return;
  }
  const double k = std::min(double(width()) / r_.view.width(), double(height()) / r_.view.height());
  const QSizeF size(r_.view.width() * k, r_.view.height() * k);
  const QRectF target((width() - size.width()) / 2, (height() - size.height()) / 2, size.width(), size.height());

  // image and geometry in a (possibly mirrored) frame coordinate system
  p.save();
  p.translate(target.topLeft());
  if (mirror_) {
    p.translate(size.width(), 0);
    p.scale(-1, 1);
  }
  p.drawImage(QRectF(QPointF(0, 0), size), r_.view);
  const double s = k * r_.viewScale;  // camera pixel -> widget pixel
  const QColor c = stateColor(r_.check.state);
  if (r_.hand.found) {
    const QRectF box(r_.check.box.x * s, r_.check.box.y * s, r_.check.box.width * s, r_.check.box.height * s);
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(c, 3));
    p.drawRect(box);
    p.setPen(QPen(QColor(255, 255, 255, 180), 2));
    for (const auto& b : kBones)
      p.drawLine(QPointF(r_.hand.pts[b[0]].x * s, r_.hand.pts[b[0]].y * s),
                 QPointF(r_.hand.pts[b[1]].x * s, r_.hand.pts[b[1]].y * s));
    p.setPen(Qt::NoPen);
    p.setBrush(c);
    for (const auto& q : r_.hand.pts) p.drawEllipse(QPointF(q.x * s, q.y * s), 3.5, 3.5);
  } else {
    // where to put the hand
    const double side = size.height() * 0.6;
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(QColor(255, 255, 255, 150), 2, Qt::DashLine));
    p.drawRoundedRect(QRectF((size.width() - side) / 2, (size.height() - side) / 2, side, side), 12, 12);
  }
  p.restore();

  const QRect area = target.toRect().adjusted(10, 10, -10, -10);
  drawLabel(p, area, QString::fromUtf8(palm::stateMessage(r_.check.state)), c.darker(130), 24, Qt::AlignTop);
  if (!banner_.isEmpty()) {
    QFont f = p.font();
    f.setPixelSize(22);
    p.setFont(f);
    const int h = p.fontMetrics().boundingRect(area, Qt::TextWordWrap, banner_).height() + 16;
    drawLabel(p, QRect(area.left(), area.bottom() - h, area.width(), h), banner_, bannerColor_, 22, Qt::AlignVCenter);
  }
}

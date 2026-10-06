#include "event_log.h"

#include <QDateTime>

namespace {

constexpr qint64 kMaxFileBytes = 10 << 20;

const char* levelColor(EventLog::Level l) {
  switch (l) {
    case EventLog::Ok: return "#2ecc71";
    case EventLog::Warn: return "#f0a030";
    case EventLog::Fail: return "#e74c3c";
    case EventLog::Info: break;
  }
  return "#cccccc";
}

}  // namespace

EventLog::EventLog(const QString& filePath, QWidget* parent) : QPlainTextEdit(parent), file_(filePath) {
  setReadOnly(true);
  setMaximumBlockCount(2000);  // older lines stay in the file only
  setLineWrapMode(QPlainTextEdit::NoWrap);
  QFont f(QStringLiteral("monospace"));
  f.setStyleHint(QFont::Monospace);
  f.setPixelSize(12);
  setFont(f);
  setStyleSheet("QPlainTextEdit { background:#151515; border:none; border-top:1px solid #555; }");
  if (!filePath.isEmpty() && !file_.open(QIODevice::Append | QIODevice::Text)) fileError_ = file_.errorString();
}

void EventLog::add(const QString& tag, const QString& text, Level level) {
  const QDateTime now = QDateTime::currentDateTime();
  const QString body = tag.leftJustified(7) + text;
  writeFile(now.toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz ")) + body);
  appendHtml(QStringLiteral("<span style='white-space:pre; color:%1'>%2</span>")
                 .arg(levelColor(level), (now.toString(QStringLiteral("HH:mm:ss.zzz ")) + body).toHtmlEscaped()));
}

void EventLog::writeFile(const QString& line) {
  if (!file_.isOpen()) return;
  if (file_.size() > kMaxFileBytes) {
    file_.close();
    const QString old = file_.fileName() + ".1";
    QFile::remove(old);
    QFile::rename(file_.fileName(), old);
    if (!file_.open(QIODevice::Append | QIODevice::Text)) {
      fileError_ = file_.errorString();
      return;
    }
  }
  file_.write(line.toUtf8() + '\n');
  file_.flush();  // readable with tail -f, kept if the app dies
}

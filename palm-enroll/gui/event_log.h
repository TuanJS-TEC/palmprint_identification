#pragma once
// Event log of palm_gui: shown in a panel under the video and appended to a
// file (UTF-8, one line per event, rotated to <file>.1 at 10 MB).
#include <QFile>
#include <QPlainTextEdit>
#include <QString>

class EventLog : public QPlainTextEdit {
  Q_OBJECT
 public:
  enum Level { Info, Ok, Warn, Fail };

  // filePath empty: panel only
  explicit EventLog(const QString& filePath, QWidget* parent = nullptr);

  void add(const QString& tag, const QString& text, Level level = Info);

  QString filePath() const { return file_.fileName(); }
  bool fileOpen() const { return file_.isOpen(); }
  QString fileError() const { return fileError_; }

 private:
  void writeFile(const QString& line);

  QFile file_;
  QString fileError_;
};

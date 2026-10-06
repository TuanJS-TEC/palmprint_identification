// palm_gui: right-hand palmprint enrollment and identification on a live camera.
//
// usage: palm_gui [config.yaml] [--source IMAGE|VIDEO [--mirrored]] [--fullscreen]
//                 [--enroll ID | --identify] [--verbose] [--screenshot FILE --quit-after SEC]
//   config.yaml   default: next to the executable
//   --source      replay an image or a video instead of the camera (testing);
//                 --mirrored if it is a mirror image of the camera view
//   --enroll ID   start enrolling ID right away; --identify: start identifying
//   --verbose     timing summary on stderr every second
//   --screenshot  save the window after --quit-after seconds and quit (smoke test)
#include <iostream>

#include <QApplication>
#include <QDir>
#include <QMessageBox>
#include <QTimer>

#include "config.h"
#include "main_window.h"

int main(int argc, char** argv) {
  QApplication app(argc, argv);  // consumes Qt options such as -platform
  qRegisterMetaType<FrameResult>("FrameResult");

  QString cfgPath = QDir(QCoreApplication::applicationDirPath()).filePath("config.yaml");
  std::string source;
  bool mirrored = false, fullscreen = false, identify = false, verbose = false;
  QString shot, enrollId;
  int quitAfter = 0;
  const QStringList args = QCoreApplication::arguments();
  for (int i = 1; i < args.size(); ++i) {
    const QString& a = args[i];
    auto value = [&]() -> QString {
      if (i + 1 >= args.size()) throw std::runtime_error("missing value for " + a.toStdString());
      return args[++i];
    };
    try {
      if (a == "--source") source = value().toStdString();
      else if (a == "--mirrored") mirrored = true;
      else if (a == "--fullscreen") fullscreen = true;
      else if (a == "--enroll") enrollId = value();
      else if (a == "--identify") identify = true;
      else if (a == "--verbose") verbose = true;
      else if (a == "--screenshot") shot = value();
      else if (a == "--quit-after") quitAfter = value().toInt();
      else if (!a.startsWith("--")) cfgPath = a;
      else throw std::runtime_error("unknown option " + a.toStdString());
    } catch (const std::exception& e) {
      std::cerr << "palm_gui: " << e.what() << "\n";
      return 1;
    }
  }

  palm::Config cfg;
  try {
    cfg = palm::loadConfig(cfgPath.toStdString());
  } catch (const std::exception& e) {
    QMessageBox::critical(nullptr, "palm_gui", QStringLiteral("Không đọc được cấu hình %1:\n%2").arg(cfgPath, e.what()));
    return 1;
  }

  MainWindow w(cfg, source, mirrored);
  w.setVerbose(verbose);
  const bool embedded = QGuiApplication::platformName().startsWith("eglfs") ||
                        QGuiApplication::platformName().startsWith("linuxfb");
  if (fullscreen || embedded) {
    w.showFullScreen();
  } else {
    w.resize(1366, 768);
    w.show();
  }
  if (!enrollId.isEmpty()) w.beginEnroll(enrollId);
  else if (identify) w.beginIdentify();
  if (quitAfter > 0) {
    QTimer::singleShot(quitAfter * 1000, &w, [&] {
      if (!shot.isEmpty()) w.grab().save(shot);
      w.close();
    });
  }
  return app.exec();
}

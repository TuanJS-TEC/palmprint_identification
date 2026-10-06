#pragma once
// From a camera frame + hand landmarks to the hand crop EE-PRnet gets, and the
// checks a live capture applies before using it.
#include <string>

#include <opencv2/core.hpp>

#include "hand_detector.h"

namespace palm {

// Square box like output/hand_crop: centred on the landmark bounding box,
// side = scale * max(width, height). scale 1.17 was fitted on 596 crops of the
// capture log (p10-p90 1.14-1.20, centre offset ~0).
cv::Rect handCropBox(const HandLandmarks& h, float scale);

// Variance of the Laplacian of the crop (gray, resized to 224x224 INTER_AREA).
float sharpness(const cv::Mat& bgrCrop);

// Mean landmark displacement between two detections, relative to `side`.
float landmarkMotion(const HandLandmarks& a, const HandLandmarks& b, float side);

struct CaptureLimits {
  float cropScale = 1.17f;
  float minCropFrac = 0.33f;  // crop side / frame height
  float maxMotion = 0.02f;    // per frame, relative to the crop side
  float blurMin = 0.f;        // sharpness()
};

enum class HandState {
  NoHand,
  LeftHand,      // a left palm: only the right hand is accepted
  BackOfHand,    // the right hand shows its back (or a left back)
  TooSmall,
  OutOfFrame,
  Moving,
  Blurry,
  Ok,
};
const char* stateMessage(HandState s);  // Vietnamese UI text

struct HandCheck {
  HandState state = HandState::NoHand;
  cv::Rect box;      // hand crop box in the frame
  float sharp = 0.f;
  float motion = 0.f;
};

// frame: non-mirrored camera frame (what the detector saw). prev: detection of
// the previous frame (motion), may be not found.
HandCheck checkHand(const cv::Mat& frame, const HandLandmarks& h, const HandLandmarks& prev, const CaptureLimits& lim);

}  // namespace palm

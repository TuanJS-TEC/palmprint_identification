#ifndef PALM_API_H
#define PALM_API_H
/*
 * C API of palm-enroll: right-hand palmprint registration and login from raw
 * camera frames. Plain C so that it can be wrapped by JNI (Android NDK .so)
 * or called through any FFI.
 *
 *   palm_init            load the models and the gallery (database)
 *   palm_extract         frame -> 512-d feature vector
 *   palm_compare         two feature vectors -> cosine score, same person?
 *   palm_register        frames of a new user -> new id (Register)
 *   palm_check_register  one frame -> id of the enrolled user (Check_Register)
 *
 * Conventions
 * - Every function returns a PalmStatus (PALM_OK = 0) and never throws.
 * - NULL for a required pointer -> PALM_ERR_ARGUMENT. Optional outputs
 *   (score, same, frame_status) may be NULL and are then not written.
 * - Every id_out buffer must hold at least PALM_ID_MAX bytes (id_max_len
 *   says how many); on failure id_out is "" except for
 *   PALM_ERR_ALREADY_REGISTERED, where it is the id already enrolled.
 * - Calls on one handle are serialised by a mutex. Do not call palm_release
 *   while another thread still uses the handle.
 * - Only the physical right hand is accepted, palm facing the camera.
 */
#include <stddef.h>

#if defined(_WIN32)
#define PALM_API
#else
#define PALM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PALM_FEATURE_DIM 512
#define PALM_ID_MAX 64      /* minimum size of an id_out buffer, '\0' included */
#define PALM_MAX_FRAMES 32  /* palm_register takes 1..PALM_MAX_FRAMES frames */

typedef struct PalmContext* PalmHandle;

typedef enum PalmStatus {
  PALM_OK = 0,
  PALM_ERR_ARGUMENT = 1,   /* NULL pointer, buffer too small, bad frame description */
  PALM_ERR_INIT = 2,       /* models or gallery cannot be loaded, missing paths */
  PALM_ERR_IO = 3,         /* gallery cannot be saved */
  PALM_ERR_INTERNAL = 4,   /* unexpected error, see palm_last_error */
  /* the frame has no usable right palm */
  PALM_ERR_NO_HAND = 10,
  PALM_ERR_LEFT_HAND = 11,
  PALM_ERR_BACK_OF_HAND = 12,
  PALM_ERR_HAND_TOO_SMALL = 13,
  PALM_ERR_HAND_OUT_OF_FRAME = 14,
  PALM_ERR_BLURRY = 15,
  /* palm_register */
  PALM_ERR_TOO_FEW_SAMPLES = 20,     /* fewer usable or consistent frames than template.min_valid */
  PALM_ERR_ALREADY_REGISTERED = 21,  /* the palm matches an enrolled user, whose id is in id_out */
  /* palm_check_register */
  PALM_ERR_GALLERY_EMPTY = 30,
  PALM_ERR_NOT_RECOGNIZED = 31
} PalmStatus;

typedef enum PalmPixelFormat {
  PALM_FMT_BGR = 0,  /* 3 bytes per pixel */
  PALM_FMT_RGB = 1,
  PALM_FMT_BGRA = 2, /* 4 bytes per pixel */
  PALM_FMT_RGBA = 3, /* Android Bitmap ARGB_8888 memory layout, CameraX RGBA_8888 */
  PALM_FMT_NV21 = 4  /* Y plane, then interleaved V/U at half resolution */
} PalmPixelFormat;

typedef struct PalmFrame {
  const unsigned char* data;
  size_t size;    /* bytes readable at data: at least height*stride, plus (height/2)*stride for NV21 */
  int width;      /* 1..16384, even for NV21 */
  int height;     /* 1..16384, even for NV21 */
  int stride;     /* bytes per row (of the Y plane for NV21, whose V/U plane starts at
                     data + height*stride with the same stride); 0 = tightly packed */
  int format;     /* PalmPixelFormat */
  int rotation;   /* 0, 90, 180 or 270: clockwise rotation that makes the frame upright
                     (Android ImageInfo.getRotationDegrees()) */
  int mirrored;   /* 1 if the upright image is a left-right mirror image of the scene
                     (selfie preview); flipped after the rotation. Camera buffers,
                     front camera included, are not mirrored: 0 */
} PalmFrame;

typedef struct PalmInitOptions {
  /* config.yaml (thresholds, model and gallery paths relative to it), or NULL
     for the built-in thresholds measured on 2026-10-05. With NULL, the four
     paths below are required. A non-NULL path below overrides the config. */
  const char* config_path;
  const char* embed_onnx;     /* EE_PRnet-epoch-60.onnx */
  const char* palm_onnx;      /* palm_detection_mediapipe_2023feb.onnx */
  const char* landmark_onnx;  /* handpose_estimation_mediapipe_2023feb.onnx */
  const char* gallery_path;   /* database file, created by the first registration */
  int threads;                /* intra-op threads of the embedding model; 0 = config / 4 */
} PalmInitOptions;

/* *out = NULL on failure; the reason is palm_last_error(NULL). */
PALM_API int palm_init(const PalmInitOptions* opt, PalmHandle* out);
/* NULL is a no-op, like free(). */
PALM_API void palm_release(PalmHandle h);

/* feature: buffer of feature_len >= PALM_FEATURE_DIM floats, receives the
   L2-normalised feature of the right palm in the frame. */
PALM_API int palm_extract(PalmHandle h, const PalmFrame* frame, float* feature, int feature_len);

/* dim must be PALM_FEATURE_DIM. score: cosine of the two vectors (normalised
   first); same: score >= match.accept_cos_min. A zero or non-finite vector
   gives PALM_ERR_ARGUMENT with score 0, same 0. */
PALM_API int palm_compare(PalmHandle h, const float* a, const float* b, int dim, float* score, int* same);

/* Register: frames of one new user, ideally 5 taken ~0.7 s apart with the hand
   tilted slightly between them. Fails with PALM_ERR_ALREADY_REGISTERED when
   the palm matches an enrolled user; otherwise stores the user and writes the
   new id ("user_0001", ...) to id_out. score: highest cosine to another
   enrolled user (0 if none). frame_status: NULL, or an array of at least n
   ints receiving the status of each frame. When the gallery cannot be saved
   (PALM_ERR_IO) the id is not reused, like a database auto-increment. */
PALM_API int palm_register(PalmHandle h, const PalmFrame* frames, int n, char* id_out, int id_max_len,
                           float* score, int* frame_status);

/* Check_Register: 1:N search of one frame. PALM_OK and the id when the best
   score reaches match.accept_cos_min, else PALM_ERR_NOT_RECOGNIZED (id_out "",
   score = best score, for logging). One frame can be wrong: require a few
   consecutive PALM_OK with the same id for a login. */
PALM_API int palm_check_register(PalmHandle h, const PalmFrame* frame, char* id_out, int id_max_len, float* score);

/* Static English text, never NULL. */
PALM_API const char* palm_status_text(int status);
/* Why the last call on h failed, or a warning ("" otherwise); valid until
   the next call on h. With h NULL: why the last palm_init on this thread
   failed. Never NULL. */
PALM_API const char* palm_last_error(PalmHandle h);

#ifdef __cplusplus
}
#endif

#endif /* PALM_API_H */

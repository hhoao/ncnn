// Link anchor for Apple platforms (macOS / iOS).
//
// ncnn is a pure-FFI pod: no ObjC/Swift plugin class ever references
// the hn_* entry points from native code, so when linking the app the
// static-archive members that define them are never pulled in (and would be
// dead-code-stripped even if they were). dart:ffi resolves them via
// dlsym(RTLD_DEFAULT, ...) at runtime, which only sees symbols that actually
// made it into the binary.
//
// CocoaPods passes -ObjC when linking the app, which loads every static
// archive member containing ObjC metadata — including this one. +load of a
// registered ObjC class is a dead-strip root, and its body stores the
// addresses of all hn_* entry points into a volatile table, forcing the
// linker to pull in (and keep) ncnn_api.o and, transitively, the ncnn
// static frameworks.
#import <Foundation/Foundation.h>

#include "ncnn_api.h"

static void *volatile ncnn_link_anchors[9];

@interface NcnnLinkAnchor : NSObject
@end

@implementation NcnnLinkAnchor

+ (void)load {
  // Never executed beyond the stores; addresses must simply be resolved.
  ncnn_link_anchors[0] = (void *)&hn_create;
  ncnn_link_anchors[1] = (void *)&hn_load;
  ncnn_link_anchors[2] = (void *)&hn_output_count;
  ncnn_link_anchors[3] = (void *)&hn_output_shape;
  ncnn_link_anchors[4] = (void *)&hn_extract;
  ncnn_link_anchors[5] = (void *)&hn_extract_f32;
  ncnn_link_anchors[6] = (void *)&hn_destroy;
  ncnn_link_anchors[7] = (void *)&hn_gpu_count;
  ncnn_link_anchors[8] = (void *)&hn_gpu_devices;
}

@end

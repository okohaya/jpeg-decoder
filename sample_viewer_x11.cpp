#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

uint8_t* load_jpeg(const char* filename, int* width, int* height, int* channel);


XImage* create_image(Display* display, Visual* visual, uint8_t* image, int width, int height) {
    uint8_t* p = (uint8_t*)malloc(width * height * 4);    // [B,G,R,_]
    uint8_t* src = image;
    uint8_t* dst = p;

    for (int i = 0; i < width * height; i++) {
        dst[3] = 0;
        dst[2] = src[0];
        dst[1] = src[1];
        dst[0] = src[2];
        dst += 4;
        src += 3;
    }

    return XCreateImage(display, visual, 24, ZPixmap, 0, (char*)p, width, height, 32, 0);
}


int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("usage: %s <filename>\n", argv[0]);
        return 1;
    }
    const char* filename = argv[1];
    int width, height, n;
    uint8_t* data = load_jpeg(filename, &width, &height, &n);

    Display* display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "Cannot open display\n");
        return 1;
    }

    int screen = DefaultScreen(display);
    unsigned long white = WhitePixel(display, screen);
    unsigned long black = BlackPixel(display, screen);
    Window root = RootWindow(display, screen);
    Window window = XCreateSimpleWindow(display, root, 0, 0, width, height, 1, white, black);
    XSelectInput(display, window, ExposureMask | KeyPressMask);
    XMapWindow(display, window);

    Visual* visual = DefaultVisual(display, 0);
    XImage* ximage = create_image(display, visual, data, width, height);
    GC gc = DefaultGC(display, screen);
    XEvent event;
    KeyCode escape = XKeysymToKeycode(display, XK_Escape);
    Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete_window, 1);

    while (1) {
        XNextEvent(display, &event);
        if (event.type == Expose) {
            XPutImage(display, window, gc, ximage, 0, 0, 0, 0, width, height);
        }
        if (event.type == KeyPress && event.xkey.keycode == escape) {
            break;
        }
        if (event.type == ClientMessage && (Atom)event.xclient.data.l[0] == wm_delete_window) {
            break;
        }
    }

    delete[] data;
    XDestroyImage(ximage);
    XCloseDisplay(display);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <algorithm>
using namespace std;

#define ERROR(fmt, ...) do { fprintf(stderr, "%s:%d %s(), error: " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); exit(1); } while (0)


const int ZZ[64] = {      // zigzag order index
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

double DCT_COS[8][8] = {};
double DCT_FACTOR[8] = {};

void initialize_table()
{
    for (int x = 0; x < 8; x++) {
        for (int u = 0; u < 8; u++) {
            DCT_COS[x][u] = cos((2 * x + 1) * u * M_PI / 16);
        }
    }

    DCT_FACTOR[0] = 1 / sqrt(2);
    for (int i = 1; i < 8; i++) {
        DCT_FACTOR[i] = 1;
    }
}

//--------------
struct node {
    node*   ch[2];
    uint8_t val;
    bool is_leaf() { return !ch[0] && !ch[1]; }
};

void destroy_huffman_tree(node* p) {
    if (!p) return;
    if (p->ch[0]) destroy_huffman_tree(p->ch[0]);
    if (p->ch[1]) destroy_huffman_tree(p->ch[1]);
    delete p;
}

void huffman_tree_insert(node* root, int code, int bitsize, int val)
{
    node* p = root;
    for (int i = bitsize - 1; i >= 0; i--) {
        int x = (code >> i) & 1;
        if (!p->ch[x])
            p->ch[x] = new node{NULL, NULL, 0};
        p = p->ch[x];
    }
    p->val = val;
}


struct context {
    FILE* fp;               // input
    int _byte;              // bitstream buffer
    int _cnt;               // number of valid bits

    uint8_t q_table[4][64]; // quantization table [table id][element]
    node* huffman_DC[2];    // DC huffman tree [id]
    node* huffman_AC[2];    // AC huffman tree [id]
    int prev_dc_val[3];     // previous block DC value [component]

    int width;              // image width
    int height;             // image height
    int mcu_w;              // MCU width
    int mcu_h;              // MCU height
    int mcu_n;              // number of total MCUs
    int comp_n;             // number of components: 1 or 3
    struct {
        int h;              // horizontal sampling factor
        int v;              // vertical sampling factor
        int tq;             // quantization table id: 0-3
        int td;             // DC huffman table id: 0-1
        int ta;             // AC huffman table id: 0-1
        int16_t* coeff;     // DCT coefficients
    } comp[3];              // components 0:Y, 1:Cb, 2:Cr
};

int get_pos(context* ctx)
{
    return ftell(ctx->fp);
}

void skip_byte(context* ctx, int len)
{
    fseek(ctx->fp, len, SEEK_CUR);
}

int get_1byte(context* ctx)
{
    return fgetc(ctx->fp);
}

int get_2byte(context* ctx)
{
    int a = get_1byte(ctx);
    int b = get_1byte(ctx);
    return (a << 8) + b;
}

void read_bytes_into(context* ctx, int len, uint8_t* buf)
{
    fread(buf, len, 1, ctx->fp);
}

int nextbit(context* ctx)
{
    if (ctx->_cnt == 0) {
        ctx->_byte = get_1byte(ctx);
        ctx->_cnt = 8;
        if (ctx->_byte == 0xff) {
            int x = get_1byte(ctx);
            if (x != 0) {           // not stuffing byte
                ERROR("found marker 0x%02x in compressed image data", x);
            }
        }
    }
    ctx->_cnt--;
    return (ctx->_byte >> ctx->_cnt) & 1;
}

void clearbit(context* ctx)
{
    ctx->_byte = 0;
    ctx->_cnt = 0;
}

int get_marker(context* ctx)
{
    int marker = get_1byte(ctx);
    if (marker != 0xff) {
        ERROR("expected marker 0xff, but got 0x%02x", marker);
    }
    while (marker == 0xff)          // optional fill bytes
        marker = get_1byte(ctx);
    return marker;
}

void parse_JFIF(context* ctx, int /* len */)
{
    int version = get_2byte(ctx);       // version  (e.g. 0x0102 => 1.02)
    int units = get_1byte(ctx);         // units for H/V densities (0:unspecified, 1:dot/inch, 2:dot/cm)
    int h_density = get_2byte(ctx);     // horizontal pixel density
    int v_density = get_2byte(ctx);     // vertical pixel density
    int h_thumbnail = get_1byte(ctx);   // thumbnail width (may be 0)
    int v_thumbnail = get_1byte(ctx);   // thumbnail height (may be 0)
    skip_byte(ctx, h_thumbnail * v_thumbnail * 3);  // thumbnail 24bit RGB values (optional)

    printf("JFIF ver:%04x\n", version);
    printf("  density unit:%d (0:x,1:dpi,2:dpcm), H:%d, V:%d\n", units, h_density, v_density);
    printf("  thumbnail width:%d, height:%d\n", h_thumbnail, v_thumbnail);
}

void parse_JFXX(context* ctx, int len)
{
    printf("JFIF extension\n");
    skip_byte(ctx, len);                // thumbnail
}

void parse_APP0(context* ctx)
{
    printf("[APP0]\n");
    int len = get_2byte(ctx) - 2;
    uint8_t buf[5];
    read_bytes_into(ctx, 5, buf);

    if (memcmp(buf, "JFIF\x00", 5) == 0) { parse_JFIF(ctx, len - 5); return; }
    if (memcmp(buf, "JFXX\x00", 5) == 0) { parse_JFXX(ctx, len - 5); return; }

    ERROR("not JFIF");
}

// Quantization table
void parse_DQT(context* ctx)
{
    printf("[DQT]\n");
    int len = get_2byte(ctx) - 2;
    int table_count = len / 65;
    for (int i = 0; i < table_count; i++) {
        int x = get_1byte(ctx);
        int Pq = x >> 4;                        // precision: 0(=8bit)
        int Tq = x & 0xf;                       // table id: 0-3
        assert(Pq == 0);
        for (int n = 0; n < 64; n++)
            ctx->q_table[Tq][ZZ[n]] = get_1byte(ctx);   // element (zigzag ordering)
        printf("  table:%d\n", Tq);
    }
}

// Frame header (Baseline DCT)
void parse_SOF0(context* ctx)
{
    printf("[SOF0]\n");
    get_2byte(ctx);                             // header length
    get_1byte(ctx);                             // precision: 8
    ctx->height = get_2byte(ctx);               // number of lines
    ctx->width = get_2byte(ctx);                // number of samples per line
    ctx->comp_n = get_1byte(ctx);               // number of image components: 1-255 (JFIF : 1 or 3)

    printf("  width:%d, height:%d, comp_n:%d\n", ctx->width, ctx->height, ctx->comp_n);
    if (ctx->comp_n != 1 && ctx->comp_n != 3) {
        ERROR("comp_n %d not supported", ctx->comp_n);
    }

    int hmax = 1;
    int vmax = 1;
    for (int i = 0; i < ctx->comp_n; i++) {
        get_1byte(ctx);                                 // component id: 0-255 (JFIF : Y=1,Cb=2,Cr=3)
        int hv = get_1byte(ctx);
        int h  = ctx->comp[i].h  = hv >> 4;             // horizontal sampling factor: 1-4
        int v  = ctx->comp[i].v  = hv & 0xf;            // vertical sampling factor: 1-4
        int tq = ctx->comp[i].tq = get_1byte(ctx);      // quantization table id: 0-3
        hmax = max(hmax, h);
        vmax = max(vmax, v);
        printf("  component i:%d, hv:%02x, q_table:%d\n", i, hv, tq);
    }

    ctx->mcu_w = 8 * hmax;
    ctx->mcu_h = 8 * vmax;
    int n1 = (ctx->width + (ctx->mcu_w - 1)) / ctx->mcu_w;
    int n2 = (ctx->height + (ctx->mcu_h - 1)) / ctx->mcu_h;
    ctx->mcu_n = n1 * n2;

    for (int i = 0; i < ctx->comp_n; i++) {
        ctx->comp[i].coeff = new int16_t[64 * ctx->comp[i].h * ctx->comp[i].v * ctx->mcu_n]{};       // initialized
    }
}

void create_huffman_tree(context* ctx)
{
    int x = get_1byte(ctx);
    int Tc = x >> 4;                // table class: 0(=DC table), 1(=AC table)
    int Th = x & 0xf;               // huffman table id: 0-1
    printf("  class:%d (0:DC,1:AC), table:%d\n", Tc, Th);

    int L[16];
    for (int i = 0; i < 16; i++)
        L[i] = get_1byte(ctx);      // number of huffman codes of length i

    node*& root = (Tc == 0) ? ctx->huffman_DC[Th] : ctx->huffman_AC[Th];
    root = new node{NULL, NULL, 0};

    int code = 0;
    for (int size = 0; size < 16; size++) {
        code <<= 1;
        for (int n = 0; n < L[size]; n++) {
            int val = get_1byte(ctx);       // value associated with each huffman code
            huffman_tree_insert(root, code, size + 1, val);
            code++;
        }
    }
}

// Huffman table
void parse_DHT(context* ctx)
{
    printf("[DHT]\n");
    int len  = get_2byte(ctx);
    int read = 2;
    int prev = get_pos(ctx);

    // one or more huffman table defined
    while (read < len) {
        create_huffman_tree(ctx);
        int cur = get_pos(ctx);
        read += cur - prev;
        prev = cur;
    }
    assert(read == len);
}

int decode_huffman_tree(context* ctx, node* root)
{
    node* p = root;
    while (!p->is_leaf()) {
        p = p->ch[nextbit(ctx)];
    }
    return p->val;
}

// read next n bits from stream
int receive(context* ctx, int size)
{
    int x = 0;
    for (int i = 0; i < size; i++) {
        x = (x << 1) | nextbit(ctx);
    }
    return x;
}

// convert to equivalent signed value
int extend(int v, int size) {
    if ((v >> (size - 1)) ^ 1)      // MSB == '0'
        v -= (1 << size) - 1;
    return v;
}

void load_huffman_DC(context* ctx, int* blk, int comp_idx) {
    int table_id = ctx->comp[comp_idx].td;
    int size = decode_huffman_tree(ctx, ctx->huffman_DC[table_id]);
    int x = receive(ctx, size);
    x = extend(x, size);

    x += ctx->prev_dc_val[comp_idx];
    ctx->prev_dc_val[comp_idx] = x;
    blk[0] = x;
}

void load_huffman_AC(context* ctx, int* blk, int comp_idx) {
    int table_id = ctx->comp[comp_idx].ta;
    int i = 1;
    while (i < 64) {
        int val = decode_huffman_tree(ctx, ctx->huffman_AC[table_id]);
        int run = (val >> 4) & 0xf;                 // run-length of zero
        int size = val & 0xf;

        if (size == 0) {
            if (run ==  0) { return; }              // EOB
            if (run == 15) { i += 16; continue; }   // ZRL
            assert(false);                          // not come here
        }
        i += run;

        int x = receive(ctx, size);
        x = extend(x, size);

        blk[ZZ[i++]] = x;
    }
}

void dequantization(int* blk, uint8_t* qtable) {
    for (int i = 0; i < 64; i++) {
        blk[i] *= qtable[i];
    }
}

int clamp(int x) { return max(0, min(255, x)); }

void inverseDCT(int* blk)
{
    double* A = DCT_FACTOR;
    double tbl[64] = {};
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            for (int u = 0; u < 8; u++) {
                for (int v = 0; v < 8; v++) {
                    double val = A[u] * A[v] * blk[v * 8 + u] * DCT_COS[x][u] * DCT_COS[y][v];
                    tbl[y * 8 + x] += val;
                }
            }
            tbl[y * 8 + x] /= 4;
        }
    }
    for (int i = 0; i < 64; i++) {
        blk[i] = clamp(round(tbl[i]) + 128);
    }
}

void copy_to_mcu(context* ctx, int comp_i, int blk_i, int* data, int* out) {
    int H = ctx->comp[comp_i].h;
    out += (blk_i % H) * 8;         // x offset
    out += (blk_i / H) * (64 * H);  // y offset

    for (int y = 0; y < 8; y++) {
        for (int x = 0 ; x < 8; x++) {
            out[y * ctx->mcu_w + x] = data[y * 8 + x];
        }
    }
}

// nearest-neighbor
void upsample(context *ctx, int comp_i, int* out) {
    int H = ctx->mcu_w / 8 / ctx->comp[comp_i].h;
    int V = ctx->mcu_h / 8 / ctx->comp[comp_i].v;
    if (H == 1 && V == 1)
        return;

    for (int y = ctx->mcu_h - 1; y >= 0; y--) {
        for (int x = ctx->mcu_w - 1; x >= 0; x--) {
            out[(y * ctx->mcu_w) + x] = out[(y / V * ctx->mcu_w) + x / H];
        }
    }
}

void load_DCT_coeff(context* ctx) {
    for (int mcu_i = 0; mcu_i < ctx->mcu_n; mcu_i++) {
        for (int comp_i = 0; comp_i < ctx->comp_n; comp_i++) {

            int block_n = ctx->comp[comp_i].h * ctx->comp[comp_i].v;
            int16_t* out = ctx->comp[comp_i].coeff + mcu_i * 64 * block_n;

            for (int blk_i = 0; blk_i < block_n; blk_i++) {
                int data[64] = {0};

                load_huffman_DC(ctx, data, comp_i);
                load_huffman_AC(ctx, data, comp_i);
                dequantization(data, ctx->q_table[ctx->comp[comp_i].tq]);

                copy(data, data + 64, out);
                out += 64;
            }
        }
    }
    clearbit(ctx);
}

void YCbCr_to_RGB(uint8_t* in, uint8_t* out)
{
    uint8_t Y  = in[0];
    uint8_t Cb = in[1];
    uint8_t Cr = in[2];
    out[0] = clamp(round(Y                       + 1.402  * (Cr - 128)));   // R
    out[1] = clamp(round(Y - 0.3441 * (Cb - 128) - 0.7141 * (Cr - 128)));   // G
    out[2] = clamp(round(Y + 1.772  * (Cb - 128)                      ));   // B
}

void output_to_pixel(context* ctx, int mcu_i, int data[3][32*32], uint8_t* pixel)
{
    uint8_t in[3];

    int hmcu_n = (ctx->width + ctx->mcu_w - 1) / ctx->mcu_w;    // number of MCUs in horizontal direction
    int mcu_x = mcu_i % hmcu_n;
    int mcu_y = mcu_i / hmcu_n;

    for (int y = 0; y < ctx->mcu_h; y++) {

        uint8_t* out = pixel + ((mcu_y * ctx->mcu_h + y) * ctx->width + mcu_x * ctx->mcu_w) * ctx->comp_n;

        if (mcu_y * ctx->mcu_h + y >= ctx->height) break;

        for (int x = 0; x < ctx->mcu_w; x++) {

            if (mcu_x * ctx->mcu_w + x >= ctx->width) break;

            if (ctx->comp_n == 1) {
                *out = data[0][ctx->mcu_w * y + x];
                out += 1;
            } else {    // comp_n == 3
                in[0] = data[0][ctx->mcu_w * y + x];
                in[1] = data[1][ctx->mcu_w * y + x];
                in[2] = data[2][ctx->mcu_w * y + x];

                YCbCr_to_RGB(in, out);
                out += 3;
            }
        }
    }
}

// Scan header
void parse_SOS(context* ctx) {
    printf("[SOS]\n");
    get_2byte(ctx);                         // header length
    uint8_t Ns = get_1byte(ctx);            // number of image components in scan: 1-4
    assert(Ns == 1 || Ns == 3);
    for (int i = 0; i < Ns; i++) {
        get_1byte(ctx);                     // component id
        uint8_t T = get_1byte(ctx);
        ctx->comp[i].td = T >> 4;           // DC huffman table id: 0-1
        ctx->comp[i].ta = T & 0xf;          // AC huffman table id: 0-1
        printf("  component i:%d, huff_table DC:%d, AC:%d\n", i, T >> 4, T & 0xf);
    }
                                            // (for progressive/lossless?)
    get_1byte(ctx);                         // start of spectral
    get_1byte(ctx);                         // end of spectral selection
    get_1byte(ctx);                         // successive approximation bit position

    load_DCT_coeff(ctx);
}

void parse_jpeg(context* ctx)
{
    int signature = get_2byte(ctx);
    if (signature != 0xffd8) {
        ERROR("not jpeg file");
    }
    while (1) {
        int marker = get_marker(ctx);
        switch (marker) {
            case 0xe0:
                parse_APP0(ctx);
                break;
            case 0xdb:
                parse_DQT(ctx);
                break;
            case 0xc0:
                parse_SOF0(ctx);
                break;
            case 0xc4:
                parse_DHT(ctx);
                break;
            case 0xda:
                parse_SOS(ctx);
                break;
            case 0xd9:
                printf("[EOI]\n");
                return;
            default:
                if (0xe0 <= marker && marker <= 0xffef) {
                    printf("[APP%d]\n", marker & 0xf);
                    int len = get_2byte(ctx) - 2;
                    skip_byte(ctx, len);
                    break;
                }
                ERROR("unkown marker 0x%02x", marker);
        }
    }
}

uint8_t* convert_to_pixel_data(context* ctx)
{
    int data[3][32*32];
    int buf[64];

    int size = ctx->width * ctx->height * ctx->comp_n;
    uint8_t* pixels = new uint8_t[size];

    for (int mcu_i = 0; mcu_i < ctx->mcu_n; mcu_i++) {
        for (int comp_i = 0; comp_i < ctx->comp_n; comp_i++) {

            int block_n = ctx->comp[comp_i].h * ctx->comp[comp_i].v;
            int16_t* p = ctx->comp[comp_i].coeff + mcu_i * 64 * block_n;

            for (int blk_i = 0; blk_i < block_n; blk_i++) {

                copy(p, p + 64, buf);
                inverseDCT(buf);

                copy_to_mcu(ctx, comp_i, blk_i, buf, data[comp_i]);
                p += 64;
            }
            upsample(ctx, comp_i, data[comp_i]);
        }
        output_to_pixel(ctx, mcu_i, data, pixels);
    }
    return pixels;
}

uint8_t* decode_jpeg(context* ctx)
{
    parse_jpeg(ctx);                    // decode entropy coded segments
    return convert_to_pixel_data(ctx);
}

uint8_t* load_jpeg(const char* filename, int* width, int* height, int* channel)
{
    initialize_table();
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        ERROR("%s", strerror(errno));
    }
    context ctx = {};
    ctx.fp = fp;

    uint8_t* data = decode_jpeg(&ctx);
    *width   = ctx.width;
    *height  = ctx.height;
    *channel = ctx.comp_n;

    fclose(fp);
    for (int i = 0; i < 2; i++) {
        destroy_huffman_tree(ctx.huffman_DC[i]);
        destroy_huffman_tree(ctx.huffman_AC[i]);
    }
    for (int i = 0; i < ctx.comp_n; i++) {
        delete[] ctx.comp[i].coeff;
    }
    return data;
}

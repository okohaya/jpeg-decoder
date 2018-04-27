#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <algorithm>
using namespace std;

#define ERROR(fmt, ...) fprintf(stderr, "%s:%d %s(), error: " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)


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

    int x;                  // image width
    int y;                  // image height
    int mcu_x;              // MCU width: 8
    int mcu_y;              // MCU height: 8
    int h_mcus;             // number of MCUs in horizontal direction
    int v_mcus;             // number of MCUs in vertical direction
    int comp_n;             // number of components: 3
    struct {
        int id;             // component id
        int h;              // horizontal sampling factor
        int v;              // vertical sampling factor
        int tq;             // quantization table id: 0-3
        int td;             // DC huffman table id: 0-1
        int ta;             // AC huffman table id: 0-1
        uint8_t* data;
    } comp[3];              // components
};

uint8_t* decode_jpeg(context* ctx);

uint8_t* load_jpeg(const char* filename, int* width, int* height, int* channel)
{
    initialize_table();
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        ERROR("%s", strerror(errno));
        exit(1);
    }
    context ctx = {};
    ctx.fp = fp;

    uint8_t* data = decode_jpeg(&ctx);
    *width = ctx.x;
    *height = ctx.y;
    *channel = ctx.comp_n;

    fclose(fp);
    for (int i = 0; i < 2; i++) {
        destroy_huffman_tree(ctx.huffman_DC[i]);
        destroy_huffman_tree(ctx.huffman_AC[i]);
    }
    for (int i = 0; i < ctx.comp_n; i++)
        delete[] ctx.comp[i].data;
    return data;
}

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
                exit(1);
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
        exit(1);
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
    exit(1);
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
    ctx->y = get_2byte(ctx);                    // number of lines
    ctx->x = get_2byte(ctx);                    // number of samples per line
    ctx->comp_n = get_1byte(ctx);               // number of image components: 1-255

    printf("  width:%d, height:%d, comp_n:%d\n", ctx->x, ctx->y, ctx->comp_n);
    if (ctx->comp_n != 3) {
        ERROR("format not supported");
        exit(1);
    }

    int hmax = 1;
    int vmax = 1;
    for (int i = 0; i < ctx->comp_n; i++) {
        ctx->comp[i].id = get_1byte(ctx);       // component id: 0-255
        int hv = get_1byte(ctx);
        ctx->comp[i].h  = hv >> 4;              // horizontal sampling factor: 1-4
        ctx->comp[i].v  = hv & 0xf;             // vertical sampling factor: 1-4
        ctx->comp[i].tq = get_1byte(ctx);       // quantization table id: 0-3
        hmax = max(hmax, ctx->comp[i].h);
        vmax = max(vmax, ctx->comp[i].v);
        if (hv != 0x11) {
            ERROR("subsampling not supported (hv:%02x)", hv);
            exit(1);
        }
        printf("  component:%d, hv:%02x, q_table:%d\n", ctx->comp[i].id, hv, ctx->comp[i].tq);
    }

    ctx->mcu_x = 8 * hmax;
    ctx->mcu_y = 8 * vmax;
    ctx->h_mcus = (ctx->x + (ctx->mcu_x - 1)) / ctx->mcu_x;
    ctx->v_mcus = (ctx->y + (ctx->mcu_y - 1)) / ctx->mcu_y;
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

void decode_block(context* ctx, int mcu_idx) {
    int data[64];
    int offset = 64 * mcu_idx;

    for (int i = 0; i < ctx->comp_n; i++) {
        memset(data, 0, sizeof(data));

        load_huffman_DC(ctx, data, i);
        load_huffman_AC(ctx, data, i);
        dequantization(data, ctx->q_table[ctx->comp[i].tq]);
        inverseDCT(data);

        uint8_t* p = ctx->comp[i].data + offset;
        copy(data, data+64, p);
    }
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

void output_to_pixel(context* ctx, uint8_t* out)
{
    uint8_t in[3];

    for (int y = 0; y < ctx->y; y++) {
        for (int x = 0; x < ctx->x; x++) {
            int mcu_idx = (y / ctx->mcu_y * ctx->h_mcus) + (x / ctx->mcu_x);
            int offset = ((y % ctx->mcu_y) * ctx->mcu_x) + (x % ctx->mcu_x);
            int pos = (ctx->mcu_x * ctx->mcu_y) * mcu_idx + offset;

            in[0] = ctx->comp[0].data[pos];
            in[1] = ctx->comp[1].data[pos];
            in[2] = ctx->comp[2].data[pos];

            YCbCr_to_RGB(in, out);
            out += 3;
        }
    }
}

// Scan header
uint8_t* parse_SOS(context* ctx) {
    printf("[SOS]\n");
    get_2byte(ctx);                         // header length
    uint8_t Ns = get_1byte(ctx);            // number of image components in scan: 1-4
    for (int i = 0; i < Ns; i++) {
        uint8_t Cs = get_1byte(ctx);        // component id: 0-255
        uint8_t T = get_1byte(ctx);
        ctx->comp[Cs - 1].td = T >> 4;      // DC huffman table id: 0-1
        ctx->comp[Cs - 1].ta = T & 0xf;     // AC huffman table id: 0-1
        printf("  component:%d, huff_table DC:%d, AC:%d\n", Cs, T >> 4, T & 0xf);
    }
                                            // (for progressive/lossless?)
    get_1byte(ctx);                         // start of spectral
    get_1byte(ctx);                         // end of spectral selection
    get_1byte(ctx);                         // successive approximation bit position

    for (int i = 0; i < ctx->comp_n; i++) {
        int size = (ctx->mcu_x * ctx->h_mcus) * (ctx->mcu_y * ctx->v_mcus);
        ctx->comp[i].data = new uint8_t[size];
    }

    for (int i = 0; i < ctx->h_mcus * ctx->v_mcus; i++) {
        decode_block(ctx, i);
    }

    clearbit(ctx);
    int marker = get_marker(ctx);
    if (marker != 0xd9) {       // EOI
        ERROR("unsupported marker %02x", marker);
        exit(1);
    }

    int size = ctx->x * ctx->y * 3;     // RGB
    uint8_t* pixel = new uint8_t[size];
    output_to_pixel(ctx, pixel);

    return pixel;
}

uint8_t* decode_jpeg(context* ctx)
{
    int signature = get_2byte(ctx);
    if (signature != 0xffd8) {
        ERROR("not jpeg file");
        exit(1);
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
                return parse_SOS(ctx);
            default:
                if (0xe0 <= marker && marker <= 0xffef) {
                    printf("[APP%d]\n", marker & 0xf);
                    int len = get_2byte(ctx) - 2;
                    skip_byte(ctx, len);
                    break;
                }
                ERROR("unkown marker 0x%02x", marker);
                exit(1);
        }
    }
    return 0;
}

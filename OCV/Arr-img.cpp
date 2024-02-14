#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Windows.h>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;
using namespace samples;

const int hist_size = 256;
const int interval_half = 5; // интервал сглаживани€
const int delta = 12; // отличие от экстремума
const int base_frame_size = 8; // полуширина рамки в пиксел€х
const int delta_Q = 1; // дельта из статьи
const int sigma_multiplier = 4; // ћножитель дл€ искусственного увеличени€ дисперсии
const int border_R_mult = 90;
const int BUF_PREV_COEFF = 2 / 3;
const int BLACK = 0;
const int WHITE = 255;
const int COL1 = 100;
const int COL2 = 155;
const char ANG0 = '-';
const char ANG45 = '/';
const char ANG90 = '|';
const char ANG135 = '\\';
string name = "sharp_cells.jpg";
string path = "C:/OCV/IMG/";

void adv_spread(int** clusters, int** R_img, int height, int width, int max_col);
Mat arr_to_3ch_mat(Mat ch0, int** ch1, int** ch2 = NULL);
int** buf_to_clust(int*** buf, int** clusters, int height, int width);
Mat clust_cent_R(Mat img, int frame_size, bool need_thinning = FALSE);
void clust_color(int** img, int top, int bot, int left, int right, int col, int col2);
int** clust_form(int** centers, int** R_img, char** border_dir_mat, int height, int width, int* max_col_p);
void fill_buf(int** centers, int** R_img, char** border_dir_mat, int*** buf, int height, int width);
int** find_peak(int** frame_clust, int** frame_R, char dir, int frame_size, int col);
float get_R(Mat img, char& border_dir, int frame_size = base_frame_size);
Mat get_R_img(Mat img, int frame_size = base_frame_size, char** border_dir_mat = NULL);
Mat get_R_multiplex(Mat img, int* L_arr, int size_L_arr, char** border_dir_mat = NULL);
void get_tblr(int** img, int* top, int* bot, int* left, int* right, int height, int width, int col);
int** mat_to_arr(Mat img);
Mat mult_R(Mat img1, Mat img2);
int*** process_center(int** R_img, int*** buf, int i_c, int j_c, char dir);
void process_clust(int** clusters, int** R_img, int top, int left, int right, int bot, int col, int height, int width);
int** spread(int** img, int height, int width);

int main(int argc, char** argv) //C:\OCV\OCV\x64\Debug\OCV.exe
{
    Mat gray_sys, img;
    addSamplesDataSearchPath(path);
    img = imread(findFile(name), IMREAD_COLOR);
    cvtColor(img, gray_sys, COLOR_BGR2GRAY);

    int width = img.cols;
    int height = img.rows;
    char** border_dir_mat = new char* [height];
    for (int i = 0; i < height; i++) {
        border_dir_mat[i] = new char [width];
    }
    int*** buf_spread = new int** [height];
    for (int i = 0; i < height; i++) {
        buf_spread[i] = new int*[width];
        for (int j = 0; j < width; j++) {
            buf_spread[i][j] = new int[3];
            for (int k = 0; k < 3; k++) {
                buf_spread[i][j][k] = 0;
            }
        }
    }
    int L_arr[10] = { base_frame_size, 5, 10, 32, 6, 7, 8, 9, 10 };
    Mat R_mult = get_R_multiplex(gray_sys, L_arr, 2, border_dir_mat);
    Mat centers, temp;
    Mat blank = Mat(height, width, CV_8UC3, WHITE);
    int** arr_centers;
    int** arr_R;
    int** clusters = new int* [height];
    for (int i = 0; i < height; i++) {
        clusters[i] = new int[width];
    }
    int** gray;
    int** in_cl;
    int* max_col_p = new int;
    int N_epoch = 1001;
    *max_col_p = 0;
    imshow("R mult", R_mult);
    imwrite(path + "R_img_" + name, R_mult);
    centers = clust_cent_R(R_mult, L_arr[0], false);
    //imshow("cluster centers", centers);
    arr_centers = mat_to_arr(centers);
    arr_R = mat_to_arr(R_mult);
    gray = mat_to_arr(gray_sys);
    /*in_cl = clust_form(arr_centers, arr_R, border_dir_mat, height, width, max_col_p);
    imshow("clusters", arr_to_3ch_mat(gray_sys, in_cl)); 
    imwrite(path + "cl_init_" + name, arr_to_3ch_mat(gray_sys, in_cl));
    
    for (int i = 0; i < N_epoch; i++){
        spread(clusters, height, width);
        //cout << "spread\n";
        adv_spread(clusters, arr_R, height, width, *max_col_p);
        //adv_spread(clusters, gray, height, width, *max_col_p);
        //cout << "adv spread\n";
        if (i % 50 == 0) { 
            temp = arr_to_3ch_mat(gray_sys, clusters);
            imshow(format("iter %i", i), temp); 
            imwrite(path + "iter_" + to_string(i) + "_L_" + to_string(base_frame_size) + "_sigma_x_" + to_string(sigma_multiplier) + "_" + name, temp);
        }
    }*/

    fill_buf(arr_centers, arr_R, border_dir_mat, buf_spread, height, width);
    buf_to_clust(buf_spread, clusters, height, width);
    spread(clusters, height, width);
    imshow("clusters", arr_to_3ch_mat(gray_sys, clusters));
    imwrite(path + "cl_new_" + name, arr_to_3ch_mat(R_mult, clusters));
    waitKey(0);
    for (int i = 0; i < height; i++) {
        delete[width]border_dir_mat[i];
    }
    delete[height]border_dir_mat;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            delete[3]buf_spread[i][j];
        }
        delete[width]buf_spread[i];
    }
    delete[height]buf_spread;
    delete max_col_p;
    return 0;
}

void adv_spread(int** clusters, int** R_img, int height, int width, int max_col){
    int top = height - 1, left = width - 1, right = 0, bot = 0;
    for (int col_ctr = 2; col_ctr <= max_col; col_ctr++) {
        get_tblr(clusters, &top, &bot, &left, &right, height, width, col_ctr);
        top -= (top != 0);
        left -= (left != 0);
        bot += (bot != height - 1);
        right += (right != width - 1);
        process_clust(clusters, R_img, top, left, right, bot, col_ctr, height, width);
        //cout << "\nMAX COL = " << max_col << "\n";
        top = height - 1;
        left = width - 1;
        right = 0;
        bot = 0;
    }
}

Mat arr_to_3ch_mat(Mat ch0, int** ch1, int** ch2) {
    int width = ch0.cols;
    int height = ch0.rows;
    Mat res = Mat(height, width, CV_8UC3, BLACK);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            res.at<Vec3b>(i, j)[0] = ch0.at<uint8_t>(i, j);
            res.at<Vec3b>(i, j)[1] = ch1[i][j];
            if (ch2 != NULL) { res.at<Vec3b>(i, j)[2] = ch2[i][j]; }
            else { res.at<Vec3b>(i, j)[2] = ch1[i][j]; }
            //cout << format("elem %i %i\n", i, j);
        }
    }
    return res;
}

int** buf_to_clust(int*** buf, int** clusters, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            clusters[i][j] = BLACK;
            if (buf[i][j][0] + buf[i][j][1] + buf[i][j][2] > 0) {
                if (buf[i][j][1] > BUF_PREV_COEFF * (buf[i][j][0] * (buf[i][j][0] > buf[i][j][2]) + buf[i][j][2] * (buf[i][j][2] > buf[i][j][0]))) {
                    clusters[i][j] = COL1;
                }
                if (buf[i][j][2] > BUF_PREV_COEFF * (buf[i][j][0] * (buf[i][j][0] > buf[i][j][1]) + buf[i][j][1] * (buf[i][j][1] > buf[i][j][0]))) {
                    clusters[i][j] = COL2;
                }
            }
        }
    }
    return clusters;
}

Mat clust_cent_R(Mat img, int frame_size, bool need_thinning) {
    int width = img.cols;
    int height = img.rows;
    Mat clust = Mat(height, width, CV_8UC1, 0.0);
    bool ismax = TRUE, pastmax = FALSE;
    int no_max_left = 2 * frame_size, no_max_right = 2 * frame_size, no_max_up = 2 * frame_size, no_max_down = 2 * frame_size;

    for (int i = frame_size; i < height - frame_size; i++) {
        for (int j = frame_size; j < width - frame_size; j++) {
            ismax = TRUE;
            pastmax = FALSE;
            float max = img.at<uint8_t>(i, j);
            if (clust.at<uint8_t>(i, j) != WHITE) {
                if (max < border_R_mult) {
                    clust.at<uint8_t>(i, j) = WHITE;
                    continue;
                }
                clust.at<uint8_t>(i, j) = BLACK;

                if (i < frame_size) { no_max_left = i - 1; }
                else { no_max_left = frame_size; }
                if (j < frame_size) { no_max_down = j - 1; }
                else { no_max_down = frame_size; }
                if (i > height - frame_size - 1) { no_max_right = height - i; }
                else { no_max_right = frame_size; }
                if (j > width - frame_size - 1) { no_max_up = width - j; }
                else { no_max_up = frame_size; }
                //cout << i << ' ' << j << ' ' << no_max_left << ' ' << no_max_down << ' ' << no_max_right << ' ' << no_max_up << "\n";
                for (int f_i = -no_max_left; f_i < no_max_right; f_i++) {
                    for (int f_j = -no_max_down; f_j < no_max_down; f_j++) {
                        if (max < img.at<uint8_t>(i + f_i, j + f_j) && (f_i * f_i + f_j * f_j != 0)) {
                            clust.at<uint8_t>(i, j) = WHITE;
                            ismax = FALSE;
                            break;
                        }
                        if (ismax && pastmax && need_thinning) {
                            clust.at<uint8_t>(i + f_i, j + f_j) = WHITE;
                            clust.at<uint8_t>(i + f_i, j - f_j) = WHITE;
                            clust.at<uint8_t>(i - f_i, j + f_j) = WHITE;
                            clust.at<uint8_t>(i - f_i, j - f_j) = WHITE;
                        }
                        if (f_i == 0 && f_j == 0) { pastmax = TRUE; }
                    }
                    if (!ismax) { break; }
                }
            }
        }

    }

    return clust;
}

void clust_color(int** img, int top, int bot, int left, int right, int col, int col2){
    for (int i = top; i <= bot; i++) {
        for (int j = left; j <= right; j++) {
            if (img[i][j] == col2) {
                img[i][j] = col;
            }
        }
    }
}

int** clust_form(int** centers, int** R_img, char** border_dir_mat, int height, int width, int* max_col_p){
    int col_ctr = 0;
    int** peaks = new int* [height];
    for (int i = 0; i < height; i++) {
        peaks[i] = new int [width];
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            peaks[i][j] = 0;
        }
    }
    int** frame_clust = new int* [2 * base_frame_size + 1];
    for (int i = 0; i < (2 * base_frame_size + 1); i++) {
        frame_clust[i] = new int[2 * base_frame_size + 1];
    }
    int** frame_R = new int* [2 * base_frame_size + 1];
    for (int i = 0; i < (2 * base_frame_size + 1); i++) {
        frame_R[i] = new int[2 * base_frame_size + 1];
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i >= base_frame_size && j >= base_frame_size && i < height - base_frame_size && j < width - base_frame_size){
                if (centers[i][j] == BLACK) {
                    for (int f_i = 0; f_i < 2 * base_frame_size + 1; f_i++) {
                        for (int f_j = 0; f_j < 2 * base_frame_size + 1; f_j++) {
                            frame_R[f_i][f_j] = R_img[i + f_i - base_frame_size][j + f_j - base_frame_size];
                        }
                    }
                    frame_clust = find_peak(frame_clust, frame_R, border_dir_mat[i][j], base_frame_size, col_ctr);
                    col_ctr += 2;
                    for (int f_i = 0; f_i < 2 * base_frame_size + 1; f_i++) {
                        for (int f_j = 0; f_j < 2 * base_frame_size + 1; f_j++) {
                            peaks[i + f_i - base_frame_size][j + f_j - base_frame_size] = frame_clust[f_i][f_j];
                        }
                    }
                }
            }
            else { peaks[i][j] = 0; }
        }
    }

    for (int i = 0; i < (2 * base_frame_size + 1); i++) {
        delete[2 * base_frame_size + 1]frame_clust[i];
    }
    delete[2 * base_frame_size + 1] frame_clust;
    for (int i = 0; i < (2 * base_frame_size + 1); i++) {
        delete[2 * base_frame_size + 1]frame_R[i];
    }
    delete[2 * base_frame_size + 1] frame_R;
    *max_col_p = col_ctr + 1;
    return peaks;
}

void fill_buf(int** centers, int** R_img, char** border_dir_mat, int*** buf, int height, int width){
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (centers[i][j] == BLACK && !(i == base_frame_size || i == height - base_frame_size || j == base_frame_size || j == width - base_frame_size)){
                process_center(R_img, buf, i, j, border_dir_mat[i][j]);
            }
        }
    }
}

int** find_peak(int** frame_clust, int** frame_R, char dir, int frame_size, int col){
    float max_R1 = 0, max_R2 = 0, B1 = 0, sigma1 = 0, B2 = 0, sigma2 = 0;
    int m_i1 = 0, m_j1 = 0, m_i2 = 0, m_j2 = 0;
    for (int i = 0; i < 2 * frame_size + 1; i++) {
        for (int j = 0; j < 2 * frame_size + 1; j++) {
            if (dir == ANG90) {
                if (j < frame_size + 1) {
                    if (max_R1 < frame_R[i][j]) {
                        max_R1 = frame_R[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame_R[i][j];
                }
                else {
                    if (max_R2 < frame_R[i][j]) {
                        max_R2 = frame_R[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame_R[i][j];
                }
            }
            if (dir == ANG135) {
                if (i < 2*frame_size + 1 - j){
                    if (max_R1 < frame_R[i][j]) {
                        max_R1 = frame_R[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame_R[i][j];
                }
                else{
                    if (max_R2 < frame_R[i][j]) {
                        max_R2 = frame_R[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame_R[i][j];
                }
            }
            if (dir == ANG0) {
                if (i < frame_size + 1) {
                    if (max_R1 < frame_R[i][j]) {
                        max_R1 = frame_R[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame_R[i][j];
                }
                else {
                    if (max_R2 < frame_R[i][j]) {
                        max_R2 = frame_R[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame_R[i][j];
                }
            }
            if (dir == ANG45) {
                if (i < j) {
                    if (max_R1 < frame_R[i][j]) {
                        max_R1 = frame_R[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame_R[i][j];
                }
                else {
                    if (max_R2 < frame_R[i][j]) {
                        max_R2 = frame_R[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame_R[i][j];
                }
            }
        }
    }

    B1 = 2 * B1 / (2 * frame_size + 1) / (2 * frame_size + 1);
    B2 = 2 * B2 / (2 * frame_size + 1) / (2 * frame_size + 1);

    for (int i = 0; i < 2 * frame_size + 1; i++) {
        for (int j = 0; j < 2 * frame_size + 1; j++) {
            if (dir == ANG90) {
                if (j < frame_size + 1) {
                    sigma1 = sigma1 + (frame_R[i][j] - B1) * (frame_R[i][j] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame_R[i][j] - B2) * (frame_R[i][j] - B2);
                }
            }
            if (dir == ANG135) {
                if (i < 2 * frame_size + 1 - j) {
                    sigma1 = sigma1 + (frame_R[i][j] - B1) * (frame_R[i][j] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame_R[i][j] - B2) * (frame_R[i][j] - B2);
                }
            }
            if (dir == ANG0) {
                if (i < frame_size + 1) {
                    sigma1 = sigma1 + (frame_R[i][j] - B1) * (frame_R[i][j] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame_R[i][j] - B2) * (frame_R[i][j] - B2);
                }
            }
            if (dir == ANG45) {
                if (i < j) {
                    sigma1 = sigma1 + (frame_R[i][j] - B1) * (frame_R[i][j] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame_R[i][j] - B2) * (frame_R[i][j] - B2);
                }
            }
        }
    }

    sigma1 = sqrt(sigma1) / (2 * frame_size + 1);
    sigma2 = sqrt(sigma2) / (2 * frame_size + 1);

    for (int i = 0; i < 2 * frame_size + 1; i++) {
        for (int j = 0; j < 2 * frame_size + 1; j++) {
            if (dir == ANG90) {
                if (j < frame_size + 1) {
                    if (frame_R[i][j] >= B1 - sigma1 && frame_R[i][j] <= B1 + sigma1) {
                        frame_clust[i][j] = col;
                    }
                    else { 
                        frame_clust[i][j] = 0;
                    }
                }
                else {
                    if (frame_R[i][j] >= B2 - sigma2 && frame_R[i][j] <= B2 + sigma2) {
                        frame_clust[i][j] = col + 1;
                    }
                    else { 
                        frame_clust[i][j] = 0; 
                    }
                }
            }
            if (dir == ANG135) {
                if (i < 2 * frame_size + 1 - j) {
                    if (frame_R[i][j] >= B1 - sigma1 && frame_R[i][j] <= B1 + sigma1) {
                        frame_clust[i][j] = col;
                    }
                    else { 
                        frame_clust[i][j] = 0; 
                    }
                }
                else {
                    if (frame_R[i][j] >= B2 - sigma2 && frame_R[i][j] <= B2 + sigma2) {
                        frame_clust[i][j] = col + 1;
                    }
                    else { 
                        frame_clust[i][j] = 0; 
                    }
                }
            }
            if (dir == ANG0) {
                if (i < frame_size + 1){
                    if (frame_R[i][j] >= B1 - sigma1 && frame_R[i][j] <= B1 + sigma1) {
                        frame_clust[i][j] = col;
                    }
                    else { 
                        frame_clust[i][j] = 0; 
                    }
                }
                else {
                    if (frame_R[i][j] >= B2 - sigma2 && frame_R[i][j] <= B2 + sigma2) {
                        frame_clust[i][j] = col + 1;
                    }
                    else { 
                        frame_clust[i][j] = 0;
                    }
                }
            }
            if (dir == ANG45) {
                if (i < j) {
                    if (frame_R[i][j] >= B1 - sigma1 && frame_R[i][j] <= B1 + sigma1) {
                        frame_clust[i][j] = col;
                    }
                    else { 
                        frame_clust[i][j] = 0;
                    }
                }
                else {
                    if (frame_R[i][j] >= B2 - sigma2 && frame_R[i][j] <= B2 + sigma2) {
                        frame_clust[i][j] = col + 1;
                    }
                    else { 
                        frame_clust[i][j] = 0;
                    }
                }
            }
        }
    }

    return frame_clust;
}

void fill_holes(int** img, int top, int bot, int left, int right, int col, int height, int width){
    int* tops   = new int [width];
    int* bots   = new int [width];
    int* lefts  = new int [height];
    int* rights = new int [height];

    for (int i = 0; i < height; i++) {
        lefts[i] = width;
        rights[i] = -1;
    }
    for (int j = 0; j < width; j++) {
        tops[j] = height;
        bots[j] = -1;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i < lefts[i] && img[i][j] == col) {
                lefts[i] = j;
            }
            if (i > rights[i] && img[i][j] == col) {
                rights[i] = j;
            }
            if (j < tops[j] && img[i][j] == col) {
                tops[j] = i;
            }
            if (j > bots[i] && img[i][j] == col) {
                bots[j] = i;
            }
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (img[i][j] != col && i >= tops[j] && i <= bots[j] && j >= lefts[i] && j >= rights[i]) {
                img[i][j] = col;
            }
        }
    }
    delete[]tops;
    delete[]bots;
    delete[]lefts;
    delete[]rights;
}

void get_B_sigma(int** clusters, int** R_img, float* B, float* sigma, int top, int bot, int left, int right, int col, int height, int width) {
    int pix_ctr = 0;
    *B = 0;
    *sigma = 0;
    for (int i = top; i <= bot; i++) {
        for (int j = left; j <= right; j++) {
            if (clusters[i][j] == col) {
                *B += R_img[i][j];
                pix_ctr++;
            }
            //else { cout << format("col %i, pix %i", col, clusters[i][j]); }
        }
    }
    if (pix_ctr) {
        *B /= pix_ctr;
        for (int i = top; i <= bot; i++) {
            for (int j = left; j <= right; j++) {
                if (clusters[i][j] == col) {
                    *sigma += (R_img[i][j] - *B) * (R_img[i][j] - *B);
                }
            }
        }
        *sigma = sqrt(*sigma) / pix_ctr;
    }
    else {
        //cout << "\nNO SUCH COLOR COL = " << col << "\n";
        *B = 0;
        *sigma = 0;
    }
}

float get_R(Mat img, char& border_dir, int frame_size) {
    float B_1 = 0, B_2 = 0;
    float s_1 = 0, s_2 = 0;
    float Q, Q_temp;
    border_dir = '0';
    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница --, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i <= frame_size + 1) {
                B_1 += img.at<uint8_t>(i, j);
            }
            if (i >= frame_size + 1) {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница --, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i < frame_size + 1) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            if (i > frame_size + 1) {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2) {
        Q = 2* max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2);
        border_dir = ANG0;
    }
    else { Q = 1000; }

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница /, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i >= j) {
                B_1 += img.at<uint8_t>(i, j);
            }
            if (i <= j) {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница /, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i >= j) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            if (i <= j) {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2) { Q_temp = 2 * max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2); }
    else { Q_temp = 1000; }

    if (Q_temp < Q) {
        Q = Q_temp;
        border_dir = ANG45;
    }

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница |, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (j <= frame_size + 1) {
                B_1 += img.at<uint8_t>(i, j);
            }
            if (j >= frame_size + 1) {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница |, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (j <= frame_size + 1) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            if (j >= frame_size + 1) {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2) { Q_temp = 2 * max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2); }
    else { Q_temp = 1000; }

    if (Q_temp < Q) {
        Q = Q_temp;
        border_dir = ANG90;
    }

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница \, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (frame_size * 2 + 1 - i >= j) {
                B_1 += img.at<uint8_t>(i, j);
            }
            if (frame_size * 2 + 1 - i <= j) {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница \, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (frame_size * 2 + 1 - i >= j) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            if (frame_size * 2 + 1 - i <= j) {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2) { Q_temp = 2 * max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2); }
    else { Q_temp = 1000; }

    if (Q_temp < Q) {
        Q = Q_temp;
        border_dir = ANG135;
    }

    if (Q == 1000) { Q = 0; }
    else { Q = 1 / (1 + Q); }
    return Q;
}

Mat get_R_img(Mat img, int frame_size, char** border_dir_mat) {
    int width = img.cols;
    int height = img.rows;
    char border_dir_temp;
    Mat frame = Mat(frame_size * 2 + 1, frame_size * 2 + 1, CV_8UC1, 0.0);
    Mat R_img = Mat(height, width, CV_8UC1, BLACK);
    for (int i = frame_size; i < height - frame_size; i++) {
        for (int j = frame_size; j < width - frame_size; j++) {
            for (int f_i = 0; f_i < frame_size * 2 + 1; f_i++) { //заполнение рамки
                for (int f_j = 0; f_j < frame_size * 2 + 1; f_j++) {
                    frame.at<uint8_t>(f_i, f_j) = img.at<uint8_t>(i - frame_size + f_i, j - frame_size + f_j);
                }
            }
            R_img.at<uint8_t>(i, j) = get_R(frame, border_dir_temp, frame_size) * WHITE;
            if (border_dir_mat) { border_dir_mat[i][j] = border_dir_temp; }
        }
    }
    return R_img;
}

Mat get_R_multiplex(Mat img, int* L_arr, int size_L_arr, char** border_dir_mat) {
    int width = img.cols;
    int height = img.rows;
    Mat R_mult = Mat(height, width, CV_8UC1, 0.0), R_temp = Mat(height, width, CV_8UC1, 0.0);
    R_mult = get_R_img(img, L_arr[0], border_dir_mat);
    R_temp = get_R_img(img, L_arr[1]);
    for (int i = 2; i < size_L_arr; i++) {
        R_temp = get_R_img(img, L_arr[i]);
        R_mult = mult_R(R_mult, R_temp);
    }

    return R_mult;
}

void get_tblr(int** img, int* top, int* bot, int* left, int* right, int height, int width, int col){
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (img[i][j] == col) {
                *top = (*top >= i) * i + (*top < i) * *top;
                *left = (*left >= j) * j + (*left < j) * *left;
                *bot = (*bot >= i) * *bot + (*bot < i) * i;
                *right = (*right >= j) * *right + (*right < j) * j;
            }
        }
    }
}

int** mat_to_arr(Mat img) {
    int height = img.rows;
    int width = img.cols;
    int** arr = new int* [height];

    for (int i = 0; i < height; i++) {
        arr[i] = new int [width];
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            arr[i][j] = img.at<uint8_t>(i, j);
        }
    }
    return arr;
}

Mat mult_R(Mat img1, Mat img2) {
    int width = img1.cols;
    int height = img1.rows;
    Mat res = Mat(height, width, CV_8UC1, 0.0);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            res.at<uint8_t>(i, j) = img1.at<uint8_t>(i, j) * img2.at<uint8_t>(i, j) / 255;
        }
    }
    return res;
}

int*** process_center(int** R_img, int*** buf, int i_c, int j_c, char dir) {
    float max_R1 = 0, max_R2 = 0, B1 = 0, sigma1 = 0, B2 = 0, sigma2 = 0;
    int m_i1 = 0, m_j1 = 0, m_i2 = 0, m_j2 = 0;
    for (int i = i_c - base_frame_size; i < i_c + base_frame_size; i++) {
        for (int j = j_c - base_frame_size; j < j_c + base_frame_size; j++) {
            if (dir == ANG90) {
                if (j <= j_c) {
                    if (max_R1 < R_img[i][j]) {
                        max_R1 = R_img[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += R_img[i][j];
                }
                if (j >= j_c) {
                    if (max_R2 < R_img[i][j]) {
                        max_R2 = R_img[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += R_img[i][j];
                }
            }
            if (dir == ANG45) {
                if (i - i_c >= j - j_c) {
                    if (max_R1 < R_img[i][j]) {
                        max_R1 = R_img[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += R_img[i][j];
                }
                if (i - i_c <= j - j_c) {
                    if (max_R2 < R_img[i][j]) {
                        max_R2 = R_img[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += R_img[i][j];
                }
            }
            if (dir == ANG0) {
                if (i <= i_c) {
                    if (max_R1 < R_img[i][j]) {
                        max_R1 = R_img[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += R_img[i][j];
                }
                if (i >= i_c) {
                    if (max_R2 < R_img[i][j]) {
                        max_R2 = R_img[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += R_img[i][j];
                }
            }
            if (dir == ANG135) {
                if (i - i_c <= j_c - j) {
                    if (max_R1 < R_img[i][j]) {
                        max_R1 = R_img[i][j];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += R_img[i][j];
                }
                if (i - i_c >= j_c - j) {
                    if (max_R2 < R_img[i][j]) {
                        max_R2 = R_img[i][j];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += R_img[i][j];
                }
            }
        }
    }

    B1 = 2 * B1 / (2 * base_frame_size + 1) / (2 * base_frame_size + 1);
    B2 = 2 * B2 / (2 * base_frame_size + 1) / (2 * base_frame_size + 1);

    for (int i = i_c - base_frame_size; i < i_c + base_frame_size; i++) {
        for (int j = j_c - base_frame_size; j < j_c + base_frame_size; j++) {
            if (dir == ANG90) {
                if (j <= j_c) {
                    sigma1 = sigma1 + (R_img[i][j] - B1) * (R_img[i][j] - B1);
                }
                if (j >= j_c) {
                    sigma2 = sigma2 + (R_img[i][j] - B2) * (R_img[i][j] - B2);
                }
            }
            if (dir == ANG45) {
                if (i - i_c >= j - j_c) {
                    sigma1 = sigma1 + (R_img[i][j] - B1) * (R_img[i][j] - B1);
                }
                if (i - i_c <= j - j_c) {
                    sigma2 = sigma2 + (R_img[i][j] - B2) * (R_img[i][j] - B2);
                }
            }
            if (dir == ANG0) {
                if (i <= i_c) {
                    sigma1 = sigma1 + (R_img[i][j] - B1) * (R_img[i][j] - B1);
                }
                if (i >= i_c) {
                    sigma2 = sigma2 + (R_img[i][j] - B2) * (R_img[i][j] - B2);
                }
            }
            if (dir == ANG135) {
                if (i - i_c <= j_c - j) {
                    sigma1 = sigma1 + (R_img[i][j] - B1) * (R_img[i][j] - B1);
                }
                if (i - i_c >= j_c - j) {
                    sigma2 = sigma2 + (R_img[i][j] - B2) * (R_img[i][j] - B2);
                }
            }
        }
    }

    sigma1 = sqrt(sigma1) / (2 * base_frame_size + 1);
    sigma2 = sqrt(sigma2) / (2 * base_frame_size + 1);

    for (int i = i_c - base_frame_size; i < i_c + base_frame_size; i++) {
        for (int j = j_c - base_frame_size; j < j_c + base_frame_size; j++) {
            if (dir == ANG90) {
                if (j <= j_c) {
                    if (R_img[i][j] >= B1 - sigma1 && R_img[i][j] <= B1 + sigma1) {
                        buf[i][j][1] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
                if (j >= j_c) {
                    if (R_img[i][j] >= B2 - sigma2 && R_img[i][j] <= B2 + sigma2) {
                        buf[i][j][2] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
            }
            if (dir == ANG45) {
                if (i - i_c >= j - j_c) {
                    if (R_img[i][j] >= B1 - sigma1 && R_img[i][j] <= B1 + sigma1) {
                        buf[i][j][1] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
                if (i - i_c <= j - j_c) {
                    if (R_img[i][j] >= B2 - sigma2 && R_img[i][j] <= B2 + sigma2) {
                        buf[i][j][2] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
            }
            if (dir == ANG0) {
                if (i <= i_c) {
                    if (R_img[i][j] >= B1 - sigma1 && R_img[i][j] <= B1 + sigma1) {
                        buf[i][j][1] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
                if (i >= i_c) {
                    if (R_img[i][j] >= B2 - sigma2 && R_img[i][j] <= B2 + sigma2) {
                        buf[i][j][2] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
            }
            if (dir == ANG135) {
                if (i - i_c <= j_c - j) {
                    if (R_img[i][j] >= B1 - sigma1 && R_img[i][j] <= B1 + sigma1) {
                        buf[i][j][1] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
                if (i - i_c >= j_c - j) {
                    if (R_img[i][j] >= B2 - sigma2 && R_img[i][j] <= B2 + sigma2) {
                        buf[i][j][2] += 1;
                    }
                    else {
                        buf[i][j][0] += 1;
                    }
                }
            }
        }
    }
    return buf;
}

void process_clust(int** clusters, int** R_img, int top, int left, int right, int bot, int col, int height, int width){
    float B = 0, sigma = 0;
    int pix_ctr = 0;

    get_B_sigma(clusters, R_img, &B, &sigma, top, bot, left, right, col, height, width); 
    for (int i = top; i <= bot; i++) {
        for (int j = left; j <= right; j++) {
            //cout << format("%f %f %f %i\n", B - sigma, B + sigma, (float)R_img[i][j], R_img[i][j]);
            if ((float)R_img[i][j] >= B - sigma_multiplier * sigma && (float)R_img[i][j] <= B + sigma_multiplier * sigma) {
                if (clusters[i][j] == BLACK || clusters[i][j] == col) { clusters[i][j] = col; } //cout << "ADD\n";
                else { 
                    //cout << "MERGE"; 
                    int top2 = height - 1, left2 = width - 1, right2 = 0, bot2 = 0, col2 = clusters[i][j];
                    float B2 = 0, sigma2 = 0;
                    get_tblr(clusters, &top2, &bot2, &left2, &right2, height, width, col2);
                    get_B_sigma(clusters, R_img, &B2, &sigma2, top2, bot2, left2, right2, col2, height, width);
                    if (abs(B - B2) <= sigma || abs(B - B2) <= sigma2 || abs(B - B2) <= abs(sigma - sigma2)) {
                       clust_color(clusters, top2, bot2, left2, right2, col, col2);
                       cout << format("Merged %i %i\n", col, col2); 
                    }
                }
            }
        }
    }
    fill_holes(clusters, top, bot, left, right, col, height, width);
    //cout << "proc_clust\n";
}

int** spread(int** img, int height, int width){
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == 0 && j == 0 || i == 0 && j == width - 1 || i == height - 1 && j == 0 || i == height - 1 && j == width - 1) { continue; }
            else {
                if (i == 0 || i == height - 1) {
                    if (img[i][j - 1] == img[i][j + 1] && img[i][j - 1] == img[i + (i == 0) - (i == height - 1)][j]) {
                        img[i][j] = img[i][j - 1];
                    }
                    continue;
                }
                if (j == 0 || j == width - 1) {
                    if (img[i - 1][j] == img[i + 1][j] && img[i - 1][j] == img[i][j + (j == 0) - (j == width - 1)]) {
                        img[i][j] = img[i - 1][j];
                    }
                    continue;
                }
                if (img[i - 1][j] == img[i + 1][j]) {
                    if (img[i + 1][j] == img[i][j - 1] || img[i + 1][j] == img[i][j + 1]) { img[i][j] = img[i - 1][j]; }
                }
                else {
                    if (img[i - 1][j] == img[i][j - 1]) {
                        if (img[i][j - 1] == img[i][j + 1]) { img[i][j] = img[i - 1][j]; }
                    }
                    else {
                        if (img[i + 1][j] == img[i][j - 1] && img[i][j - 1] == img[i][j + 1]) { img[i][j] = img[i + 1][j]; }
                    }
                }
            }
        }
    }
    return img;
}
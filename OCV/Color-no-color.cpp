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
const int border_R_mult = 90;
const int BLACK = 0;
const int WHITE = 255;
const char ANG0 = '-';
const char ANG45 = '/';
const char ANG90 = '|';
const char ANG135 = '\\';
string name = "cell.jpg";
string path = "C:/OCV/IMG/";

// ѕрототипы
Mat advanced_spread(Mat img);
Mat clust_cent_R(Mat img, int frame_size); 
Mat clust_lines(Mat cent_img, char** border_dir_mat, int frame_size);
Mat contrast_R(Mat img);
void draw_hist(Mat& img, string name, string path);
float get_R(Mat img, char& border_dir, int frame_size = base_frame_size);
Mat get_R_img(Mat img, int frame_size = base_frame_size, char** border_dir_mat = NULL);
Mat get_R_multiplex(Mat img, int* L_arr, int size_L_arr, char** border_dir_mat = NULL);
Mat gray_img(Mat img);
Mat filter_img(Mat img, Mat hist);
vector<float> find_extr(Mat func);
Mat find_peak(Mat frame, char dir, int frame_size, int col);
Mat find_peaks(Mat img, Mat centers, char** border_dir_mat, int frame_size);
Mat follow_inner_border(Mat img, int color, int* left, int* right, int* top, int* bot, int i_start = 0, int j_start = 0);
Mat follow_outer_border(Mat img, int color, int left, int right, int top, int bot, float B, float sigma, vector <vector <int>> &merge_params);
Mat layer_RGB(Mat img, Mat peaks); 
void merge_clusters(Mat img, int col1, int col2, int i, int j);
Mat mult_R(Mat img1, Mat img2);
Mat negative(Mat img);
int process_clust(Mat img, int col, int i = 0, int j = 0, int mode = 0);
Mat smooth(Mat func);
Mat spread(Mat img);
Mat transp_plus(Mat top_img, Mat bot_img);

int main(int argc, char** argv) //C:\OCV\OCV\x64\Debug\OCV.exe
{
    Mat gray_sys, img;
    addSamplesDataSearchPath(path);
    img = imread(findFile(name), IMREAD_COLOR);
    cvtColor(img, gray_sys, COLOR_BGR2GRAY);

    int width = img.cols;
    int height = img.rows;
    char** border_dir_mat = (char**)malloc(height * sizeof(char*));
    for (int i = 0; i < height; i++) {
        border_dir_mat[i] = (char*)malloc(width * sizeof(char));
    }

    int L_arr[10] = { base_frame_size, 5, 16, 32, 6, 7, 8, 9, 10 };
    Mat R_mult = get_R_multiplex(gray_sys, L_arr, 2, border_dir_mat);
    Mat centers, clusters, img_plus_clusters, R_plus_clusters;
    imshow("R mult", R_mult);
    centers = clust_cent_R(R_mult, L_arr[0]);
    imshow("cluster centers", centers);
    clusters = find_peaks(R_mult, centers, border_dir_mat, base_frame_size);
    imshow("clusters", clusters);
    imwrite(path + "cluster_peaks_" + name, clusters);
    R_plus_clusters = layer_RGB(R_mult, clusters);
    //advanced_spread(R_plus_clusters);
    imshow("adv spread", R_plus_clusters);
    //imwrite(path + "spread_" + to_string(base_frame_size) + "_" + name, R_plus_clusters);
    img_plus_clusters = layer_RGB(gray_sys, clusters);
    imshow("result", img_plus_clusters);
    imwrite(path + "RGB_" + to_string(base_frame_size) + "_" + name, img_plus_clusters);
    waitKey(0);
}

Mat advanced_spread(Mat img){
    int width = img.cols;
    int height = img.rows;
    int i = 0, j = 0, color_ctr = 2, clust_ctr = 0, top = height - 1, left = width - 1, right = 0, bot = 0;
    while (i < height && j < width){
        if (img.at<Vec3b>(i, j)[1] == color_ctr) {
            (i, j) = process_clust(img, color_ctr, i, j, 1);
            spread(img);
            follow_inner_border(img, color_ctr, &left, &right, &top, &bot);
            //follow_outer_border(img, color_ctr, left, right, top, bot,);
            //imwrite(path + "spreading_" + to_string(color_ctr) + "_" + to_string(clust_ctr) + "_" + name, img);
            color_ctr++;
            if (color_ctr >= 255) { 
                color_ctr = 2;
                clust_ctr++;
            }
            if (clust_ctr > 10) { break; }
            cout << color_ctr << ' ' << clust_ctr << '\n';
        }
        else {
            i++;
            if (i >= height) { 
                i = 0;
                j++;
                if (j >= width) { j = 0; }
            }
            continue;
        }
    }
    return img;
}

Mat clust_cent_R(Mat img, int frame_size) {
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
                        if (max < img.at<uint8_t>(i + f_i, j + f_j) && (f_i*f_i + f_j*f_j != 0)) {
                            clust.at<uint8_t>(i, j) = WHITE;
                            ismax = FALSE;
                            break;
                        }
                        if (ismax && pastmax) {
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

Mat clust_lines(Mat cent_img, char ** border_dir_mat, int frame_size) {
    int width = cent_img.cols;
    int height = cent_img.rows;
    Mat clust_l = Mat(height, width, CV_8UC1, 0.0); 
    bool break_flag_0 = FALSE;
    for (int i = frame_size; i < height - frame_size; i++) {
        for (int j = frame_size; j < width - frame_size; j++) {
            clust_l.at<uint8_t>(i, j) = WHITE;
            if (cent_img.at<uint8_t>(i, j) == BLACK) {
                if (border_dir_mat[i][j] == '0') {
                    break_flag_0 = TRUE;
                    continue;
                }
                if (border_dir_mat[i][j] == ANG45){
                    for (int f_i = -frame_size; f_i < frame_size; f_i++) {
                        clust_l.at<uint8_t>(i + f_i, j - f_i) = BLACK;
                    }
                }
                if (border_dir_mat[i][j] == ANG0) {
                    for (int f_i = -frame_size; f_i < frame_size; f_i++) {
                        clust_l.at<uint8_t>(i, j + f_i) = BLACK;
                    }
                }
                if (border_dir_mat[i][j] == ANG135) {
                    for (int f_i = -frame_size; f_i < frame_size; f_i++) {
                        clust_l.at<uint8_t>(i + f_i, j + f_i) = BLACK;
                    }
                }
                if (border_dir_mat[i][j] == ANG90) {
                    for (int f_i = -frame_size; f_i < frame_size; f_i++) {
                        clust_l.at<uint8_t>(i + f_i, j) = BLACK;
                    }
                }
                break_flag_0 = FALSE;
            }
        }
        if (break_flag_0) { continue; }

    }
    return clust_l;
}

Mat contrast_R(Mat img) {
    int width = img.cols;
    int height = img.rows;
    Mat cont = Mat(height, width, CV_8UC1, 0.0); 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (img.at<uint8_t>(i, j) > border_R_mult) { cont.at<uint8_t>(i, j) = BLACK; }
            else { cont.at<uint8_t>(i, j) = WHITE; }
        }
    }
    return cont;
}

void draw_hist(Mat& img, string name, string path) {

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / hist_size);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(img, img, 0, histImage.rows, NORM_MINMAX, -1,
        Mat());

    for (int i = 1; i < hist_size; i++) {
        line(
            histImage,
            Point(bin_w * (i - 1), hist_h - cvRound(img.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(img.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    namedWindow("calcHist", WINDOW_AUTOSIZE);
    imshow("calcHist", histImage);

    imwrite(path + "hist_" + name, histImage);
}

Mat filter_img(Mat img, Mat hist) {

    int width = img.cols;
    int height = img.rows;
    Mat new_img = Mat(height, width, CV_8UC1, 0.0);
    vector<float> extr = find_extr(hist);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < extr.size(); k++) {
                if ((extr[k] > 0) && (int(img.at<uint8_t>(i, j)) >= extr[k] - delta) && (int(img.at<uint8_t>(i, j)) <= extr[k] + delta)) {
                    new_img.at<uint8_t>(i, j) = WHITE;
                    break;
                }
                else{ new_img.at<uint8_t>(i, j) = BLACK; }
            }
        }
    }

    return new_img;
}

vector<float> find_extr(Mat func) { //по сглаженной функции ищем местоположение экстремума 

    vector<float> loc_extr;  

    Mat sfunc = smooth(func);

    for (int i = 1; i < hist_size - 1; i++)
    {   
        if ((sfunc.at<float>(i) > sfunc.at<float>(i - 1)) && (sfunc.at<float>(i) > sfunc.at<float>(i + 1))) {
            loc_extr.push_back(i);
            cout << "max: " << i << ' ' << sfunc.at<float>(i) << '\n';
        }
        if ((sfunc.at<float>(i) < sfunc.at<float>(i - 1)) && (sfunc.at<float>(i) < sfunc.at<float>(i + 1))) {
            loc_extr.push_back(-i);
            cout << "min: " << i << ' ' << sfunc.at<float>(i) << '\n';
        }
    }
    return loc_extr;
}

Mat find_peak(Mat frame, char dir, int frame_size, int col){
    float max_R1 = 0, m_i1 = 0, m_j1 = 0, max_R2 = 0, m_i2 = 0, m_j2 = 0, B1 = 0, sigma1 = 0, B2 = 0, sigma2 = 0;

    for (int i = 0; i < 2 * frame_size + 1; i++) {
        for (int j = 0; j < 2 * frame_size + 1; j++) {
            if (dir == ANG90) {
                if (j < frame_size + 1) {
                    if (max_R1 < frame.at<Vec3b>(i, j)[0]) {
                        max_R1 = frame.at<Vec3b>(i, j)[0];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame.at<Vec3b>(i, j)[0];
                }
                else {
                    if (max_R2 < frame.at<Vec3b>(i, j)[0]) {
                        max_R2 = frame.at<Vec3b>(i, j)[0];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame.at<Vec3b>(i, j)[0];
                }
            }
            if (dir == ANG45) {
                if (i < 2*frame_size + 1 - j){
                    if (max_R1 < frame.at<Vec3b>(i, j)[0]) {
                        max_R1 = frame.at<Vec3b>(i, j)[0];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame.at<Vec3b>(i, j)[0];
                }
                else{
                    if (max_R2 < frame.at<Vec3b>(i, j)[0]) {
                        max_R2 = frame.at<Vec3b>(i, j)[0];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame.at<Vec3b>(i, j)[0];
                }
            }
            if (dir == ANG0) {
                if (i < frame_size + 1) {
                    if (max_R1 < frame.at<Vec3b>(i, j)[0]) {
                        max_R1 = frame.at<Vec3b>(i, j)[0];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame.at<Vec3b>(i, j)[0];
                }
                else {
                    if (max_R2 < frame.at<Vec3b>(i, j)[0]) {
                        max_R2 = frame.at<Vec3b>(i, j)[0];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame.at<Vec3b>(i, j)[0];
                }
            }
            if (dir == ANG135) {
                if (i < j) {
                    if (max_R1 < frame.at<Vec3b>(i, j)[0]) {
                        max_R1 = frame.at<Vec3b>(i, j)[0];
                        m_i1 = i;
                        m_j1 = j;
                    }
                    B1 += frame.at<Vec3b>(i, j)[0];
                }
                else {
                    if (max_R2 < frame.at<Vec3b>(i, j)[0]) {
                        max_R2 = frame.at<Vec3b>(i, j)[0];
                        m_i2 = i;
                        m_j2 = j;
                    }
                    B2 += frame.at<Vec3b>(i, j)[0];
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
                    sigma1 = sigma1 + (frame.at<Vec3b>(i, j)[0] - B1) * (frame.at<Vec3b>(i, j)[0] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame.at<Vec3b>(i, j)[0] - B2) * (frame.at<Vec3b>(i, j)[0] - B2);
                }
            }
            if (dir == ANG45) {
                if (i < 2 * frame_size + 1 - j) {
                    sigma1 = sigma1 + (frame.at<Vec3b>(i, j)[0] - B1) * (frame.at<Vec3b>(i, j)[0] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame.at<Vec3b>(i, j)[0] - B2) * (frame.at<Vec3b>(i, j)[0] - B2);
                }
            }
            if (dir == ANG0) {
                if (i < frame_size + 1) {
                    sigma1 = sigma1 + (frame.at<Vec3b>(i, j)[0] - B1) * (frame.at<Vec3b>(i, j)[0] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame.at<Vec3b>(i, j)[0] - B2) * (frame.at<Vec3b>(i, j)[0] - B2);
                }
            }
            if (dir == ANG135) {
                if (i < j) {
                    sigma1 = sigma1 + (frame.at<Vec3b>(i, j)[0] - B1) * (frame.at<Vec3b>(i, j)[0] - B1);
                }
                else {
                    sigma2 = sigma2 + (frame.at<Vec3b>(i, j)[0] - B2) * (frame.at<Vec3b>(i, j)[0] - B2);
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
                    if (frame.at<Vec3b>(i, j)[0] >= B1 - sigma1 && frame.at<Vec3b>(i, j)[0] <= B1 + sigma1) {
                        frame.at<Vec3b>(i, j)[1] = col;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
                else {
                    if (frame.at<Vec3b>(i, j)[0] >= B2 - sigma2 && frame.at<Vec3b>(i, j)[0] <= B2 + sigma2) {
                        frame.at<Vec3b>(i, j)[1] = col + 1;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
            }
            if (dir == ANG45) {
                if (i < 2 * frame_size + 1 - j) {
                    if (frame.at<Vec3b>(i, j)[0] >= B1 - sigma1 && frame.at<Vec3b>(i, j)[0] <= B1 + sigma1) {
                        frame.at<Vec3b>(i, j)[1] = col;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
                else {
                    if (frame.at<Vec3b>(i, j)[0] >= B2 - sigma2 && frame.at<Vec3b>(i, j)[0] <= B2 + sigma2) {
                        frame.at<Vec3b>(i, j)[1] = col + 1;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
            }
            if (dir == ANG0) {
                if (i < frame_size + 1){
                    if (frame.at<Vec3b>(i, j)[0] >= B1 - sigma1 && frame.at<Vec3b>(i, j)[0] <= B1 + sigma1) {
                        frame.at<Vec3b>(i, j)[1] = col;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
                else {
                    if (frame.at<Vec3b>(i, j)[0] >= B2 - sigma2 && frame.at<Vec3b>(i, j)[0] <= B2 + sigma2) {
                        frame.at<Vec3b>(i, j)[1] = col + 1;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
            }
            if (dir == ANG135) {
                if (i < j) {
                    if (frame.at<Vec3b>(i, j)[0] >= B1 - sigma1 && frame.at<Vec3b>(i, j)[0] <= B1 + sigma1) {
                        frame.at<Vec3b>(i, j)[1] = col;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
                else {
                    if (frame.at<Vec3b>(i, j)[0] >= B2 - sigma2 && frame.at<Vec3b>(i, j)[0] <= B2 + sigma2) {
                        frame.at<Vec3b>(i, j)[1] = col + 1;
                    }
                    else { frame.at<Vec3b>(i, j)[1] = 0; }
                }
            }
        }
    }

    spread(frame);

    return frame;
}

Mat find_peaks(Mat img, Mat centers, char** border_dir_mat, int frame_size = base_frame_size) {

    int width = img.cols;
    int height = img.rows;
    Mat peaks = Mat(height, width, CV_8UC1, BLACK);
    Mat frame = Mat(2 * frame_size + 1, 2 * frame_size + 1, CV_8UC3);
    int col_ctr = 0;
    for (int i = frame_size; i < height - frame_size; i++) {
        for (int j = frame_size; j < width - frame_size; j++) {
            if (centers.at<uint8_t>(i, j) == BLACK){
                for (int f_i = 0; f_i < 2 * frame_size + 1; f_i++) {
                    for (int f_j = 0; f_j < 2 * frame_size + 1; f_j++) {
                        frame.at<Vec3b>(f_i, f_j)[0] = img.at<uint8_t>(i + f_i - frame_size, j + f_j - frame_size);
                    }
                }
                col_ctr += 2;
                if (col_ctr >= 255) { col_ctr = 2; }
                frame = find_peak(frame, border_dir_mat[i][j], frame_size, col_ctr);
                process_clust(frame, col_ctr);
                for (int f_i = 0; f_i < 2 * frame_size + 1; f_i++) {
                    for (int f_j = 0; f_j < 2 * frame_size + 1; f_j++) {
                        peaks.at<uint8_t>(i + f_i - frame_size, j + f_j - frame_size) = frame.at<Vec3b>(f_i, f_j)[1];
                    }
                }
            }
        }
    }
    return peaks;
}

Mat follow_inner_border(Mat img, int color, int* left, int* right, int* top, int* bot, int i_start, int j_start) {
    int width = img.cols;
    int height = img.rows;
    int directions[8][2] = {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};
    bool flag = FALSE;
    Mat img_w_border = Mat(height, width, CV_8UC1, BLACK);

    if (i_start == 0 && j_start == 0){
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (img.at<uint8_t>(i, j)) {
                    i_start = i;
                    j_start = j;
                    flag = TRUE;
                    break;
                }
            }
            if (flag) { break; }
        }
    }
    int i = i_start, j = j_start, dir = 0;
    int problem_detector = 0; // костылик
    bool start = TRUE;
    //cout << "cycle started\n";
    while (i > i_start || j > j_start || start) {
        if (i + directions[dir][0] <= 0) {
            dir = 7 * (dir != 7 && dir != 0) + (dir == 0); 
            i = -directions[dir][0];
            if (img.at<Vec3b>(i + directions[dir][0], j - 1)[1] == color && img_w_border.at<uint8_t>(i + directions[dir][0], j - 1) == BLACK && j < width - 1) { j--; }
            else {
                cout << format("Oh, no! i = %i, j = %i, dir = %i", i, j, dir) << '\n'; 
                problem_detector++; 
                if (problem_detector >= 8) { 
                    break;
                    cout << format("Breaking at i = %i, j = %i", i, j);
                }
            }
            img_w_border.at<uint8_t>(i + directions[dir][0], j + directions[dir][1]) = WHITE;
            *top = 0;
            continue;
        }
        if (j + directions[dir][1] <= 0) {
            dir = 5 + (dir == 5) + 2 * (dir == 6); 
            j = -directions[dir][1];
            if (img.at<Vec3b>(i - 1, j + directions[dir][1])[1] == color && img_w_border.at<uint8_t>(i - 1, j + directions[dir][1]) == BLACK && i > 0) { i--; }
            else { 
                cout << format("Oh, no! i = %i, j = %i, dir = %i", i, j, dir) << '\n';
                problem_detector++;
                if (problem_detector >= 8) { 
                    break;
                    cout << format("Breaking at i = %i, j = %i", i, j);
                }
            }
            img_w_border.at<uint8_t>(i + directions[dir][0], j + directions[dir][1]) = WHITE;
            *left = 0;
            continue;
        }
        if (i + directions[dir][0] >= height - 1) {
            dir = 3 + (dir == 3) + 2 * (dir == 4); 
            i = height - 1 - directions[dir][0];
            if (img.at<Vec3b>(i + directions[dir][0], j + 1)[1] == color && img_w_border.at<uint8_t>(i + directions[dir][0], j + 1) == BLACK && j < width - 1) { j++; }
            else {
                cout << format("Oh, no! i = %i, j = %i, dir = %i", i, j, dir) << '\n';
                problem_detector++;
                if (problem_detector >= 8) { 
                    break;
                    cout << format("Breaking at i = %i, j = %i", i, j);
                }
            }
            img_w_border.at<uint8_t>(i, j) = WHITE;
            *bot = height - 1;
            continue;
        }
        if (j + directions[dir][1] >= width - 1){
            dir = 1 + (dir == 1) + 2 * (dir == 2);
            j = width - 1 - directions[dir][0];
            if (img.at<Vec3b>(i - 1, j + directions[dir][1])[1] == color && img_w_border.at<uint8_t>(i - 1, j + directions[dir][1]) == BLACK && i > 0) { i--; }
            else {
                cout << format("Oh, no! i = %i, j = %i, dir = %i", i, j , dir) << '\n'; 
                problem_detector++;
                if (problem_detector >= 8) { 
                    break;
                    cout << format("Breaking at i = %i, j = %i", i, j);
                }
            }
            img_w_border.at<uint8_t>(i, j) = WHITE;
            *right = width - 1;
            continue;
        }

        if (    i + directions[dir][0] < height
            &&  j + directions[dir][1] < width
            &&  i + directions[dir][0] - directions[dir][1] < height
            &&  j + directions[dir][0] + directions[dir][1] < width
            &&  i + directions[dir][0] >= 0
            &&  j + directions[dir][1] >= 0
            &&  i + directions[dir][0] - directions[dir][1] >= 0
            &&  j + directions[dir][0] + directions[dir][1] >= 0
            &&  img.at<Vec3b>(i + directions[dir][0] - directions[dir][1], j + directions[dir][0] + directions[dir][1])[1] != color
            &&  img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[1] == color
            &&  img_w_border.at<uint8_t>(i + directions[dir][0], j + directions[dir][1]) == BLACK) {

            img_w_border.at<uint8_t>(i + directions[dir][0], j + directions[dir][1]) = WHITE;
            if (j < *left) { *left = j; }
            if (j > *right){ *right = j; }
            if (i < *top)  { *top = i; }
            if (i > *bot)  { *bot = i; }
            i += directions[dir][0];
            j += directions[dir][1];
            start = FALSE;
            problem_detector = 0;
        }
        else {
            if (dir < 8) { dir++; }
            else { dir = 0; }
            problem_detector++;
            if (problem_detector == 8) { 
                cout << format("Breaking no dir found i = %i, j = %i, dir = %i, color = %i, w = %i, h = %i ", i, j, dir, color, width, height) << '\n'; 
                break;
            }
        }
    }
    return img_w_border;
}

Mat follow_outer_border(Mat img, int color, int left, int right, int top, int bot, float B, float sigma, vector <vector <int>>& merge_params) { // !!!!!!! сли€ние кластеров
    int width = img.cols;
    int height = img.rows;
    int directions[8][2] = { {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1} };
    int dir = 0;
    bool flag = FALSE;
    Mat img_w_border = Mat(height, width, CV_8UC1, BLACK);
    top -= (top > 0);
    left -= (left > 0);
    bot += (bot < height - 1);
    right += (right < width - 1);
    for (int i = top; i <= bot; i++) {
        for (int j = left; j <= right; j++) {
            if (i <= 0) {
                i = 0;
                dir = 6;
                if (img.at<Vec3b>(i, j - 1)[1] == color && j > 0) { while (j < width - 1 && img.at<Vec3b>(i, j - 1)[1] == color) { j--; } continue; }
            }
            if (j <= 0) {
                j = 0;
                dir = 4;
                if (img.at<Vec3b>(i + 1, j)[1] == color && i < height - 1) { while (i < height - 1 && img.at<Vec3b>(i + 1, j)[1] == color) { i++; } continue; }
            }
            if (i >= height - 1) {
                i = height - 1;
                dir = 2;
                if (img.at<Vec3b>(i, j + 1)[1] == color && j < width - 1) { while (j < width && img.at<Vec3b>(i, j + 1)[1] == color) { j++; } continue; }
            }
            if (j >= width - 1) {
                j = width - 1;
                dir = 0;
                if (img.at<Vec3b>(i - 1, j)[1] == color && i > 0) { while (i > 0 && img.at<Vec3b>(i - 1, j)[1] == color) { i--; } continue; }
            }

            if (i + directions[dir][0] < height
                && j + directions[dir][1] < width
                && i - directions[dir][0] + directions[dir][1] < height
                && j - directions[dir][0] - directions[dir][1] < width
                && i + directions[dir][0] >= 0
                && j + directions[dir][1] >= 0
                && i - directions[dir][0] + directions[dir][1] >= 0
                && j - directions[dir][0] - directions[dir][1] >= 0
                && img.at<Vec3b>(i - directions[dir][0] + directions[dir][1], j - directions[dir][0] - directions[dir][1])[1] == color
                && img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[1] != color
                && img_w_border.at<uint8_t>(i + directions[dir][0], j + directions[dir][1]) == BLACK) {
                img_w_border.at<uint8_t>(i + directions[dir][0], j + directions[dir][1]) = WHITE;
                if (img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[0] <= B + sigma && img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[0] >= B - sigma) {
                    if (img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[1] == BLACK || img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[1] == WHITE) {
                        img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[1] = color; 
                        cout << format("\nYAY, PIXEL ADDED, i = %i, j = %i\n", i + directions[dir][0], j + directions[dir][1]);
                    }
                    else { 
                        cout << format("\nWANNA MERGE, i = %i, j = %i\n", i + directions[dir][0], j + directions[dir][1]);
                        merge_params.push_back({ i, j, color, img.at<Vec3b>(i + directions[dir][0], j + directions[dir][1])[1] }); 
                    }

                    i += directions[dir][0];
                    j += directions[dir][1];
                }
                else {
                    if (dir < 8) { dir++; }
                    else { dir = 0; }
                }
            }
        }
    }
    return img_w_border;
}

float get_R(Mat img, char &border_dir, int frame_size) {
    float B_1 = 0, B_2 = 0;
    float s_1 = 0, s_2 = 0;
    float Q, Q_temp;
    border_dir = '0';
    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница --, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i < frame_size + 1) {
                B_1 += img.at<uint8_t>(i,j);
            }
            else {
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
            else {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2){ 
        Q = max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2);
        border_dir = ANG0;
    }
    else { Q = 1000; }

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница \, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i > j) {
                B_1 += img.at<uint8_t>(i, j);
            }
            else {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница \, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (i > j) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            else {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2){ Q_temp = max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2); }
    else { Q_temp = 1000; }

    if (Q_temp < Q) { 
        Q = Q_temp;
        border_dir = ANG135;
    }

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница |, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (j < frame_size + 1) {
                B_1 += img.at<uint8_t>(i, j);
            }
            else {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница |, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (j < frame_size + 1) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            else {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2) { Q_temp = max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2); }
    else { Q_temp = 1000; }

    if (Q_temp < Q) { 
        Q = Q_temp;
        border_dir = ANG90;
    }

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница /, средн€€ €ркость
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (frame_size * 2 + 1 - i > j) {
                B_1 += img.at<uint8_t>(i, j);
            }
            else {
                B_2 += img.at<uint8_t>(i, j);
            }
        }
    }
    B_1 = B_1 / (frame_size * (frame_size * 2 - 1));
    B_2 = B_2 / (frame_size * (frame_size * 2 - 1));

    for (int i = 0; i < frame_size * 2 + 1; i++) {      // √раница /, дисперси€
        for (int j = 0; j < frame_size * 2 + 1; j++) {
            if (frame_size * 2 + 1 - i > j) {
                s_1 += (img.at<uint8_t>(i, j) - B_1) * (img.at<uint8_t>(i, j) - B_1);
            }
            else {
                s_2 += (img.at<uint8_t>(i, j) - B_2) * (img.at<uint8_t>(i, j) - B_2);
            }
        }
    }
    s_1 = sqrt(s_1 / (frame_size * (frame_size * 2 - 1)));
    s_2 = sqrt(s_2 / (frame_size * (frame_size * 2 - 1)));

    if (B_1 != B_2) { Q_temp = max(min(s_1, s_2), delta_Q) / abs(B_1 - B_2); }
    else { Q_temp = 1000; }

    if (Q_temp < Q) { 
        Q = Q_temp;
        border_dir = ANG45;
    }

    if (Q == 1000) { Q = 0; }
    else { Q = 1 / (1 + Q); }
    return Q;
}

Mat get_R_img(Mat img, int frame_size, char **border_dir_mat) {
    int width = img.cols;
    int height = img.rows;
    char border_dir_temp;
    Mat frame = Mat(frame_size * 2 + 1, frame_size * 2 + 1, CV_8UC1, 0.0);
    Mat R_img = Mat(height, width, CV_8UC1, 0.0); 
    for (int i = frame_size; i < height - frame_size; i++) {
        for (int j = frame_size; j < width - frame_size; j++) {
            for (int f_i = 0; f_i < frame_size * 2 + 1; f_i++) { //заполнение рамки
                for (int f_j = 0; f_j < frame_size * 2 + 1; f_j++) {
                    frame.at<uint8_t>(f_i, f_j) = img.at<uint8_t>(i - frame_size + f_i, j - frame_size + f_j);
                }
            }
            R_img.at<uint8_t>(i, j) = get_R(frame, border_dir_temp, frame_size) * WHITE;
            if (border_dir_mat){ border_dir_mat[i][j] = border_dir_temp; }
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

Mat gray_img(Mat img) {
    int width = img.cols;
    int height = img.rows;
    //uint8_t* myData = img.data;
    //int _stride = img.step;
    Mat gray = Mat(height, width, CV_8UC1, 0.0);
    //cout << width << ' ' << height << ' ' << _stride << '\n'; //553 271 1659 = 553*3
    //cout << sizeof(myData) << '\n'; //8 какие-то буквы
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            //cout << img.at<Vec3b>(i,j)[0] << ' '; // оп€ть буквы
            //cout << i << ' ' << j << ' ';
            gray.at<uint8_t>(i, j) = (int(img.at<Vec3b>(i, j)[0]) * 114 + int(img.at<Vec3b>(i, j)[1]) * 587 + int(img.at<Vec3b>(i, j)[2]) * 299 + 500)/1000; //myData[i * _stride + j]; 
            //cout << int(gray.at<uint8_t>(j, i)) << ' ';
        }
        cout << '\n';
    }
    //Gray = (R * 299 + G * 587 + B * 114 + 500) / 1000;https://russianblogs.com/article/2005306899/
    return gray;
}

Mat layer_RGB(Mat img, Mat peaks){
    int width = img.cols;
    int height = img.rows;
    Mat result = Mat(height, width, CV_8UC3, BLACK);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            result.at<Vec3b>(i, j)[0] = img.at<uint8_t>(i, j);
            result.at<Vec3b>(i, j)[1] = peaks.at<uint8_t>(i, j);
        }
    }
    return result;
}

void merge_clusters(Mat img, int col1, int col2, int i, int j){
    img.at<Vec3b>(i, j)[1] = col1;
    if (i > 0 && img.at<Vec3b>(i - 1, j)[1] == col2) { merge_clusters(img, col1, col2, i - 1, j); }
    if (i < img.rows - 1 && img.at<Vec3b>(i + 1, j)[1] == col2) { merge_clusters(img, col1, col2, i + 1, j); }
    if (j > 0 && img.at<Vec3b>(i, j - 1)[1] == col2) { merge_clusters(img, col1, col2, i, j - 1); }
    if (j < img.cols - 1 && img.at<Vec3b>(i, j + 1)[1] == col2) { merge_clusters(img, col1, col2, i, j + 1); }
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

Mat negative(Mat img) {
    int width = img.cols;
    int height = img.rows;
    Mat neg = Mat(height, width, CV_8UC1, 0.0);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            neg.at<uint8_t>(i, j) = 255 - img.at<uint8_t>(i, j);
        }
    }
    return neg;
}

int process_clust(Mat img, int col, int i, int j, int mode){
    int width = img.cols;
    int height = img.rows;
    int left = j, right = j + width - 1, top = i, bot = i + height - 1;
    int pix_ctr1 = 0, pix_ctr2 = 0;
    float B1 = 0, B2 = 0, sigma1 = 0, sigma2 = 0;
    if (mode) {
        //follow_inner_border(img, col, &left, &right, &top, &bot, i, j)
        if (bot > height - 1) { bot = height - 1; }
        if (right > width - 1) { right = width - 1; }
    }
    //€ркость
    //cout << "B\n";
    for (i = top; i <= bot; i++) {
        for (j = left; j <= right; j++) {
            if (img.at<Vec3b>(i, j)[1] == col) {
                B1 += img.at<Vec3b>(i, j)[0];
                pix_ctr1++;
            }
            if (img.at<Vec3b>(i, j)[1] == col + 1) {
                B2 += img.at<Vec3b>(i, j)[0];
                pix_ctr2++;
            }
        }
    }
    B1 /= pix_ctr1;
    B2 /= pix_ctr2;
    //дисперси€
    //cout << "sigma\n";
    for (i = top; i <= bot; i++) {
        for (j = left; j <= right; j++) {
            if (img.at<Vec3b>(i, j)[1] == col) {
                sigma1 += (img.at<Vec3b>(i, j)[0] - B1) * (img.at<Vec3b>(i, j)[0] - B1);
            }
            if (img.at<Vec3b>(i, j)[1] == col + 1) {
                sigma2 += (img.at<Vec3b>(i, j)[0] - B2) * (img.at<Vec3b>(i, j)[0] - B2);
            }
        }
    }
    sigma1 = sqrt(sigma1) / pix_ctr1;
    sigma2 = sqrt(sigma2) / pix_ctr2;

    for (i = top; i <= bot; i++) {
        for (j = left; j <= right; j++) {
            if (img.at<Vec3b>(i, j)[0] <= B1 + sigma1 && img.at<Vec3b>(i, j)[0] >= B1 - sigma1 && img.at<Vec3b>(i, j)[1] != col) {
                if (img.at<Vec3b>(i, j)[1] == BLACK) { img.at<Vec3b>(i, j)[1] = col; }
                else {
                    cout << "\nMERGE\n";
                    merge_clusters(img, col, img.at<Vec3b>(i, j)[1], i, j);
                    cout << "\nMERGE DONE\n";
                }
            }
            if (img.at<Vec3b>(i, j)[0] <= B2 + sigma2 && img.at<Vec3b>(i, j)[0] >= B2 - sigma2 && img.at<Vec3b>(i, j)[1] != col + 1) {
                if (img.at<Vec3b>(i, j)[1] == BLACK) { img.at<Vec3b>(i, j)[1] = col + 1; }
                else {
                    cout << "\nMERGE\n";
                    merge_clusters(img, col + 1, img.at<Vec3b>(i, j)[1], i, j);
                    cout << "\nMERGE DONE\n";
                }
            }
        }
    }

    return (top, left);
}

Mat smooth(Mat func) { // сглаженна€ функци€, экстр имеют другое значение
    float val = 0;
    for (int i = interval_half; i < hist_size - interval_half; i++)
    {   
        val = func.at<float>(i);
        for (int j = 1; j < interval_half; j++) {
            val += func.at<float>(i + j) + func.at<float>(i - j);
        }
        func.at<float>(i) = val / (interval_half * 2 - 1);
    }
    return func;
}

Mat spread(Mat img) {
    int width = img.cols;
    int height = img.rows;
    int val1, val2, val3, val4;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == 0 && j == 0 || i == 0 && j == width - 1 || i == height - 1 && j == 0 || i == height - 1 && j == width - 1) { continue; }
            else {
                if (i == 0 || i == height - 1) {
                    if (img.at<Vec3b>(i, j - 1)[1] == img.at<Vec3b>(i, j + 1)[1] && img.at<Vec3b>(i, j - 1) == img.at<Vec3b>(i + (i == 0) - (i == height - 1), j)) {
                        img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j - 1)[1];
                    }
                    continue;
                }
                if (j == 0 || j == width - 1) {
                    if (img.at<Vec3b>(i - 1, j)[1] == img.at<Vec3b>(i + 1, j)[1] && img.at<Vec3b>(i - 1, j)[1] == img.at<Vec3b>(i, j + (j == 0) - (j == width - 1))[1]) {
                        img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i - 1, j)[1];
                    }
                    continue;
                }
                val1 = img.at<Vec3b>(i - 1, j)[1];
                val2 = img.at<Vec3b>(i + 1, j)[1];
                val3 = img.at<Vec3b>(i, j - 1)[1];
                val4 = img.at<Vec3b>(i, j + 1)[1];
                if (val1 == val2) {
                    if (val2 == val3 || val2 == val4) { img.at<Vec3b>(i, j)[1] = val1; }
                }
                else {
                    if (val1 == val3) {
                        if (val3 == val4) { img.at<Vec3b>(i, j)[1] = val1; }
                    }
                    else {
                        if (val2 == val3 && val3 == val4) { img.at<Vec3b>(i, j)[1] = val2; }
                    }
                }
            }
        }
    }
    return img;
}

Mat transp_plus(Mat top_img, Mat bot_img){
    int width = top_img.cols;
    int height = top_img.rows;
    Mat res_img = Mat(height, width, CV_8UC1, 0.0);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (top_img.at<uint8_t>(i, j) == BLACK) { res_img.at<uint8_t>(i, j) = BLACK; }
            else { res_img.at<uint8_t>(i, j) = 255 - (255 - bot_img.at<uint8_t>(i, j)) * 0.7; }
        }

    }

    return res_img;
}

/*Mat gr_plus_lines;
gr_plus_lines = transp_plus(cl_with_lines, gray_sys);
imshow("plus", gr_plus_lines);
imwrite(path + "gr_plus_lines_" + name, gr_plus_lines);
waitKey(0);

imshow("Gray Sys", gray_sys);
imwrite(path + "gray_" + name, gray_sys);
Mat gray_mine;
gray_mine = gray_img(img);
imshow("Gray Mine", gray_mine);
imwrite(path + "gray_" + name + "_mine.jpg", gray_mine);

Mat hist;
float range[] = { 0, 256 };
const float* histRange = { range };

bool uniform = true;
bool accumulate = false;

calcHist(&gray_sys, 1, 0, cv::Mat(), hist, 1, &hist_size, &histRange, uniform, accumulate);

draw_hist(hist, name, path);
Mat shist = smooth(hist);
draw_hist(shist, "smooth_" + name, path);

Mat filt_img = filter_img(gray_sys, hist);
imshow("Filtered", filt_img);
imwrite(path + "filt_" + name, filt_img);
Mat R_img = get_R_img(gray_sys, base_frame_size);
imshow("R", R_img);
imwrite(path + "R_map_" + name, R_img);
waitKey(0);

int L_arr[10] = {4, 8, 16, 32, 6, 7, 8, 9, 10};
Mat R_mult = get_R_multiplex(gray_sys, L_arr, 3);
imshow("R_mult", R_mult);
imshow("BW", black_white_R(R_mult));
imshow("Negative", negative(R_mult));
imwrite(path + "R_mult_" + name, R_mult);
waitKey(0);

Mat gray_bord, img_bord;
img_bord = imread(findFile("test_img.bmp"), IMREAD_COLOR);
cvtColor(img_bord, gray_bord, COLOR_BGR2GRAY);
img_bord = draw_border(gray_bord);
imshow("Border", img_bord);
imwrite(path + "test_img_w_border.bmp", img_bord);
waitKey(0);
return 0;*/
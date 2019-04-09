
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <dirent.h>
#include <fstream>

using namespace cv;

void doDFT(Mat *src, Mat *dst);
void scaleN(int N, void* pin);
int computeOutput(int x, int r1, int s1, int r2, int s2);
uint8_t *getGrayHistogram(Mat *src);
void createHistImage(Mat *src, Mat *dest);
int getHistCDF(Mat *src, float percentage);
int getThreshval(Mat *src);

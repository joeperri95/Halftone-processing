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

//struct used to pass a pair of images as context
//useful when using trackbars
typedef struct pair{
    Mat *in;
    Mat *out;
}pair;

void contrastStretch(Mat *src, Mat *dest, int r1, int s1, int r2, int s2);
void doDFT(Mat *src, Mat *dst);
void scaleN(int N, void* pin);
int computeOutput(int x, int r1, int s1, int r2, int s2);
void getGrayHistogram(Mat *src);
void createHistImage(Mat *src, Mat *dest);
int getHistMidpoint(Mat *src);

int main(int argc, char ** argv)
{
    //take command line argument else use standard image
    std::string filename = argc >=2 ? std::string(argv[1]) : "halftone.png";

    //input image
    Mat halftoneImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    Mat contoursImage, outputImage, freqImage;
    
    RNG rng(123456);

    outputImage = halftoneImage.clone();

    //Create GUI windows
    namedWindow("spectrum magnitude", CV_WINDOW_AUTOSIZE);
    moveWindow("spectrum magnitude", 500, 500);

    namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    moveWindow("Output Image", 0, 0);

    //canny filter output
    Mat canny;
    
    //contours containers
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    
    //used for drawing contours
    Mat drawing;

    //image container
    pair p;

    //thresholding value
    int thresh_val = 50;

    int num_contours = 0;

    //loop condition
    bool q = false;

    Mat freqOrig;
    Mat hist;

    //continue to scale until parameters are met
    while(!q){

        doDFT(&outputImage, &freqImage);
        freqOrig = freqImage.clone();

        Mat roi (freqImage,Rect(0, 0, 100, 100));
        createHistImage(&roi, &hist);

        for(int y = 0; y < freqImage.rows; y++){
            for(int x = 0; x < freqImage.cols; x++){
                
                int output = computeOutput(freqImage.at<uchar>(y,x), 90, 0, 170, 255);
                freqImage.at<uchar>(y,x) = saturate_cast<uchar>(output);
                //freqOrig.at<uchar>(y,x) = saturate_cast<uchar>(output);
            }
        }

        //need to decide which smoothing method to use
        //medianBlur(freqImage, freqImage, 3);
        GaussianBlur(freqImage, freqImage, Size(3,3), 1, 1);
        morphologyEx(freqImage, freqImage, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3,3)));
        threshold(freqImage, contoursImage, thresh_val, 255, THRESH_BINARY);

        findContours(contoursImage, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        drawing = Mat::zeros(contoursImage.size(), CV_8UC3);

        for(unsigned int i = 0; i < contours.size(); i++){
            if(hierarchy.at(i)[3] == -1){
                drawContours(drawing, contours, i, Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, hierarchy, 0);
                num_contours++;
            }
        }


        p = {&outputImage, &outputImage};

        //print number of contours found
        std::cout << num_contours << std::endl;

        //Display results
        imshow("Input Image"       , halftoneImage   ); 
        imshow("Output Image"       , outputImage );    
        imshow("thresholded", contoursImage);
        imshow("spectrum magnitude", drawing);
        imshow("frequency domain", freqOrig);
        imshow("hist", hist);

        //freq response may be clipped at edge of image
        //this allows for 2 contours lines at each edge
        if(num_contours > 3){
            
            //If dots found rescale image
            scaleN(1, (void*) &p);
            num_contours = 0;
            
            waitKey();
        }
        else{
            q = true;
        }

        waitKey(2);
    }

    waitKey();
    destroyAllWindows();

    imshow("final image", outputImage);
    waitKey();

    //create filename
    int outputNumber = 0;
    std::string outputFileName;
    outputFileName = "out" + std::to_string(outputNumber) + ".png";    
    
    std::ifstream f(outputFileName);
    //if file already exists increment number
    while(f.good()){
        outputNumber++;
        outputFileName.clear();
        outputFileName = "out" + std::to_string(outputNumber) + ".png";    
        f = std::ifstream(outputFileName);
    }
    
    //write png file
    std::vector<int> params;
    params.push_back(IMWRITE_PNG_STRATEGY_DEFAULT);
    
    try{
        imwrite( outputFileName , outputImage, params);
    }
    catch (Exception &e) {
        std::cerr << "error writing file " << e.what() << std::endl;
        return -1;
    }

    std::cout << "file written to: " << outputFileName << std::endl;

    return 0;

}   




void doDFT(Mat *src, Mat *dst){   
    
    Mat padded;                            
    //expand input image to optimal size
    int m = getOptimalDFTSize( src->rows );
    int n = getOptimalDFTSize( src->cols ); 
    // on the border add zero values
    copyMakeBorder(*src, padded, 0, m - src->rows, 0, n - src->cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    //switch to logarithmic scale
    magI += Scalar::all(1);                    
    log(magI, magI);

    //crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    //adjust the output image to get accurate representation
    Mat q0(magI, Rect(0, 0, cx, cy));   
    Mat q1(magI, Rect(cx, 0, cx, cy));      
    Mat q2(magI, Rect(0, cy, cx, cy));  
    Mat q3(magI, Rect(cx, cy, cx, cy)); 

    Mat tmp;                           
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);


    //convert to workable grayscale
    normalize(magI, magI, 0, 255, CV_MINMAX);
    magI.convertTo(magI, CV_8U);

    *dst = magI.clone();
}


void scaleN(int N, void* pin){

    //get pair of images from void pointer
    pair* p = (pair*) pin;

    int rows = (p->in)->rows;
    int cols = (p->in)->cols;

    if(N == 0){
        return;
    }

    //scale images at least once
    pyrDown(*(p->in), *(p->out), Size((cols + 1)/2, (rows + 1)/2));
    pyrUp(*(p->out), *(p->out), Size(cols , rows));

    //repeat for N times
    for(int i = 1; i < N; i++){
        pyrDown(*(p->out), *(p->out), Size((cols + 1)/2, (rows + 1)/2));
        pyrUp(*(p->out), *(p->out), Size(cols , rows));
    }

}


void contrastStretch(Mat *src, Mat *dest, int r1, int s1, int r2, int s2){
    
    *dest = Mat::zeros(src->size(), CV_8UC1);
    float result = 0.0;
    uchar x = 0;
    uchar y;

    for(int i = 0; i < src->rows; i++){
        for(int j = 0; j < src->cols; j++){
            
            //store for brevity
            x = src->at<uchar>(Point(i,j)); 
    
            if(0 <= x && x <= r1){
                result = s1/r1 * x;
            }else if(r1 < x && x <= r2){
                result = ((s2 - s1)/(r2 - r1)) * (x - r1) + s1;
            }else if(r2 < x && x <= 255){
                result = ((255 - s2)/(255 - r2)) * (x - r2) + s2;
            }

            y = (uchar) result;
            std::cout << x << ',' << y << std::endl;

            dest->at<uchar>(Point(i,j)) = y;
            
        }

    }
}

//contrast stretch one point
int computeOutput(int x, int r1, int s1, int r2, int s2)
{
    float result;
    if(0 <= x && x <= r1){
        result = s1/r1 * x;
    }else if(r1 < x && x <= r2){
        result = ((s2 - s1)/(r2 - r1)) * (x - r1) + s1;
    }else if(r2 < x && x <= 255){
        result = ((255 - s2)/(255 - r2)) * (x - r2) + s2;
    }
    return (int)result;
}

void getGrayHistogram(Mat *src){
    
    uint8_t hist[256];

    for(int i = 0; i < 256; i++){
        *(hist + i) = 0;
    }
    
    uchar k;

    for(int i = 0; i < src->rows; i++){
        for(int j = 0; j < src->cols; j++){
            k = src->at<uchar>(i,j);
            hist[(uint8_t)k]++;
        }
    }

    for(int i = 0; i < 256; i++){
        if(*(hist + i) != 0)
            printf("%d: %d\n", i ,*(hist + i));
    }
}  

int getHistMidpoint(Mat *src){
    uint8_t hist[256];
    int h = src->rows;
    int w = src->cols;

    int total = h * w;
    int acc = 0;

    for(int i = 0; i < 256; i++){
        *(hist + i) = 0;
    }
    
    uchar k;

    for(int i = 0; i < src->rows; i++){
        for(int j = 0; j < src->cols; j++){
            k = src->at<uchar>(i,j);
            hist[(uint8_t)k]++;
        }
    }

    for(int i = 0; i < 256; i++){
        acc += *(hist + i);
        if(acc > (total >> 1)){
            return i;
        }
    }

}

void createHistImage(Mat *src, Mat *dest){
    
    uint8_t hist[256];

    for(int i = 0; i < 256; i++){
        *(hist + i) = 0;
    }
    
    uchar k;

    for(int i = 0; i < src->rows; i++){
        for(int j = 0; j < src->cols; j++){
            k = src->at<uchar>(i,j);
            hist[(uint8_t)k]++;
        }
    }

    *dest = Mat::zeros(Size(512, 512), CV_8UC1); 

    int h;

    for(int i = 0; i < 256; i++){
        h = *(hist + i);
        line(*dest, Point(i << 1, 512), Point(i << 1, 512 - h), Scalar(255,255,255) , 2);
    }


}
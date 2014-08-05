//
//  main.cpp
//  eulerian_video_mag
//
//  Created by Zhiyuan Wang on 5/16/14.
//  Copyright (c) 2014 Zhiyuan Wang. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <string.h>
#include <iomanip>

#include <Python.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

template <class T>
void lappyr(T, int, vector<T> &, vector<Size> &);
void rgb2ntsc(Mat_<Vec3f>, Mat_<Vec3f> &);
void ntsc2rgb(Mat_<Vec3f>, Mat_<Vec3f> &);
char* amplify_spatial_lpyr_temporal_butter(const char* vidFile, const char* outDir, int alpha, int lambda_c, float fl, float fh, int samplingRate, float chromAttenuation, char*);
template <class T>
/* Image reconstruction from Laplace Pyramid Vector */
T reconLpyr(vector<T>);
/* compute maximum pyramid height of given image and filter sizes. */
int maxPyrHt(Size imsize, Size filtsize);
/* show result, compared with the input video */
void showresult(const char* in, const char* out);


void showresult(const char* in, const char* out)
{
    while (1)
    {
        VideoCapture src(in);
        VideoCapture dst(out);
        int vidWidth = dst.get(CV_CAP_PROP_FRAME_WIDTH);
        int vidHeight = dst.get(CV_CAP_PROP_FRAME_HEIGHT);
        assert(src.isOpened() || dst.isOpened());  // check if we succeeded
        namedWindow("Input Video", WINDOW_NORMAL);
        namedWindow("Output Video", WINDOW_NORMAL);
        moveWindow("Input Video", 40, 40);
        moveWindow("Output Video", 40 + vidWidth + 5, 40);
        for(;;)
        {
            Mat sfr, dfr;
            src >> sfr;
            dst >> dfr;
            if (sfr.empty() || dfr.empty())
                break;
            imshow("Input Video", sfr);
            imshow("Output Video", dfr);
            if(waitKey(30) >= 0) break;
        }
        if(waitKey(27) >= 0) break;
    }
    //return;
}

template <class T>
void showim(char* win, T img)
{
    namedWindow(win, CV_WINDOW_AUTOSIZE);
    while(1) {
        imshow(win, img);
        if (waitKey())
            break;
    }
    destroyWindow(win);
}

template <class T_p>
void showpyr(vector<T_p> pyr)
{
    for (int i = 0; i < pyr.size(); i++)
    {
        showim("Image Pyramids, press any key to continue", pyr[i]);
    }
}

/* embedding Python code */
const char *pycode =
    "from scipy import signal\n"
    "def bwlp(order, f, samplingRate):\n"
    "   b, a = signal.butter(order, f/samplingRate, 'low', analog=False)\n"
    "   return b, a\n"
    "a, b = bwlp(1, f, samplingRate)\n"
    "a0 = a[0]\n"
    "a1 = a[1]\n"
    "b0 = b[0]\n"
    "b1 = b[1]\n";

int main(int argc, const char * argv[])
{
    // insert code here...
    const char* inFile;
    const char* resultsDir;
    if (argc == 3)
        resultsDir = argv[2];
    else
        resultsDir = "./result";
        
    if (argc >= 2 )
        inFile = argv[1];
    else
        inFile = "subway.mp4";
    /*
    Mat_<Vec3f> src = imread(inFile);
    rgb2ntsc(src, src);
    //Mat_<Vec3f> lap;
    //namedWindow("show", 1);
    vector<Mat_<Vec3f> > lap_arr;
    vector<Size> nind;
    lappyr(src, 5, lap_arr, nind);
    // show laplacian result
    vector<Mat_<Vec3f> >::iterator it;
    //cout << lap_arr.empty() << endl;
    for (it = lap_arr.begin(); it != lap_arr.end(); it++)
    {
        showim("Lap", *it);
    }
    */
    /*
    // test rgb2ntsc
    Mat_<Vec3f> ntsc = Mat_<Vec3f>(src);
    rgb2ntsc(src, ntsc);
    showim("NTSC", ntsc);
    cout << ntsc.at<Vec3f>(0, 0).val[0] << endl;
    */
    char* baby = "baby.mp4";
    char* subway = "subway.mp4";
    char* outName = new char[128];
    if (strcmp(inFile, baby) == 0)
        amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 30, 16, 0.4, 3, 30, 0.1, outName);
    else if (strcmp(inFile, subway) == 0)
        amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 60, 90, 3.6, 6.2, 30, 0.3, outName);
    //cout << pos;
    //cout << outName << endl;
    showresult(inFile, outName);
    return 0;
}
/* level = 0 means 'auto', i.e. full stack */
template <class T>
void lappyr(T src, int level, vector<T> &lap_arr, vector<Size> & nind)
{
    lap_arr.clear();
    T down, up, dst;
    int max_ht = maxPyrHt(src.size(), cvSize(4,4));
    if (level == 0) // 'auto' full pyr stack
    {
        level = max_ht;
    }
    
    //cout << level << endl;
    
    for (int l = 0; l < level - 1; l++) {
        pyrDown(src, down);
        pyrUp(down, up, src.size());
        dst = src - up;
        lap_arr.push_back(dst);
        nind.push_back(dst.size());
        src = down;
        if (down.rows < 2)
        {
            cout << "Laplacian pyramid stop at level" << l << ", due to the image size" << endl;
        }
    }
    //pyrDown(src, down);
    //pyrUp(down, up, src.size());
    lap_arr.push_back(down);
    nind.push_back(down.size());
}

template <class _Tp>
Mat_<_Tp> reconLpyr(vector<Mat_<_Tp> > lpyr)
{
    Mat_<_Tp> dst = lpyr.back();
    //pyrUp(lpyr.back(), dst, cvSize(lpyr.back().cols*2, lpyr.back().rows*2));
    //cout << dst.rows << '\t' << dst.cols << endl;
    for (int i = lpyr.size()-2; i >= 0; i--)
    {
        //cout << lpyr[i].rows << '\t' << lpyr[i].cols << endl;
        //cout << i << endl;
        pyrUp(dst, dst, cvSize(lpyr[i].cols, lpyr[i].rows));
        dst += lpyr[i];
    }
    return dst;
}

/*
template <class T>
vector<Mat_<Vec3f> > operator* (vector<Mat_<Vec3f> > foo, T f)
{
    vector<Mat_<Vec3f> > r;
    vector<Mat_<Vec3f> >::iterator it;
    for (it = foo.begin(); it != foo.end(); it++)
    {
        //*it *= f;
        r.push_back(*it * f);
    }
    return r;
}
*/

template <class T>
vector<Mat_<Vec3f> > operator* (T f, vector<Mat_<Vec3f> > foo)
{
    vector<Mat_<Vec3f> > r;
    vector<Mat_<Vec3f> >::iterator it;
    for (it = foo.begin(); it != foo.end(); it++)
    {
        //*it *= f;
        r.push_back(*it * f);
    }
    return r;
}

template <class T>
vector<Mat_<Vec3f> > operator/ (vector<Mat_<Vec3f> > foo, T f)
{
    vector<Mat_<Vec3f> > r;
    vector<Mat_<Vec3f> >::iterator it;
    for (it = foo.begin(); it != foo.end(); it++)
    {
        //*it *= f;
        r.push_back(*it / f);
    }
    return r;
}


//template <class T>
vector<Mat_<Vec3f> > operator- (vector<Mat_<Vec3f> > a, vector<Mat_<Vec3f> > b)
{
    assert(a.size() == b.size());
    vector<Mat_<Vec3f> > r;
    vector<Mat_<Vec3f> >::iterator ita;
    vector<Mat_<Vec3f> >::iterator itb;
    for (ita = a.begin(), itb = b.begin(); ita != a.end(); ita++, itb++)
    {
        //*ita -= *itb;
        r.push_back(*ita - *itb);
    }
    return r;
}

vector<Mat_<Vec3f> > operator+ (vector<Mat_<Vec3f> > a, vector<Mat_<Vec3f> > b)
{
    //cout << "In operator+ a.size(): " << a.size() << endl;
    //cout << "In operator+ b.size(): " << b.size() << endl;
    assert(a.size() == b.size());
    vector<Mat_<Vec3f> > r;
    vector<Mat_<Vec3f> >::iterator ita;
    vector<Mat_<Vec3f> >::iterator itb;
    for (ita = a.begin(), itb = b.begin(); ita != a.end(); ita++, itb++)
    {
        //*ita += *itb;
        r.push_back(*ita + *itb);
    }
    return r;
}

char* amplify_spatial_lpyr_temporal_butter(const char* vidFile, const char* outDir, int alpha, int lambda_c, float fl, float fh, int samplingRate, float chromAttenuation, char* outName)
{
    /*
     [low_a, low_b] = butter(1, fl/samplingRate, 'low');
     [high_a, high_b] = butter(1, fh/samplingRate, 'low');
     */
    double low_a[2], low_b[2], high_a[2], high_b[2];
    // call python butter func
    PyObject *main_module, *main_dict;
    PyObject *b0_obj, *b1_obj, *a0_obj, *a1_obj, *f_obj, *samplerate_obj;
    
    Py_Initialize();
    
    /* Setup the __main__ module for us to use */
    main_module = PyImport_ImportModule("__main__");
    main_dict   = PyModule_GetDict(main_module);
    
    /* C var to Python var */
    samplerate_obj = PyInt_FromLong(long(samplingRate));
    f_obj = PyFloat_FromDouble(fl);
    /* inject var into __main__ */
    PyDict_SetItemString(main_dict, "f", f_obj);
    PyDict_SetItemString(main_dict, "samplingRate", samplerate_obj);
    
    PyRun_SimpleString(pycode);
    
    /* Extract the resultant var, double a, b */
    b0_obj = PyMapping_GetItemString(main_dict, "b0");
    a0_obj = PyMapping_GetItemString(main_dict, "a0");
    b1_obj = PyMapping_GetItemString(main_dict, "b1");
    a1_obj = PyMapping_GetItemString(main_dict, "a1");
    low_b[0] = PyFloat_AsDouble(b0_obj);
    low_b[1] = PyFloat_AsDouble(b1_obj);
    low_a[0] = PyFloat_AsDouble(a0_obj);
    low_a[1] = PyFloat_AsDouble(a1_obj);
    
    f_obj = PyFloat_FromDouble(fh);
    PyDict_SetItemString(main_dict, "f", f_obj);
    
    PyRun_SimpleString(pycode);
    b0_obj = PyMapping_GetItemString(main_dict, "b0");
    a0_obj = PyMapping_GetItemString(main_dict, "a0");
    b1_obj = PyMapping_GetItemString(main_dict, "b1");
    a1_obj = PyMapping_GetItemString(main_dict, "a1");
    high_b[0] = PyFloat_AsDouble(b0_obj);
    high_b[1] = PyFloat_AsDouble(b1_obj);
    high_a[0] = PyFloat_AsDouble(a0_obj);
    high_a[1] = PyFloat_AsDouble(a1_obj);

    
    //cout << low_a[0] << "\t" << low_a[1] << endl << low_b[0] << "\t" << low_b[1] << endl;
    //cout << high_a[0] << "\t" << high_a[1] << endl << high_b[0] << "\t" << high_b[1] << endl;
    
    Py_Finalize();

    // produce char* outName
    //cout << vidFile << endl;
    //cout << outDir << endl;
    //char outName[128];
    char* pos;
    strcpy(outName, outDir);
    if (outName[strlen(outName) - 1] != '/')
        outName[strlen(outName)] = '/';
    //cout << outName << endl;
    pos = strrchr(vidFile, '/');
    char tmp[128];
    if (pos)
        strcpy(tmp, pos);
    else
        strcpy(tmp, vidFile);
    strcat(outName, tmp);
    pos = strrchr(outName, '.');
    sprintf(pos, "-butter-from-%.1f-to-%.1f-alpha-%d-lambda_c-%d-chromAtn-%.1f.avi", fl, fh, alpha, lambda_c, chromAttenuation);
    //cout << outName << endl;
    
    // video reader
    cout << vidFile << endl;
    VideoCapture cap(vidFile);
    Mat rgbframe;
    Mat_<Vec3f> frame;
    if (!cap.isOpened())
    {
        throw "Error when reading video.";
    }
    int vidHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int vidWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double fr = cap.get(CV_CAP_PROP_FPS);
    int len = cap.get(CV_CAP_PROP_FRAME_COUNT);
    
    //cout << "Total number of frames: " << fr << endl;
    
    VideoWriter vidOut(outName, cap.get(CV_CAP_PROP_FOURCC), fr, cvSize(vidWidth, vidHeight), true);
    
    // firstFrame
    cap >> rgbframe;
    //cout << "rgbframe\t" << rgbframe.depth() << endl;
    //showim("rgbframe", rgbframe);
    frame = Mat_<Vec3f>(rgbframe);
    //Mat_<Vec3f> ntsc(frame.rows, frame.cols, CV_32FC3);
    rgb2ntsc( Mat_<Vec3f>(rgbframe), frame );
    vector<Mat_<Vec3f> > pyr, lowpass1, lowpass2, pyr_prev;
    vector<Size> pind;
    lappyr(frame, 0, pyr, pind);
    lowpass1 = lowpass2 = pyr_prev =  pyr;
    //vidOut.write(rgbframe);
    vidOut << rgbframe;
    
    int nLevels = pind.size();
    
    vector<Mat_<Vec3f> > filtered;
    cout << len << endl;
    // process each frame
    for (int i = 0;; i++)
    {
        cap >> rgbframe;
        rgb2ntsc(Mat_<Vec3f>(rgbframe), frame);
        if (rgbframe.empty())
            break;
        rgb2ntsc(rgbframe, frame);
        lappyr(frame, 0, pyr, pind);
        //imshow("rgbframe", rgbframe);
        // temporal filtering
        lowpass1 = (-high_b[1] * lowpass1 + high_a[0] * pyr + high_a[1] * pyr_prev) / high_b[0];
        lowpass2 = (-low_b[1] * lowpass2 + low_a[0] * pyr + low_a[1] * pyr_prev) / low_b[0];
        filtered = lowpass1 - lowpass2;
        
        pyr_prev = pyr;
        
        // /* amplify each spatial frequency bands according to Figure 6 of et al. paper */
        //int ind = pyr.size();
        
        double delta = (double)lambda_c / 8 / (1 + alpha);
        
        /* the factor to boost alpha above the bound et al. have in the paper (for better visualization) */
        int exaggeration_factor = 2;
        
        /* compute the representative wavelength lambda for the lowest spatial frequency 
           band of Laplacian pyramid */
        double lambda = sqrt(double(vidHeight*vidHeight + vidWidth * vidWidth)) / 3;  // 3 is experimental constant
        
        for (int l = nLevels - 1; l >=0; l--)
        {
            // indices = ind - prod(pind(l, :))+1:ind;
            // no need to calc indices
            // because the et al.'s matlab code build pyr stack
            // in one dimension points vec
            // while we are use vector container for each level
            /* compute modified alpha for this level */
            double currAlpha = (double)lambda / delta / 8 - 1;
            currAlpha *= exaggeration_factor;
            
            if (l == nLevels - 1 || l == 0)    // ignore the highest and lowest frequency band
                filtered[l] *= 0;
            else if (currAlpha > alpha)    // representative lambda exceeds
                filtered[l] *= alpha;
            else
                filtered[l] *= currAlpha;
                
            /* go one level down on pyramid, 
                representative lambda will reduce by factor of 2 */
            lambda /= 2;
        }
        
        /* Render on the input video */
        Mat_<Vec3f> output = Mat_<Vec3f>::zeros(frame.rows, frame.cols);
        output = reconLpyr(filtered);
        // test
        //showim("orign", rgbframe);
        //showim("reconstructed output", output);
        //showim("frame", frame);
        
        output += frame;
        //showim("output+frame", output);
        ntsc2rgb(output, output);
        //filtered = rgbframe + filtered.*mask;
        //showim("output", output);
        
        for (int row = 0; row < output.rows; row++)
        {
            for (int col = 0; col < output.cols; col++)
            {
                Vec3f *p = &output.at<Vec3f>(row, col);
                for (int c = 0; c < 3; c++)
                {
                    if (p->val[c] > 1)
                        p->val[c] = 1;
                    if (p->val[c] < 0)
                        p->val[c] = 0;
                }
            }
        }
        output *= 255;
        //imshow("output", output);
        cout << setiosflags(ios::fixed)<<setprecision(2) << i / double(len) * 100 << "%" << endl;
        //cout << i / len * 100 << "%" << endl;
        //showim("debug", output);
        //cout << output.cv::Mat::depth() << endl;
        Mat tmp;
        output.convertTo(tmp, CV_8UC3);
        //cout << output.depth() << endl;
        //showim("tmp_final", tmp);
        //cout << rgbframe.depth() << endl;
        vidOut << tmp;
    }
    return outName;
}

Size operator/(Size a, double b)
{
    a.height /= b;
    a.width /= b;
    return a;
}

bool operator<(Size a, Size b)
{
    return a.height < b.height or a.width < b.width;
}

Size floor(Size a)
{
    a.height = floor(a.height);
    a.width = floor(a.width);
    return a;
}

int maxPyrHt(Size imsz, Size filtsz)
{
    // assume 2D image
    if (imsz < filtsz)
        return 0;
    else
        return 1 + maxPyrHt( floor(imsz / 2), filtsz);
}

/* RGB 2 YIQ conversion, compared with Matlab, it works well */
void rgb2ntsc(Mat_<Vec3f> src, Mat_<Vec3f> &dst)
{
    //Mat_<Vec3f> cp;
    //src.copyTo(cp);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f i = src.at<Vec3f>(y, x);
            dst.at<Vec3f>(y, x).val[2] = (0.299 * i.val[2] + 0.587 * i.val[1] + 0.114 * i.val[0]) / 255;
            dst.at<Vec3f>(y, x).val[1] = (0.595716 * i.val[2] - 0.274453 * i.val[1] - 0.321263 * i.val[0]) / 255;
            dst.at<Vec3f>(y, x).val[0] = (0.211456 * i.val[2] - 0.522591 * i.val[1] + 0.311135 * i.val[0]) / 255;
        }
    }
}

/* YIQ 2 RGB convert */
void ntsc2rgb(Mat_<Vec3f> src, Mat_<Vec3f> &dst)
{
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3f i = src.at<Vec3f>(y, x);
            dst.at<Vec3f>(y, x).val[2] = (1 * i.val[2] + 0.9563 * i.val[1] + 0.621 * i.val[0]);
            dst.at<Vec3f>(y, x).val[1] = (1 * i.val[2] - 0.2721 * i.val[1] - 0.6474 * i.val[0]);
            dst.at<Vec3f>(y, x).val[0] = (1 * i.val[2] - 1.107 * i.val[1] + 1.7046 * i.val[0]);
        }
    }
}


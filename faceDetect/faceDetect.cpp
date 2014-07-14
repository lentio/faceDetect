// faceDetect.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "GaborFR.h"

#define FACEDETECT_DATA_PATH              "../data"
#define FACEDETECT_IMAGE_PATH_HAPPY        "../data/smilebak"
#define FACEDETECT_IMAGE_PATH_ANGRY        "../data/angrybak"
#define FACEDETECT_IMAGE_PATH_SAD          "../data/sadnessbak"
#define FACEDETECT_IMAGE_PATH_SURPRISE     "../data/surprise"
#define FACEDETECT_IMAGE_PATH_FEAR         "../data/fear"
#define FACEDETECT_IMAGE_PATH_DISGUST      "../data/disgust"
#define FACEDETECT_IMAGE_PATH_NEUTRALITY   "../data/neutrality"

#define FACEDETECT_NOMALIZE_IMAGE          "../data/KA.AN2.40.tiff" // YM.HA3.54.tiff" // test.tiff" //KL.AN1.167.tiff" // 

#define FACEDETECT_EMOTION_ID_HAPPY      0
#define FACEDETECT_EMOTION_ID_ANGRY      1
#define FACEDETECT_EMOTION_ID_SAD        2
#define FACEDETECT_EMOTION_ID_MAX        3

#define FACEDETECT_BOW_MAX_Fold          10

static char *arryEmotionPath[FACEDETECT_EMOTION_ID_MAX] = { FACEDETECT_IMAGE_PATH_HAPPY, FACEDETECT_IMAGE_PATH_ANGRY, FACEDETECT_IMAGE_PATH_SAD };

string face_cascade_name = "../data/haarcascade_frontalface_alt.xml";
string face_cascade_mouth_name = "../data/haarcascade_mcs_mouth.xml";

CascadeClassifier face_cascade;
CascadeClassifier mouth_cascade;
string window_name = "人脸识别";

void test_debug(char* format, ...)
{
    char buf[2048] = { 0 };

    va_list args;
    va_start(args, format);
    _vsnprintf_s(buf, sizeof(buf)-1, format, args);
    va_end(args);

    time_t now_time;
    now_time = time(NULL);

    struct tm *local;
    local = localtime(&now_time);  //获取当前系统时间  

    cout << asctime(local);
    cout << "[debug]" << buf << endl;
}

bool scanFiles(string szFolder, vector<string> &szFilesName)
{
    // loop all image files from the path
    _finddata_t fileDir;
    string searchDir = szFolder + "/*.*";
    long lfDir;

    //system("dir");
    if ((lfDir = _findfirst(searchDir.c_str(), &fileDir)) == -1l) {
        cout << "No file is found at " << szFolder << endl;
    }
    else {
        do{
            // skip . and .. and 
            if (strcmp(fileDir.name, ".") == 0
                || strcmp(fileDir.name, "..") == 0
                || fileDir.attrib &_A_SUBDIR)
            {
                continue;
            }

            //cout << fileDir.name << endl;
            string szFoundFile = szFolder + "/" + fileDir.name;
            cout << "find a file " << szFoundFile << " from " << szFolder << endl;

            // save
            szFilesName.push_back(szFoundFile);
        } while (_findnext(lfDir, &fileDir) == 0);
    }

    _findclose(lfDir);
    return true;
}

bool detectMouth(Mat frame, Mat &scaleMouth);

Mat extract_features(string fileImage)
{
    cout << "[info ] extract file: " << fileImage << endl;
    Mat mImage = imread(fileImage, CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat mInput;
    if (!detectMouth(mImage, mInput)){
        cout << "[error] fail to find mouth" << endl;
        assert(0);
    }

    // Mat image = cvLoadImage(fileImage, -1);
    normalize(mInput, mInput, 1, 0, CV_MINMAX, CV_32F);
    assert(mInput.rows != 0);

    Mat mGaborM, mGaborOutput;
    Mat mResult;
    int iSize = 50;//如果数值比较大，比如50则接近论文中所述的情况了！估计大小和处理的源图像一样！

    // by8x5个gabor滤波器
    for (int i = 0; i < 8; i++)
    {
        mGaborM.release();
        for (int j = 0; j < 5; j++)
        {
            Mat mKernelReal = GaborFR::getRealGaborKernel(Size(iSize, iSize), 2 * CV_PI, i*CV_PI / 8 + CV_PI / 2, j, 1);
            Mat mKernelImag = GaborFR::getImagGaborKernel(Size(iSize, iSize), 2 * CV_PI, i*CV_PI / 8 + CV_PI / 2, j, 1);

            //加了CV_PI/2才和大部分文献的图形一样，不知道为什么！
            Mat outR, outI;
            GaborFR::getFilterRealImagPart(mInput, mKernelReal, mKernelImag, outR, outI);
            //			M=GaborFR::getPhase(M1,M2);
            //			M=GaborFR::getMagnitude(M1,M2);
            //			M=GaborFR::getPhase(outR,outI);
            //			M=GaborFR::getMagnitude(outR,outI);
            //			M=GaborFR::getMagnitude(outR,outI);
            // 			MatTemp2=GaborFR::getPhase(outR,outI);

            mResult = outR;
            // M=M1;
            // resize(M,M,Size(100,100));
            normalize(mResult, mResult, 0, 255, CV_MINMAX, CV_8U);
            mGaborM.push_back(mResult);
        }

        mGaborM = mGaborM.t();
        mGaborOutput.push_back(mGaborM);
    }

    return mGaborOutput.reshape(0, 1);
}

void svm_train(Mat &train_data, Mat &train_lable)
{
    CvSVMParams svmParams;
    svmParams.svm_type = CvSVM::C_SVC;
    svmParams.kernel_type = CvSVM::RBF;
    CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
    c_grid = CvSVM::get_default_grid(CvSVM::GAMMA);
    p_grid = CvSVM::get_default_grid(CvSVM::P);
    p_grid.step = 0;
    nu_grid = CvSVM::get_default_grid(CvSVM::NU);
    nu_grid.step = 0;
    coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
    coef_grid.step = 0;
    degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
    degree_grid.step = 0;

    SVM svm;
    svm.train_auto(train_data, train_lable, Mat(), Mat(), svmParams, FACEDETECT_BOW_MAX_Fold, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
    string svmSaver = (string(FACEDETECT_DATA_PATH) + "\\svm-trained.xml").c_str();
    svm.save((string(FACEDETECT_DATA_PATH) + "\\svm-trained.xml").c_str());
}

int svm_predict(Mat descriptor)
{
    SVM svm;
    svm.load((string(FACEDETECT_DATA_PATH) + "\\svm-trained.xml").c_str());

    Mat train_data(descriptor.rows, descriptor.cols, CV_32FC1);
    // cout << "verify data: " << train_data.rows << " x " << train_data.cols << " type: " << train_data.type() << endl;

    Mat subMat = train_data.row(0);
    descriptor.copyTo(subMat);


    cout << "verify data: " << train_data.rows << " x " << train_data.cols << " type: " << train_data.type() << endl;

    float score = 0.0f;
    try {
        score = svm.predict(train_data, true);
    }
    catch (exception& e) {
        cout << "SVM异常:" << e.what() << endl;
    }

    return (int)(score + 0.5);
}

void test_verify(vector<Mat> &descriptors, vector<int> &lables)
{
    int nSuccess = 0;
    int nTotle = descriptors.size();
    for (int i = 0; i < descriptors.size(); ++i) {
        int nScore = svm_predict(descriptors[i]);

        // cout << "svm predict " << nScore << " with real lable " << lables[i] << endl;
        test_debug("svm predict %d with real %d", nScore, lables[i]);
        if (nScore == lables[i]) {
            ++nSuccess;
        }
    }

    cout << "totle: " << nTotle << ", success: " << nSuccess << ", rate: " << nSuccess * 100 / nTotle << "%" << endl;
}

void test_train()
{
    // load happy images from the folder
    cout << "[info] test_main load images" << endl;
    map<int, vector<string>> mapFiles;
    for (int i = 0; i < FACEDETECT_EMOTION_ID_MAX; ++i){
        vector<string> szHappyFiles;
        scanFiles(arryEmotionPath[i], szHappyFiles);
        mapFiles.insert(pair<int, vector<string>>(i, szHappyFiles));
    }

    cout << "[info] test_main load cascade" << endl;
    if (!face_cascade.load(face_cascade_name)){
        cout << "[error] 无法加载级联分类器文件！\n" << endl;
        return;
    }

    if (!mouth_cascade.load(face_cascade_mouth_name)){
        cout << "[error] 无法加载级联分类器文件！\n" << endl;
        return;
    }

    // extracting features
    cout << "[info] test_main extract features" << endl;
    vector<Mat> descriptors;
    vector<int> lables;
    for (auto obj = mapFiles.begin(); obj != mapFiles.end(); ++obj) {
        vector<string> lstFileses = obj->second;
        int nLable = obj->first;
        for (auto filePath = lstFileses.begin(); filePath != lstFileses.end(); ++filePath) {
            Mat mFeature = extract_features(*filePath);
            descriptors.push_back(mFeature);
            lables.push_back(nLable);

            cout << "get gabor feature: " << mFeature.rows << " x " << mFeature.cols << " type:" << mFeature.type() << " lable: " << nLable << " of file " << *filePath << endl;
        }
    }

    // 填充train_data结构
    int nRow = descriptors.size();
    int nCol = descriptors[0].cols;
    Mat train_data(nRow, nCol, CV_32FC1);
    // cout << "descriptor: " << 1 << " x " << nCol << " type:" << descriptors[0].type() << endl;

    for (int i = 0; i < descriptors.size(); ++i) {
        Mat sub_mat = train_data.row(i);
        descriptors[i].copyTo(sub_mat);
    }

    cout << "train data: " << nRow << " x " << nCol << " type:" << train_data.type() << endl;

    Mat train_lable(descriptors.size(), 1, CV_32SC1);
    for (int j = 0; j < lables.size(); ++j){
        train_lable.at<int>(j) = lables[j];
    }

    test_debug("train with SVM...");
    svm_train(train_data, train_lable);

    test_debug("verify with SVM...");
    test_verify(descriptors, lables);
}

void test_camera()
{

}

void detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (int i = 0; i < faces.size(); i++){
        Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

        Point start(faces[i].x + faces[i].width*0.05, faces[i].y);
        Point stop(start.x + faces[i].width*0.9, faces[i].y + faces[i].height);
        rectangle(frame, start, stop, Scalar(255, 0, 255), 4, 8, 0);

        Mat cutFace = Mat(frame, Rect(faces[i].x + faces[i].width*0.05, faces[i].y, faces[i].width*0.9, faces[i].height));
        imshow("focus face", cutFace);
    }

    imshow(window_name, frame);
}

bool getFaceRect(Mat frame, Rect &face)
{
    std::vector<Rect> faces;
    
    Mat frame_gray=frame;
    //cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (int i = 0; i < faces.size(); i++){
        //Point start(faces[i].x + faces[i].width*0.05, faces[i].y);
        //Point stop(start.x + faces[i].width*0.9, faces[i].y + faces[i].height);
        //rectangle(frame, start, stop, Scalar(255, 0, 255), 4, 8, 0);
        face = faces[i];
        return true;
    }

    return false;
}

void test_detect()
{
    Mat image;
    image = imread(FACEDETECT_NOMALIZE_IMAGE, CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data){
        cout << "[error] 没有图片" << endl;
        return;
    }

    if (!face_cascade.load(face_cascade_name)){
        cout << "[error] 无法加载级联分类器文件！\n" << endl;
        return;
    }

    detectAndDisplay(image);
    waitKey(0);
}

bool cuteImage(Mat frame, Mat &scaleFace)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (int i = 0; i < faces.size(); i++){
        Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        //ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

        Point start(faces[i].x + faces[i].width*0.05, faces[i].y);
        Point stop(start.x + faces[i].width*0.9, faces[i].y + faces[i].height);
        //rectangle(frame, start, stop, Scalar(255, 0, 255), 4, 8, 0);

        Mat cutFace = Mat(frame, Rect(faces[i].x + faces[i].width*0.05, faces[i].y, faces[i].width*0.9, faces[i].height));

        // scale into 90x100
        Size dsize = Size(90, 100);
        scaleFace = Mat(dsize, CV_32S);
        resize(cutFace, scaleFace, dsize);

        imshow("focus face", scaleFace);
        return true;
    }

    return false;
}

void test_cut()
{
    vector<string> szHappyFiles;
    scanFiles(arryEmotionPath[2], szHappyFiles);

    for (int i = 0; i < szHappyFiles.size(); ++i){
        Mat image;
        image = imread(szHappyFiles[i], CV_LOAD_IMAGE_GRAYSCALE);
        if (!image.data){
            cout << "[error] 没有图片" << endl;
            continue;
        }

        if (!face_cascade.load(face_cascade_name)){
            cout << "[error] 无法加载级联分类器文件！\n" << endl;
            return;
        }

        Mat cutFace;
        if (cuteImage(image, cutFace)){
            string szSaveFile = szHappyFiles[i] + ".bmp";
            imwrite(szSaveFile, cutFace);
            cout << "cute a face into file: " << szSaveFile << endl;
        }
    }
}

bool detectMouth(Mat frame, Mat &scaleMouth)
{
    Rect face;
    if (!getFaceRect(frame, face)){
        cout << "[error] fail to get face rect " << endl;
        return false;
    }

    std::vector<Rect> faces;
   
    Mat frame_gray=frame;
    //cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //检测嘴巴  
    vector<cv::Rect> mouthVec;
    cv::Rect halfRect = face;
    halfRect.height /= 2;
    halfRect.y += halfRect.height;

    Mat halfFace = frame_gray(halfRect);
    mouth_cascade.detectMultiScale(halfFace, mouthVec, 1.1);
    for (int j = 0; j<mouthVec.size(); j++)
    {
        cv::Rect rect = mouthVec[j];
        rect.x += halfRect.x;
        rect.y += halfRect.y;
        rectangle(frame_gray, rect, CV_RGB(255, 255, 255), 2);

        Mat cutFace = Mat(frame_gray, rect);

        // scale into 90x100
        Size dsize = Size(7, 4);
        scaleMouth = Mat(dsize, CV_32S);
        resize(cutFace, scaleMouth, dsize);

        //imshow(window_name, scaleMouth);
        return true;
    }

    /*
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    for (int i = 0; i < faces.size(); i++){
        //Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        //ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

        //Point start(faces[i].x , faces[i].y);
        //Point stop(start.x + faces[i].width*0.9, faces[i].y + faces[i].height);
        if (i == 2){ // 嘴巴
            rectangle(frame, faces[i], Scalar(255, 0, 255), 4, 8, 0);
            //imshow(window_name, frame);

            Mat cutFace = Mat(frame, faces[i]);

            // scale into 90x100
            Size dsize = Size(70, 40);
            scaleMouth = Mat(dsize, CV_32S);
            resize(cutFace, scaleMouth, dsize);

            //imshow(window_name, scaleMouth);
            //return true;
        }
        else{
            rectangle(frame, faces[i], Scalar(255, 0, 255), 4, 8, 0);
        }
    }
    */

    imshow(window_name, frame);
    return false;
}

void test_mouth()
{
    Mat image, mouth;
    image = imread(FACEDETECT_NOMALIZE_IMAGE, CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data){
        cout << "[error] 没有图片" << endl;
        return;
    }

    if (!face_cascade.load(face_cascade_name)){
        cout << "[error] 无法加载级联分类器文件！\n" << endl;
        return;
    }

    if (!mouth_cascade.load(face_cascade_mouth_name)){
        cout << "[error] 无法加载级联分类器文件！\n" << endl;
        return;
    }

    if (!detectMouth(image, mouth)){
        cout << "[error] fail to find mouth" << endl;
    }

    waitKey(0);
}

void showError()
{
    cerr << "参数错误" << endl;
    cerr << "usage:" << endl;
    cerr << "faceDetect -train  ---训练" << endl;
    cerr << "faceDetect -camera ---摄像头采集" << endl;
    system("pause");
}

int _tmain(int argc, _TCHAR* argv[])
{ 
    if (argc != 2) {
        showError();
        exit(__LINE__);
    }

    cout << "excute: " << argv[0] << " " << argv[1] << endl;
    if (!strcmp(argv[1], _T("-train"))) {
        cout << "当前模式:训练" << endl;
        test_train();
    }
    else if (!strcmp(argv[1], _T("-camera"))) {
        cout << "当前模式:摄像头采集预测" << endl;
        test_camera();
    }
    else if (!strcmp(argv[1], _T("-detect_face"))) {
        cout << "当前模式:人脸检测" << endl;
        test_detect();
    }
    else if (!strcmp(argv[1], _T("-cut_face"))) {
        cout << "当前模式:人脸剪切" << endl;
        test_cut();
    }
    else if (!strcmp(argv[1], _T("-detect_mouth"))) {
        cout << "当前模式:嘴巴检测" << endl;
        test_mouth();
    }
    else {
        showError();
    }

    system("pause");
}


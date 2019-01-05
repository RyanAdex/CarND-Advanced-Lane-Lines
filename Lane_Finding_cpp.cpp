#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <io.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/calib3d.hpp>
using namespace std;
/*
@param fileDir 为文件夹目录
@param fileType 为需要查找的文件类型
@param fileName 为存放文件名的容器
@NOTE 该函数为遍历文件夹中.fileType格式的文件，并存入vector<string>容器中
@FIXME C++ 17 可用filesystem库，但在该程序未调试成功
*/
bool get_images_by_dir(const string& fileDir,const string& fileType,vector<string>& fileName){
    string buffer = fileDir+"\\*"+fileType;
    _finddata_t c_file;//存放文件名的结构体
    long hFile;
    hFile=_findfirst(buffer.c_str(),&c_file);//找到第一个文件名
    if(hFile!=-1L){//检查文件夹目录下存在cou't文件
        string fullFilePath;
        do{
            fullFilePath.clear();
            //名字
            fullFilePath = fileDir+"\\"+c_file.name;
            fileName.push_back(fullFilePath);
        }while(_findnext(hFile,&c_file)==0);//如果找到下个文件的名字成功返回0，否则返回-1
        _findclose(hFile);
        return true;
    }
    return false;
}
/*
@param images 为存放图片路径的容器
@param grid 为棋盘格行列交点数，默认为（9，6）
@param distance 实际测量得到的标定板上每个棋盘格的物理尺寸，单位mm
@param object_points 为保存所有图标定板上角点的三角坐标的容器
@param img_points 为保存检测到的所有角点的容器
@NOTE 该函数用于得到棋盘格内的世界坐标"object points"和对应图片坐标"image points"
@NOTE 对于每张棋盘格图片组的图片而言，对应"object points"都是一样
*/
void get_obj_img_points(const vector<string> & images,const cv::Size & grid,const cv::Size& distance,cv::Mat& cameraMatirx,cv::Mat& distCoeffs){
    cv::Mat img,gray;//灰度图像
    vector<cv::Point2f> corners;//用来储存t图片角点
    vector<cv::Point3f> object_point;//保存标定板上所有角点坐标
    vector<cv::Mat> rvecs,tvecs;//旋转向量和位移向量
    vector<vector<cv::Point3f>> object_points;//棋盘格三维坐标容器
    vector<vector<cv::Point2f>> img_points;//棋盘格角点容器
    for(auto & imgdir:images){
        //载入图像
        img=cv::imread(imgdir);
        //生成object points
        for(int i=0;i<grid.height;i++){
            for(int j=0;j<grid.width;j++){
                object_point.push_back(cv::Point3f(i*distance.width,j*distance.height,0));//向容器存入每个角点坐标
            }
        }
        //得到灰度图片
        cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
        //得到图片的image points
        //NOTE corners的储存方式为从左往右，从上往下每行储存，所以储存object_point的时候需从grid。width开始遍历储存
        bool ret=cv::findChessboardCorners(gray,grid,corners,cv::CALIB_CB_ADAPTIVE_THRESH+cv::CALIB_CB_NORMALIZE_IMAGE+cv::CALIB_CB_FAST_CHECK);
        if(ret){//亚像素精细化
            cv::cornerSubPix(gray,corners,cv::Size(11,11),cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.1));
            img_points.push_back(corners);
            object_points.push_back(object_point);
        }
        object_point.clear();//清空object_point以便下一幅图使用该容器
        //绘制角点并显示
        cv::drawChessboardCorners(img,grid,cv::Mat(corners),ret);
        // cv::imshow("chessboard corners",img);
        // cv::waitKey(10);
    }
    cv::calibrateCamera(object_points,img_points,img.size(),cameraMatirx,distCoeffs,rvecs,tvecs);
}
/*
@param src 为需要变换的图片路径
@param dst 为输出的图像
@param objec_points 为保存所有图标定板上角点的三角坐标的容器
@param img_points 为保存检测到的所有角点的容器
@NOTE 该函数用于矫正图像
*/
void cal_undistort(const cv::Mat& src,const vector<vector<cv::Point3f>>& object_points,const vector<vector<cv::Point2f>>& img_points,cv::Mat& cameraMatirx,cv::Mat& distCoeffs){
    // cv::Mat cameraMatirx;//内参矩阵，需初始化
    // cv::Mat distCoeffs;//畸变矩阵，需初始化
    vector<cv::Mat> rvecs,tvecs;//旋转向量和位移向量
    //标定函数
    cv::calibrateCamera(object_points,img_points,src.size(),cameraMatirx,distCoeffs,rvecs,tvecs);
    //矫正函数
    // cv::undistort(src,dst,cameraMatirx,distCoeffs);
}
/*
@param src 为原始图像
@param dst 为输出的图像
@param orient 计算'x'/'y'方向的导
@param thresh_min 为阈值
@param thresh_max 为最大值
@NOTE 该函数用于对图像进行Sobel边缘检测，内部使用的二值化默认运算方法为cv::THRESH_BINARY|cv::THRESH_OTSU
*/
void abs_sobel_thresh(const cv::Mat& src,cv::Mat& dst,const char& orient='x',const int& thresh_min=0,const int& thresh_max=255){
    cv::Mat src_gray,grad;
    cv::Mat abs_gray;
    //转换成为灰度图片
    cv::cvtColor(src,src_gray,cv::COLOR_RGB2GRAY);
    //使用cv::Sobel()计算x方向或y方向的导
    if(orient=='x'){
        cv::Sobel(src_gray,grad,CV_64F,1,0);
        cv::convertScaleAbs(grad,abs_gray);
    }
    if(orient=='y'){
        cv::Sobel(src_gray,grad,CV_64F,0,1);
        cv::convertScaleAbs(grad,abs_gray);
    }
    //二值化
    cv::inRange(abs_gray,thresh_min,thresh_max,dst);
    // cv::threshold(abs_gray,dst,thresh_min,thresh_max,cv::THRESH_BINARY|cv::THRESH_OTSU);
}
/*
@param src 为原始图像
@param dst 为输出的图像
@param sobel_kernel 为计算核心
@param thresh_min 为阈值
@param thresh_max 为最大值
@NOTE 该函数用全局颜色变化梯度来进行阈值过滤，内部使用的二值化默认运算方法为cv::THRESH_BINARY|cv::THRESH_OTSU
*/
void mag_thresh(const cv::Mat& src,cv::Mat& dst,const int& sobel_kernel=3,const int& thresh_min=0,const int& thresh_max=255){
    cv::Mat src_gray,gray_x,gray_y,grad;
    cv::Mat abs_gray_x,abs_gray_y;
    //转换成为灰度图片
    cv::cvtColor(src,src_gray,cv::COLOR_RGB2GRAY);
    //使用cv::Sobel()计算x方向或y方向的导
    cv::Sobel(src_gray,gray_x,CV_64F,1,0,sobel_kernel);
    cv::Sobel(src_gray,gray_y,CV_64F,0,1,sobel_kernel);
    //转换成CV_8U
    cv::convertScaleAbs(gray_x,abs_gray_x);
    cv::convertScaleAbs(gray_y,abs_gray_y);
    //合并x和y方向的梯度
    cv::addWeighted(abs_gray_x,0.5,abs_gray_y,0.5,0,grad);
    //二值化
    cv::inRange(grad,thresh_min,thresh_max,dst);
    // cv::threshold(grad,dst,thresh_min,thresh_max,cv::THRESH_BINARY|cv::THRESH_OTSU);

}
/*
@param src 为原始图像
@param dst 为输出的图像
@param channel 为通道选择
@param thresh_min 为阈值
@param thresh_max 为最大值
@NOTE 该函数用与提取HLS通道
*/
void hls_select(const cv::Mat& src,cv::Mat& dst,const char& channel='s',const int& thresh_min=0,const int& thresh_max=255){
    cv::Mat hls,grad;
    vector<cv::Mat> channels;
    cv::cvtColor(src,hls,cv::COLOR_RGB2HLS);
    //分离通道
    cv::split(hls,channels);
    //选择通道
    switch (channel)
    {
        case 'h':
            grad=channels.at(0);
            break;
        case 'l':
            grad=channels.at(1);
            break;
        case 's':
            grad=channels.at(2);
            break;
        default:
            break;
    }
    //二值化
    cv::inRange(grad,thresh_min,thresh_max,dst);
    // cv::threshold(grad,dst,thresh_min,thresh_max,cv::THRESH_BINARY);
}
/*
@param src 为原始图像
@param dst 为输出的图像
@param sobel_kernel 为soble算子选择
@param thresh_min 为阈值
@param thresh_max 为最大值
@NOTE 该函数用于计算图像梯度角度
*/
void dir_threshold(const cv::Mat& src,cv::Mat& dst,const int& sobel_kernel=3,const float& thresh_min=0,const float& thresh_max=CV_PI/2){
    cv::Mat src_gray,gray_x,gray_y,abs_gray_x,abs_gray_y,grad;
    cv::cvtColor(src,src_gray,cv::COLOR_RGB2GRAY);
    // cv::spatialGradient(gray,gray_x,gray_y,sobel_kernel);
    //使用cv::Sobel()计算x方向或y方向的导
    cv::Sobel(src_gray,gray_x,CV_64F,1,0,sobel_kernel);
    cv::Sobel(src_gray,gray_y,CV_64F,0,1,sobel_kernel);
    //转换成CV_8U绝对值化
    cv::convertScaleAbs(gray_x,abs_gray_x);
    cv::convertScaleAbs(gray_y,abs_gray_y);
    int nl=src_gray.rows;
    int nc=src_gray.cols*src_gray.channels();
    
    float d;
    grad=cv::Mat::zeros(nl,nc,CV_8UC1);
    //遍历图像每个像素
    for(int j=0;j<nl;++j){
        const uchar *data_x = abs_gray_x.ptr<uchar>(j);
        const uchar *data_y = abs_gray_y.ptr<uchar>(j);
        uchar *data = grad.ptr<uchar>(j);
        for(int i=0;i<nc;++i){
            d=atan2(data_y[i],data_x[i]);
            
            switch (d>thresh_min&&d<thresh_max)
            {
                case true:
                    data[i]=255;
                    break;
            
                default:
                    data[i]=0;
                    break;
            };
        }
    }
    dst=grad;
    // cv::threshold(grad,dst,thresh_min,255,cv::THRESH_BINARY|cv::THRESH_OTSU);

}
/*
@param src 为原始图像
@param dst 为输出的图像
@param channel 为通道选择
@param thresh_min 为阈值
@param thresh_max 为最大值
@NOTE 该函数用与提取LUV通道
*/
void luv_select(const cv::Mat& src,cv::Mat& dst,const char& channel='l',const int& thresh_min=0,const int& thresh_max=255){
    cv::Mat luv,grad;
    vector<cv::Mat> channels;
    cv::cvtColor(src,luv,cv::COLOR_RGB2Luv);
    cv::split(luv,channels);
    
    switch (channel)
    {
        case 'l':
            grad=channels.at(0);
            break;
        case 'u':
            grad=channels.at(1);
            break;
        case 'v':
            grad=channels.at(2);
            break;
    }
    cv::inRange(grad,thresh_min,thresh_max,dst);
    // cv::threshold(grad,dst,thresh_min,thresh_max,cv::THRESH_BINARY);
}
/*
@param src 为原始图像
@param dst 为输出的图像
@param channel 为通道选择
@param thresh_min 为阈值
@param thresh_max 为最大值
@NOTE 该函数用与提取LAB通道
*/
void lab_select(const cv::Mat& src,cv::Mat& dst,const char& channel='b',const int& thresh_min=0,const int& thresh_max=255){
    cv::Mat lab,grad;
    vector<cv::Mat> channels;
    cv::cvtColor(src,lab,cv::COLOR_RGB2Lab);
    cv::split(lab,channels);
    
    switch (channel)
    {
        case 'l':
            grad=channels.at(0);
            break;
        case 'a':
            grad=channels.at(1);
            break;
        case 'b':
            grad=channels.at(2);
            break;
    }
    cv::inRange(grad,thresh_min,thresh_max,dst);
    
    // cv::threshold(grad,dst,thresh_min,thresh_max,cv::THRESH_BINARY);
}
/*
@param in_point 为离散坐标点
@param n 为n次多项式
@param mat_k 为返回多项式的k系数，为n*1的矩阵
@NOTE 该函数用于拟合曲线
*/
cv::Mat polyfit(vector<cv::Point>& in_point, int n){
    int size = in_point.size();	//所求未知数个数	
    int x_num = n + 1;	//构造矩阵U和Y	
    cv::Mat mat_u(size, x_num, CV_64F);	
    cv::Mat mat_y(size, 1, CV_64F); 	
    for (int i = 0; i < mat_u.rows; ++i){		
        for (int j = 0; j < mat_u.cols; ++j){
            mat_u.at<double>(i, j) = pow(in_point[i].y, j);//in_point[i].y为以y为递增坐标
        }
    }	
    for (int i = 0; i < mat_y.rows; ++i){
        mat_y.at<double>(i, 0) = in_point[i].x;
    } 	//矩阵运算，获得系数矩阵K	
    cv::Mat mat_k(x_num, 1, CV_64F);
    mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
    // cout << mat_k << endl;	
    return mat_k;
}
/*
@param mat_k 为多项式的系数矩阵
@param src 为需要计算的坐标点
@param n 为计算n次多项式
@NOTE 该函数用于计算拟合过后的曲线点坐标，该函数以y为基础坐标轴，如果要以x为基础坐标轴，则修改y为x
*/
vector<cv::Point> polyval(const cv::Mat& mat_k,const vector<cv::Point>& src,int n){
    vector<cv::Point> ip;
    // cout<<src.back().y<<"kkk"<<src.front().y<<endl;
    for(int i=src.back().y;i<src.front().y;i++){//从y=0开始计算，分别计算出x的值
        cv::Point ipt;
        ipt.x=0;
        ipt.y=i;
        for(int j=0;j<n+1;j++){
            ipt.x+=mat_k.at<double>(j,0)*pow(i,j);//NOTE多项式计算
        }
        ip.push_back(ipt);
    }
    return ip;
}
/*
@param src 为图像透视变形基础点
@param dst 为图像透视变形目标点
@param M 输出为变形系数矩阵
@param Minv 输出为反变形系数矩阵，还原变形使用
@NOTE 该函数用于计算变形系数矩阵和反变形系数矩阵
*/
void get_M_Minv(const vector<cv::Point2f>& src,const vector<cv::Point2f>& dst,cv::Mat& M,cv::Mat& Minv){
    M=cv::getPerspectiveTransform(src,dst);
    Minv=cv::getPerspectiveTransform(dst,src);
}
/*
@param src 为需要计算车道线的变形后且二值化的图像
@param lp 输出为左车道线坐标
@param rp 输出为右车道线坐标
@param rightx_current 输出为左车道线基准x坐标
@param leftx_current 输出为右车道线基准x坐标
@param distance_from_center 输出为车道偏离距离
@NOTE 该函数用于计算车道线坐标点以及偏离距离
*/
void find_line(const cv::Mat& src,vector<cv::Point>& lp,vector<cv::Point>& rp,int& rightx_current,int& leftx_current,double& distance_from_center){
    cv::Mat hist,nonzero,l,r;
    vector<cv::Point> nonzerol,nonzeror,lpoint,rpoint;
    int midpoint;
    cv::Point leftx_base,rightx_base;
    //选择滑窗个数
    int nwindows = 9;
    //设置窗口高度
    int window_height = int(src.rows/nwindows);
    //设置窗口宽度
    int margin=50;
    //设置非零像素坐标最少个数
    int minpix=50;
    //TODO 加入if设置图像连续性，如果leftx_current和rightx_current为零，则认为第一次执行，需要计算该两点，如果已经计算了，则不许再次计算。
    //rowrange图像区域分割
    //将图像处理为一行，以行相加为方法    
    cv::reduce(src.rowRange(src.rows/2,src.rows),hist,0,cv::REDUCE_SUM,CV_32S);
    midpoint=int(hist.cols/2);
    //将hist分为左右分别储存，并找出最大值
    //minMaxIdx针对多通道，minMaxLoc针对单通道
    cv::minMaxLoc(hist.colRange(0,midpoint),NULL,NULL,NULL,&leftx_base);
    cv::minMaxLoc(hist.colRange(midpoint,hist.cols),NULL,NULL,NULL,&rightx_base);
    //左右车道线基础点
    leftx_current=leftx_base.x;
    rightx_current=rightx_base.x+midpoint;
    // 提前存入该基础点坐标
    lpoint.push_back(cv::Point(leftx_current,src.rows));
    rpoint.push_back(cv::Point(rightx_current,src.rows));
    for(int i=0;i<nwindows;i++){
        int win_y_low=src.rows-(i+1)*window_height;
        //计算选框x坐标点，并将计算结果限制在图像坐标内
        int win_xleft_low = leftx_current - margin;
        win_xleft_low=win_xleft_low>0?win_xleft_low:0;
        win_xleft_low=win_xleft_low<src.rows?win_xleft_low:src.rows;
        //int win_xleft_high = leftx_current + margin;
        int win_xright_low = rightx_current - margin;
        win_xright_low=win_xright_low>0?win_xright_low:0;
        win_xright_low=win_xright_low<src.rows?win_xright_low:src.rows;
        //int win_xright_high = rightx_current + margin;
        //NOTE要确保参数都大于0，且在src图像范围内，不然会报错
        //NOTE 设置为ROI矩形区域选择
        l=src(cv::Rect(win_xleft_low,win_y_low,2*margin,window_height));
        r=src(cv::Rect(win_xright_low,win_y_low,2*margin,window_height));
        //NOTE 把像素值不为零的像素坐标存入矩阵
        cv::findNonZero(l,nonzerol);
        cv::findNonZero(r,nonzeror);
        //计算每个选框的leftx_current和rightx_current中心点
        if(nonzerol.size()>minpix){
            int leftx=0;
            for(auto& n:nonzerol){
                leftx+=n.x;
            }
            leftx_current=win_xleft_low+leftx/nonzerol.size();
        }
        if(nonzeror.size()>minpix){
            int rightx=0;
            for(auto& n:nonzeror){
                rightx+=n.x;
            }
            rightx_current=win_xright_low+rightx/nonzeror.size();
        }
        //将中心点坐标存入容器
        lpoint.push_back(cv::Point(leftx_current,win_y_low));
        rpoint.push_back(cv::Point(rightx_current,win_y_low));
    }
    //拟合左右车道线坐标
    cv::Mat leftx = polyfit(lpoint,2);
    cv::Mat rightx = polyfit(rpoint,2);
    //计算拟合曲线坐标
    lp=polyval(leftx,lpoint,2);
    rp=polyval(rightx,rpoint,2);
    //计算车道偏离距离
    int lane_width=abs(rpoint.front().x-lpoint.front().x);
    double lane_xm_per_pix=3.7/lane_width;
    double veh_pos=(((rpoint.front().x+lpoint.front().x)*lane_xm_per_pix)/2);
    double cen_pos=((src.cols*lane_xm_per_pix)/2);
    distance_from_center=veh_pos-cen_pos;
    // cout<<"dis"<<distance_from_center<<endl;
    // cout<<lp<<endl;
}
/*
@param src 为原始图像
@param lp 输入为左车道线坐标
@param rp 输入为右车道线坐标
@param Minv 输入为反变形矩阵
@param distance_from_center 输入为车道偏离距离
@NOTE 该函数用于绘制车道线和可行使区域
*/
void draw_area(const cv::Mat& src,vector<cv::Point>& lp,vector<cv::Point>& rp,const cv::Mat& Minv,double& distance_from_center){
    vector<cv::Point> rflip,ptr;
    cv::Mat colormask=cv::Mat::zeros(src.rows,src.cols,CV_8UC3);
    cv::Mat dst,midst;
    //绘制车道线
    cv::polylines(colormask,lp,false,cv::Scalar(0,255,0),5);
    cv::polylines(colormask,rp,false,cv::Scalar(0,0,255),5);
    //反转坐标，以便绘制填充区域
    cv::flip(rp,rflip,1);
    //拼接坐标
    cv::hconcat(lp,rflip,ptr);
    //绘制填充区域
    const cv::Point* em[1]={&ptr[0]};
    int nop=(int)ptr.size();
    cv::fillPoly(colormask,em,&nop,1,cv::Scalar(200,200,0));
    //反变形
    cv::warpPerspective(colormask,midst,Minv,src.size(),cv::INTER_LINEAR);
    //将车道线图片和原始图片叠加
    cv::addWeighted(src,1,midst,0.3,0,dst);
    //绘制文字
    cv::putText(dst,"distance bias:"+to_string(distance_from_center)+"m",cv::Point(50,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255),2);
    cv::imshow("video",dst);
    // cv::waitKey(10000);
}

int main()
{
    vector<string> imgs;//储存照片路径的容
    cv::Size grid=cv::Size(9,6);//棋盘行列数
    cv::Size distance=cv::Size(1,1);//单位棋盘格物理尺寸
    
    string cal_dir="D:\\Code\\C++\\Advanced-Lane-Lines\\CarND-Advanced-Lane-Lines\\camera_cal";//棋盘格图片目录路径
    string test_dir="D:\\Code\\C++\\Advanced-Lane-Lines\\CarND-Advanced-Lane-Lines\\test_images";
    string video_dir="D:\\Code\\C++\\Advanced-Lane-Lines\\CarND-Advanced-Lane-Lines\\project_video.mp4";
    string filetype=".jpg";//图片格式
    cv::Mat img,cimg,imge,imgout,absm,mag,hls,dir,luv,lab;//输出图像
    int rightx_current,leftx_current,frameNum;//车道线基坐标点x轴
    vector<cv::Point> lp,rp;//车道线坐标点
    double distance_from_center;
    cv::Mat M,Minv;
    cv::Mat cameraMatirx;//内参矩阵，需初始化
    cv::Mat distCoeffs;//畸变矩阵，需初始化
    // int OFFSET=250;
    // vector<cv::Point2f> src={cv::Point2f(132,703),
    //                         cv::Point2f(540,466),
    //                         cv::Point2f(740,466),
    //                         cv::Point2f(1147,703)};
    // vector<cv::Point2f> dst={cv::Point2f(src[0].x+OFFSET,720),
    //                         cv::Point2f(src[0].x+OFFSET,0),
    //                         cv::Point2f(src[3].x-OFFSET,0),
    //                         cv::Point2f(src[3].x-OFFSET,720)};
    //变形基础点
    vector<cv::Point2f> src={cv::Point2f(203,720),
                            cv::Point2f(585,460),
                            cv::Point2f(695,460),
                            cv::Point2f(1127,720)};
    vector<cv::Point2f> dst={cv::Point2f(320,720),
                            cv::Point2f(320,0),
                            cv::Point2f(960,0),
                            cv::Point2f(960,720)};
    //获取棋盘格图片
    get_images_by_dir(cal_dir,filetype,imgs);
    //计算矫正系数
    get_obj_img_points(imgs,grid,distance,cameraMatirx,distCoeffs);
    //获取其测试图片
    imgs.clear();
    get_images_by_dir(test_dir,filetype,imgs);
    get_M_Minv(src,dst,M,Minv);
    // for(auto& image:imgs){
    //     img=cv::imread(image);
    //     cal_undistort(img,cimg,object_points,img_points);//矫正图像
    //     cv::warpPerspective(cimg,imge,M,imge.size(),cv::INTER_LINEAR);
    //     abs_sobel_thresh(imge,absm,'x',55,200);//sobel边缘识别
    //     mag_thresh(imge,mag,3,45,150);
    //     hls_select(imge,hls,'s',160,255);
    //     dir_threshold(imge,dir,3,0.7,1.3);
    //     luv_select(imge,luv,'l',180,255);
    //     //lab_select(imge,lab,'b',126,127);
        
    //     imgout=(absm&mag&luv)|(hls&luv);
    //     find_line(imgout,lp,rp,rightx_current,leftx_current,distance_from_center); 
    //     draw_area(cimg,lp,rp,Minv,distance_from_center); 
    //     // cv::imshow("result",imgout);
    //     cv::waitKey(1000);
    // }
    cv::VideoCapture iv(video_dir);
    if(!iv.isOpened()){
        cout<<"Could not open reference "<<video_dir<<endl;
        return -1;
    }
    cv::namedWindow("video",cv::WINDOW_AUTOSIZE);

    while(true){
        iv>>img;
        if(img.empty()){
            cout<<"<<< Game over! >>>";
            break;
        }
        frameNum++;
        cout<<"Frame: "<<frameNum<<"# "<<endl;
        cv::undistort(img,cimg,cameraMatirx,distCoeffs);//矫正图像
        cv::warpPerspective(cimg,imge,M,img.size(),cv::INTER_LINEAR);
        abs_sobel_thresh(imge,absm,'x',55,200);//sobel边缘识别
        mag_thresh(imge,mag,3,45,150);
        hls_select(imge,hls,'s',160,255);
        dir_threshold(imge,dir,3,0.7,1.3);
        luv_select(imge,luv,'l',180,255);
        // lab_select(imge,lab,'b',126,127);
        
        imgout=(absm&mag&luv)|(hls&luv);
        
        find_line(imgout,lp,rp,rightx_current,leftx_current,distance_from_center); 
        draw_area(cimg,lp,rp,Minv,distance_from_center); 
        // cv::imshow("video",cimg);
        cv::waitKey(5);
    }
    return 0;
}
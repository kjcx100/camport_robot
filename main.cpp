#include "../common/common.hpp"
using namespace std;
using namespace cv;

typedef unsigned short U16;
static char buffer[1024 * 1024 * 20];

static char tmpbuffer[1024 * 1024 * 20];
static int  n;
static volatile bool exit_main;
static volatile bool save_frame;
static volatile bool save_rect_color;
static int gsave_rect_count = 0;

struct CallbackData {
	int             index;
	TY_DEV_HANDLE   hDevice;
	DepthRender*    render;

	TY_CAMERA_DISTORTION color_dist;
	TY_CAMERA_INTRINSIC color_intri;
};
bool verifySizes(Rect mr) {
	// Set a min and max area. All other patchs are discarded
	// int min= 1*aspect*1; // minimum area
	// int max= 2000*aspect*2000; // maximum area
	int min = 400 * 2;  // minimum area
	int max = 568 * 340;  // maximum area

	float area = mr.height * mr.width;
	// cout << "area:" << area << endl;
	//|| mr.y > (mr.height - (mr.height >> 3))
	if (area < min || area > max || (mr.y > 413))	//坐标在下边8分之1
		return false;
	else
	{
		//cout << "mr.y:" << mr.y << "(472*7/8):" << (472*7/8) << endl;
		return true;
	}
}
//  calc safe Rect
//  if not exit, return false

bool calcSafeRect(const RotatedRect &roi_rect, const Mat &src,
	Rect_<float> &safeBoundRect) {
	Rect_<float> boudRect = roi_rect.boundingRect();

	float tl_x = boudRect.x > 0 ? boudRect.x : 0;
	float tl_y = boudRect.y > 0 ? boudRect.y : 0;

	float br_x = boudRect.x + boudRect.width < src.cols
		? boudRect.x + boudRect.width - 1
		: src.cols - 1;
	float br_y = boudRect.y + boudRect.height < src.rows
		? boudRect.y + boudRect.height - 1
		: src.rows - 1;

	float roi_width = br_x - tl_x;
	float roi_height = br_y - tl_y;

	if (roi_width <= 0 || roi_height <= 0) return false;

	//  a new rect not out the range of mat

	safeBoundRect = Rect_<float>(tl_x, tl_y, roi_width, roi_height);

	return true;
}

void depthTransfer(cv::Mat depth, uint16_t* t_data, cv::Mat* newDepth, cv::Mat* blackDepth)
{
	int i=0;
	uint16_t dst_data[480*640];
	uint16_t blk_data[480*640];
	uint16_t* src_data;
	uint16_t treshhold;
	
	src_data = (uint16_t *)depth.data;
	memset(blk_data,0,480*640*2);
	for(i=0;i<(480*640);i++)
	{
		treshhold = (uint16_t)(t_data[i]*0.03);
		if(src_data[i] == 0)
		{
			dst_data[i] = 0;
			if(i > (100*640))
			{
				blk_data[i] = 500;
			}
		}
		else if((src_data[i] - t_data[i]) < (-1 * treshhold))
		{
			dst_data[i] = 0;
			//dst_data[i] = 1200;
		}
		else if((src_data[i] - t_data[i]) > (treshhold * 1.5))
		{
			dst_data[i] = 2500;
		}
		else
		{
			dst_data[i] = 0;
		}
		/*
		if(i == (240*640 + 320))
		{
			LOGD("--------------------- treshhold:%d, src_data[i]-t_data[i]:%d", treshhold, (src_data[i] - t_data[i]));
			LOGD("---------------------------------------------- dst_data[]:%d", dst_data[i]);
		}
		*/
		//dst_data[i] = src_data[i] - t_data[i];
	}
	
	dst_data[0] = 500;
	dst_data[1] = 4000;
	
	blk_data[0] = 500;
	blk_data[1] = 4000;
	
	*newDepth = cv::Mat(480, 640, CV_16U, dst_data);
	*blackDepth = cv::Mat(480, 640, CV_16U, blk_data);
}

int DeepImgFinds_write_rgb(Mat depthColor, Mat resized_color, int blurSize, int morphW, int morphH)
{
	int SOBEL_SCALE = 0;
	int SOBEL_DELTA = 0.5;
	int SOBEL_DDEPTH = CV_16S;
	int SOBEL_X_WEIGHT = 1;

	Mat mat_blur;
	Mat In_rgb = resized_color.clone();
	mat_blur = depthColor.clone();
	GaussianBlur(depthColor, mat_blur, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);

	Mat mat_gray;
	if (mat_blur.channels() == 3)
		cvtColor(mat_blur, mat_gray, CV_RGB2GRAY);
	else
		mat_gray = mat_blur;
	//输入depth图像先切掉两块 上边，左边
	#if 0
	Mat FillImg_top = Mat::zeros(depthColor.rows,40, CV_8UC1);
	Mat FillImg_left = Mat::zeros(112,depthColor.cols - 40, CV_8UC1);
	//Rect blur_fillRect_top = Rect(0,0,depthColor.rows,40);	
	//Rect blur_fillRect_left = Rect(0,40,112,depthColor.cols - 40);
	Rect safe_fillRect_top = Rect(0,0,depthColor.rows-2,40);
	Rect safe_fillRect_left = Rect(0,42,112,depthColor.cols -44);
	cout << "mat_gray.rows:" << mat_gray.rows <<" mat_gray.cols:" << mat_gray.cols << endl;
	cout << "safe_fillRect_top.x:" << safe_fillRect_top.x << " .y:"<< safe_fillRect_top.y << " .w:"<< safe_fillRect_top.width << " .h:"<< safe_fillRect_top.height << endl;
	cout << "safe_fillRect_left.x:" << safe_fillRect_left.x << " .y:"<< safe_fillRect_left.y << " .w:"<< safe_fillRect_left.width << " .h:"<< safe_fillRect_left.height << endl;
	Mat imageROI_top = mat_gray(safe_fillRect_top);
	Mat imageROI_left = mat_gray(safe_fillRect_left);
	//floodFill(imageROI_top, Point2f(imageROI_top.cols >> 1, imageROI_top.rows >> 1), 0);
	cout << "FillImg_top.rows: " << FillImg_top.rows << " FillImg_top.cols:" << FillImg_top.cols << endl;
	cout << "imageROI_top.rows: " << imageROI_top.rows << " imageROI_top.cols:" << imageROI_top.cols << endl;
	FillImg_top.copyTo(imageROI_top);
	FillImg_left.copyTo(imageROI_left);
	for (int i = 0; i <  mat_gray.rows; i++)
	{
		for (int j = 0; j <  mat_gray.cols; j++)
		{
			//绘制出contours向量内所有的像素点
			Point P = Point(j, i);
			//输出到rgb
			if(i <= 40)
				circle(mat_gray, P, 0, Scalar(255, 255, 0));
			if( i > 40 && j <= 112)
				circle(mat_gray, P, 0, Scalar(255, 255, 0));
		}
	}
	#endif
	for (int i = 0; i <  mat_gray.rows; i++)
	{
		uchar* data_gray = mat_gray.ptr<uchar>(i);
		for (int j = 0; j <  mat_gray.cols; j++)
		{
			//输出到rgb
			if(i <= 40)
				data_gray[j] = 255;
			if( i > 40 && j <= 112)
				data_gray[j] = 255;
		}
	}
	int scale = SOBEL_SCALE;
	int delta = SOBEL_DELTA;
	int ddepth = SOBEL_DDEPTH;

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	//namedWindow("mat_gray");
	//imshow("mat_gray", mat_gray);
	//cvWaitKey(0);
	//destroyWindow("mat_gray");
	//Sobel(mat_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);


	//Mat grad;
	//addWeighted(abs_grad_x, SOBEL_X_WEIGHT, 0, 0, 0, grad);

	Mat mat_threshold;
	double otsu_thresh_val = threshold(mat_gray, mat_threshold, 30, 255, CV_THRESH_BINARY_INV);
	//threshold(grad, mat_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	//############先开操作，去掉一些小的区域####################
	int Open_morphW = 3;
	int Open_morphH = 3;
	Mat element = getStructuringElement(MORPH_RECT, Size(Open_morphW, Open_morphH));
	morphologyEx(mat_threshold, mat_threshold, MORPH_OPEN, element);
	char jpgfileopen[1024] = { 0 };
	char morphopen[1024] = { 0 };
	//strncpy_s(morphopen, filename + st_len_dirout, strlen(filename) - 4 - st_len_dirout);
	//sprintf_s(jpgfileopen, "./outdir%s_morphology.jpg", morphopen);
	//imwrite(jpgfileopen, mat_threshold);
	if (save_frame){
		LOGD(">>>>>>>>>> write jpgfileopen");
		imwrite("jpgfile_open.png", mat_threshold);
	}
	//#if 1
	Mat findContour;
	mat_threshold.copyTo(findContour);
	vector<vector<Point> > contours;
	findContours(findContour,
		contours,               // a vector of contours
		CV_RETR_LIST,
		CV_CHAIN_APPROX_NONE);  // all pixels of each contours
	vector<vector<Point> >::iterator itc = contours.begin();
	int Count_contours = 0;
	vector<Rect> first_rects;
 
	while (itc != contours.end()) {
		RotatedRect mr = minAreaRect(Mat(*itc));
		Rect_<float> safeBoundRect;
		if (!calcSafeRect(mr, mat_threshold, safeBoundRect))
		{
			cout << "calcSafeRect is false" << endl;
			//continue;
		}
		else
		{
			if (verifySizes(safeBoundRect)) {
				first_rects.push_back(safeBoundRect);
				//cout << "first_rects.size==" << first_rects.size() << endl;
				cout << "safeBoundRect.area:" << safeBoundRect.area() << endl;
				cout << "safeBoundRect.x:"<< safeBoundRect.x<<"  y:"<< safeBoundRect.y<<"  width:"<< safeBoundRect.width <<"  height:"  
					 << safeBoundRect.height << endl;
				/*
				Mat image_rects = mat_threshold(safeBoundRect);
				namedWindow("image_rects");
				imshow("image_rects", image_rects);
				cvWaitKey(0);
				destroyWindow("image_rects");
				char jpgfile_rects[1024] = { 0 };
				sprintf_s(jpgfile_rects, "%s_image_rects_%d.jpg", filename, first_rects.size());
				//imwrite(jpgfile_rects, image_rects);
				*/
				rectangle(depthColor, safeBoundRect, Scalar(0, 0, 255));
				//rectangle(In_rgb, safeBoundRect, Scalar(0, 255, 255));
				double safecontourArea = contourArea(Mat(*itc));
				//取出检测到的区域 and canny
				Mat imageIn_rects = In_rgb(safeBoundRect);
				Mat imageIn_gray,imageIn_edge;
				cvtColor(imageIn_rects, imageIn_gray, CV_RGB2GRAY);
				cv::blur(imageIn_gray, imageIn_edge, cv::Size(7, 7));
				Canny(imageIn_edge, imageIn_edge, 10, 60, 3);
				char szimageIn_edge[1024] = {0};
				sprintf(szimageIn_edge, "imageIn_edge%d.jpg", Count_contours);
				if (save_frame) {
					LOGD(">>>>>>>>>> write imageIn_edge");
					imwrite(szimageIn_edge, imageIn_edge);
				}
				Mat rectsContour;
				imageIn_edge.copyTo(rectsContour);
				vector<vector<Point> > rectscontours;
				findContours(rectsContour,
					rectscontours,               // a vector of contours
					CV_RETR_EXTERNAL,
					CV_CHAIN_APPROX_NONE);  // all pixels of each contours
				//imageIn_edge.copyTo(imageIn_rects);
				std::cout <<"rectscontours.size:" << rectscontours.size()<< endl;
				for (int i = 0; i < rectscontours.size(); i++)
				{
					double rectscontourArea = contourArea(Mat(rectscontours[i]));
					std::cout <<"rectscontourArea:" << rectscontourArea <<"  safecontourArea:" << safecontourArea<< endl;
					if(abs(rectscontourArea - safecontourArea) < safecontourArea*0.05)
					{
						for (int j = 0; j < rectscontours[i].size(); j++)
						{
							//绘制出contours向量内所有的像素点
							Point P = Point(rectscontours[i][j].x, rectscontours[i][j].y);
							//输出到rgb
							//In_rgb.at<uchar>(P) = 0;
							circle(imageIn_rects, P, 0, Scalar(255, 255, 0));
						}
					}
				}
				//namedWindow("imageIn_rects");
				//imshow("imageIn_rects", imageIn_rects);
				//cvWaitKey(0);
				//destroyWindow("imageIn_rects");
				char szimageIn_rects[1024] = {0};
				sprintf(szimageIn_rects, "imageIn_rects%d.jpg", Count_contours);
				if (save_frame) {
					LOGD(">>>>>>>>>> write imageIn_rects");
					imwrite(szimageIn_rects, imageIn_rects);
				}
				for (int j = 0; j < contours[Count_contours].size(); j++)
				{
					//绘制出contours向量内所有的像素点
					Point P = Point(contours[Count_contours][j].x, contours[Count_contours][j].y);
					//输出到rgb
					//In_rgb.at<uchar>(P) = 0;
					circle(In_rgb, P, 0, Scalar(255, 255, 0));
				}
			}
			else//不满足条件的，填充黑色
			{
				rectangle(depthColor, safeBoundRect, Scalar(0, 255, 0));
				Mat FillImg = Mat::zeros(safeBoundRect.height, safeBoundRect.width, CV_8UC1);
				cout << "FillImg.cols:" << FillImg.cols << "  FillImg.rows:" << FillImg.rows << endl;
				Rect fillRect = safeBoundRect;
				Mat imageROI = mat_threshold(fillRect);
				floodFill(imageROI, Point2f(imageROI.cols >> 1, imageROI.rows >> 1), 0);
				FillImg.copyTo(imageROI);
				//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
				for (int j = 0; j < contours[Count_contours].size(); j++)
				{
					//绘制出contours向量内所有的像素点
					Point P = Point(contours[Count_contours][j].x, contours[Count_contours][j].y);
					mat_threshold.at<uchar>(P) = 0;
				}
				{
					//Mat FillImg = Mat::zeros(safeBoundRect.height, safeBoundRect.width, CV_8UC1);
					//cout << "FillImg.cols:" << FillImg.cols << "  FillImg.rows:" << FillImg.rows << endl;
					////cout << "safeBoundRect.angle:" << safeBoundRect.angle << endl;
					//Rect fillRect = safeBoundRect;
					//Mat imageROI = mat_threshold(fillRect);
					//FillImg.copyTo(imageROI);
				}
			}
		}
		++itc;
		++Count_contours;
	}
	//out = mat_threshold;
	//namedWindow("in_add_rect");
	imshow("in_add_rect", depthColor);
	//cvWaitKey(0);
	//destroyWindow("in_add_rect");
	char jpgin_add_rect[1024] = { 0 };
	char name[1024] = { 0 };
	//strncpy_s(name, filename + st_len_dirout, strlen(filename) - 4 - st_len_dirout);
	//sprintf_s(jpgin_add_rect, "./outdir%s_in_add_rect.jpg", name);
	//imwrite(jpgin_add_rect, in);
	if (save_frame) {
		LOGD(">>>>>>>>>> write jpgin_add_rect");
		imwrite("jpgin_add_rect.png", depthColor);
	}
	char szsave_jpgin_add[1024] = {0};
	sprintf(szsave_jpgin_add, "jpgin_add_rect%d.jpg", gsave_rect_count);
	if (save_rect_color) {
		LOGD(">>>>>>>>>> write jpgin_add_rect");
		imwrite(szsave_jpgin_add, depthColor);
	}
	//#endif
	////////////////////找连通区域在膨胀腐蚀之前//////////////////////
	element = getStructuringElement(MORPH_RECT, Size(morphW, morphH));
	morphologyEx(mat_threshold, mat_threshold, MORPH_CLOSE, element);

	//namedWindow("morphologyEx");
	imshow("morphologyClose", mat_threshold);
	if (save_frame) {
		LOGD(">>>>>>>>>> write morphologyClose");
		imwrite("morphologyClose.png", mat_threshold);
	}
	//cvWaitKey(0);
	//destroyWindow("morphologyEx");
	//char jpgfile[1024] = { 0 };
	//char morphname[1024] = { 0 };
	//strncpy_s(morphname, filename + st_len_dirout, strlen(filename) - 4 - st_len_dirout);
	//sprintf_s(jpgfile, "./outdir%s_morphology.jpg", morphname);
	//imwrite(jpgfile, mat_threshold);
	//############写到rgb文件中##############
	char rgbjpgfile[1024] = { 0 };
	char write_rgbname[1024] = { 0 };
	//strncpy_s(write_rgbname, filename + st_len_dirout, strlen(filename) - 4 - st_len_dirout);
	//sprintf_s(rgbjpgfile, "./outdir%s__rgb_rect.jpg", write_rgbname);
	//imwrite(rgbjpgfile, In_rgb);
	cv::imshow("rect_resized_color", In_rgb);
	if (save_frame) {
		LOGD(">>>>>>>>>> write resized_color add rect");
		imwrite("rect_resized_color.png", In_rgb);
	}
	char szsave_rect_color[1024] = {0};
	sprintf(szsave_rect_color, "rect_resized_color%d.jpg", gsave_rect_count);
	if (save_rect_color) {
		LOGD(">>>>>>>>>> write resized_color add rect");
		imwrite(szsave_rect_color, In_rgb);
	}
	return 0;
}

cv::Mat cvMatRect2Tetra(cv::Mat mtxSrc, int iDstX1, int iDstY1, int iDstX2, int iDstY2,
						int iDstX3, int iDstY3, int iDstX4, int iDstY4, int iDstWidth, int iDstHeight)
{
	cv::Mat mtxDst;
	std::vector<cv::Point2f> src_corners(4);
	std::vector<cv::Point2f> dst_corners(4);
 
	src_corners[0]= cv::Point2f(0,0);
	src_corners[1]= cv::Point2f(mtxSrc.cols - 1,0);
	src_corners[2]= cv::Point2f(0, mtxSrc.rows - 1);
	src_corners[3]= cv::Point2f(mtxSrc.cols - 1, mtxSrc.rows - 1);
 
	dst_corners[0] = cv::Point2f(iDstX1, iDstY1);
	dst_corners[1] = cv::Point2f(iDstX2, iDstY2);
	dst_corners[2] = cv::Point2f(iDstX3, iDstY3);
	dst_corners[3] = cv::Point2f(iDstX4, iDstY4);
 
	cv::Mat transMtx = cv::getPerspectiveTransform(src_corners, dst_corners);
	cv::warpPerspective(mtxSrc, mtxDst, transMtx, cv::Size(iDstWidth, iDstHeight));
 
	return mtxDst;
}

void handleFrame(TY_FRAME_DATA* frame, void* userdata ,void* tempdata)
{
	CallbackData* pData = (CallbackData*)userdata;
	LOGD("=== Get frame %d", ++pData->index);

	cv::Mat depth, irl, irr, color, point3D;
	parseFrame(*frame, &depth, &irl, &irr, &color, &point3D);
	cv::Mat newDepth,TransDepth,blackDepth;
	cv::Mat resized_color;
	cv::Mat temp;
	
	#if 1
	if (!depth.empty()){
				/////lxl add///////
		cv::Mat tempDepth = cv::Mat(depth.rows, depth.cols, CV_16U, (uint16_t *)tempdata);
		//for(int i =0 ; i < depth.rows*depth.cols ; i ++)
		//depth = depth - tempDepth ;	//深度图减去模板
        cv::Mat Depth2 = cvMatRect2Tetra(depth, 18, 8, depth.cols - 1 + 7, 8, 46, depth.rows -1 - 7, depth.cols - 1 + 7, depth.rows - 1 - 7, depth.cols, depth.rows);
		depthTransfer(Depth2, (uint16_t*)tempdata, &TransDepth, &blackDepth);
		cv::resize(color, resized_color, depth.size());
		cv::imshow("resizedColor", resized_color);
		if (save_frame){
			LOGD(">>>>>>>>>> write resized_color");
			imwrite("resized_color.png", resized_color);
			//save_frame = false;
		}
		//lxl add output grayimg
		pData->render->SetColorType(DepthRender::COLORTYPE_GRAY);
		cv::Mat depthColor = pData->render->Compute(TransDepth);
		if (save_frame){
			LOGD(">>>>>>>>>> write depthColor");
			imwrite("TransdepthColor.png", depthColor);
			//save_frame = false;
		}
		depthColor = depthColor / 2 + resized_color / 2;
		cv::imshow("projected depth", depthColor);
		std::cout << "depthColor.channels:" << depthColor.channels() << "  rows:" << depthColor.rows << "  cols:" << depthColor.cols << std::endl;
		if (save_frame){
			LOGD(">>>>>>>>>> write projected_depth");
			imwrite("projected_depth.png", depthColor);
			//save_frame = false;
		}
		DeepImgFinds_write_rgb(depthColor,resized_color, 3, 7, 7);
	}
	// do Registration
	#else
	if (!point3D.empty() && !color.empty()) 
	{
		// ASSERT_OK( TYRegisterWorldToColor2(pData->hDevice, (TY_VECT_3F*)point3D.data, 0
		//             , point3D.cols * point3D.rows, color.cols, color.rows, (uint16_t*)buffer, sizeof(buffer)
		//             ));
		ASSERT_OK(TYRegisterWorldToColor(pData->hDevice, (TY_VECT_3F*)point3D.data, 0
			, point3D.cols * point3D.rows, (uint16_t*)buffer, sizeof(buffer)
			));
		newDepth = cv::Mat(color.rows, color.cols, CV_16U, (uint16_t*)buffer);
		//you may want to use median filter to fill holes in projected depth image or do something else here
		cv::medianBlur(newDepth, temp, 5);
		newDepth = temp;
		//resize to the same size for display
		cv::resize(newDepth, newDepth, depth.size(), 0, 0, 0);
		/////lxl add///////
		//cv::Mat tempDepth = cv::Mat(depth.rows, depth.cols, CV_16U, tempdata);
		//for(int i =0 ; i < depth.rows*depth.cols ; i ++)
		//newDepth = newDepth - tempDepth ;	//深度图减去模板

		depthTransfer(newDepth, (uint16_t*)tempdata, &TransDepth, &blackDepth);
		cv::resize(color, resized_color, depth.size());
		cv::imshow("resizedColor", resized_color);
		if (save_frame){
			LOGD(">>>>>>>>>> write resized_color");
			imwrite("resized_color.png", resized_color);
			//save_frame = false;
		}
		//lxl add output grayimg
		pData->render->SetColorType(DepthRender::COLORTYPE_GRAY);
		cv::Mat depthColor = pData->render->Compute(TransDepth);
		if (save_frame){
			LOGD(">>>>>>>>>> write depthColor");
			imwrite("TransdepthColor.png", depthColor);
			//save_frame = false;
		}
		depthColor = depthColor / 2 + resized_color / 2;
		cv::imshow("projected depth", depthColor);
		std::cout << "depthColor.channels:" << depthColor.channels() << "  rows:" << depthColor.rows << "  cols:" << depthColor.cols << std::endl;
		if (save_frame){
			LOGD(">>>>>>>>>> write projected_depth");
			imwrite("projected_depth.png", depthColor);
			//save_frame = false;
		}
		DeepImgFinds_write_rgb(depthColor,resized_color, 3, 7, 7);
	}
	#endif
	if (save_frame){
		LOGD(">>>>>>>>>> write images");
		//imwrite("depth.png", newDepth);
		imwrite("color.png", color);
		save_frame = false;
	}
	if (save_rect_color) {
		save_rect_color = false;
		gsave_rect_count++;
	}
	//cv::namedWindow("key");
	int key = cv::waitKey(1);
	//LOGD(">>>>>>>>>>key==%d\n", key);
	switch (key){
	case -1:
		break;
	case 'q': case 1048576 + 'q':
		exit_main = true;
		break;
	case 's': case 1048576 + 's':
		save_frame = true;
		break;
	case 'a': case 1048576 + 'a':
		save_rect_color = true;
		break; 
	default:
		LOGD("Pressed key %d", key);
	}

	//LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
	ASSERT_OK(TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize));
}

void eventCallback(TY_EVENT_INFO *event_info, void *userdata)
{
	if (event_info->eventId == TY_EVENT_DEVICE_OFFLINE) {
		LOGD("=== Event Callback: Device Offline!");
		// Note: 
		//     Please set TY_BOOL_KEEP_ALIVE_ONOFF feature to false if you need to debug with breakpoint!
	}
	else if (event_info->eventId == TY_EVENT_LICENSE_ERROR) {
		LOGD("=== Event Callback: License Error!");
	}
}


int main(int argc, char* argv[])
{
	const char* IP = NULL;
	const char* ID = NULL;
	TY_DEV_HANDLE hDevice;
	int m_width = 640;
	int m_hight = 480;
	FILE *filetmp;
	const char* tempimg = "./template.yuv";
	filetmp = fopen( tempimg, "rb");
	if (NULL == filetmp)
	{
		printf("Error:Open tempimg file fail!\n");
		return -1;
	}
	printf("fopen %s ok\n",tempimg);
	//U16* pfilebuftmp = new U16[m_width*m_hight];//相机采图分辨率

	if (m_width*m_hight * 2 != fread(tmpbuffer, 1, m_width*m_hight*2, filetmp))
	{
		//提示文件读取错误  
		fclose(filetmp);
		cout << "fread_s filetmp ERR!!!" << endl;
		return -1;
	}
	fclose(filetmp);	
	for (int i = 1; i < argc; i++){
		if (strcmp(argv[i], "-id") == 0){
			ID = argv[++i];
		}
		else if (strcmp(argv[i], "-ip") == 0){
			IP = argv[++i];
		}
		else if (strcmp(argv[i], "-h") == 0){
			LOGI("Usage: SimpleView_Callback [-h] [-ip <IP>]");
			return 0;
		}
	}

	LOGD("=== Init lib");
	ASSERT_OK(TYInitLib());
	TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
	ASSERT_OK(TYLibVersion(pVer));
	LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

	if (IP) {
		LOGD("=== Open device %s", IP);
		ASSERT_OK(TYOpenDeviceWithIP(IP, &hDevice));
	}
	else {
		if (ID == NULL){
			LOGD("=== Get device info");
			ASSERT_OK(TYGetDeviceNumber(&n));
			LOGD("     - device number %d", n);

			TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
			ASSERT_OK(TYGetDeviceList(pBaseInfo, 100, &n));

			if (n == 0){
				LOGD("=== No device got");
				return -1;
			}
			ID = pBaseInfo[0].id;
		}

		LOGD("=== Open device: %s", ID);
		ASSERT_OK(TYOpenDevice(ID, &hDevice));
	}

	int32_t allComps;
	ASSERT_OK(TYGetComponentIDs(hDevice, &allComps));
	if (!(allComps & TY_COMPONENT_RGB_CAM)){
		LOGE("=== Has no RGB camera, cant do registration");
		return -1;
	}

	LOGD("=== Configure components");
	int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;
	ASSERT_OK(TYEnableComponents(hDevice, componentIDs));

	LOGD("=== Prepare image buffer");
	int32_t frameSize;

	//frameSize = 1280 * 960 * (3 + 2 + 2);
	ASSERT_OK(TYGetFrameBufferSize(hDevice, &frameSize));
	LOGD("     - Get size of framebuffer, %d", frameSize);
	LOGD("     - Allocate & enqueue buffers");
	char* frameBuffer[2];
	frameBuffer[0] = new char[frameSize];
	frameBuffer[1] = new char[frameSize];
	LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
	ASSERT_OK(TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize));
	LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
	ASSERT_OK(TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize));

	LOGD("=== Register callback");
	LOGD("Note: Callback may block internal data receiving,");
	LOGD("      so that user should not do long time work in callback.");
	LOGD("      To avoid copying data, we pop the framebuffer from buffer queue and");
	LOGD("      give it back to user, user should call TYEnqueueBuffer to re-enqueue it.");
	DepthRender render;
	CallbackData cb_data;
	cb_data.index = 0;
	cb_data.hDevice = hDevice;
	cb_data.render = &render;
	// ASSERT_OK( TYRegisterCallback(hDevice, frameCallback, &cb_data) );

	LOGD("=== Register event callback");
	LOGD("Note: Callback may block internal data receiving,");
	LOGD("      so that user should not do long time work in callback.");
	ASSERT_OK(TYRegisterEventCallback(hDevice, eventCallback, NULL));

	LOGD("=== Disable trigger mode");
	ASSERT_OK(TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false));

	LOGD("=== Start capture");
	ASSERT_OK(TYStartCapture(hDevice));

	LOGD("=== Read color rectify matrix");
	{
		TY_CAMERA_DISTORTION color_dist;
		TY_CAMERA_INTRINSIC color_intri;
		TY_STATUS ret = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_DISTORTION, &color_dist, sizeof(color_dist));
		ret |= TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, TY_STRUCT_CAM_INTRINSIC, &color_intri, sizeof(color_intri));
		if (ret == TY_STATUS_OK)
		{
			cb_data.color_intri = color_intri;
			cb_data.color_dist = color_dist;
		}
		else
		{ //reading data from device failed .set some default values....
			memset(cb_data.color_dist.data, 0, 12 * sizeof(float));
			memset(cb_data.color_intri.data, 0, 9 * sizeof(float));
			cb_data.color_intri.data[0] = 1000.f;
			cb_data.color_intri.data[4] = 1000.f;
			cb_data.color_intri.data[2] = 600.f;
			cb_data.color_intri.data[5] = 450.f;
		}
	}

	LOGD("=== Wait for callback");
	exit_main = false;
	while (!exit_main){
		TY_FRAME_DATA frame;
		int err = TYFetchFrame(hDevice, &frame, -1);
		if (err != TY_STATUS_OK) {
			LOGE("Fetch frame error %d: %s", err, TYErrorString(err));
			break;
		}
		else {
			handleFrame(&frame, &cb_data , (void*)tmpbuffer);
		}
	}

	ASSERT_OK(TYStopCapture(hDevice));
	ASSERT_OK(TYCloseDevice(hDevice));
	ASSERT_OK(TYDeinitLib());
	delete frameBuffer[0];
	delete frameBuffer[1];

	LOGD("=== Main done!");
	return 0;
}

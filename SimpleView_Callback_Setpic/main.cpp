#include "../common/common.hpp"
#include<stdio.h>
#include<string.h>
#include<algorithm>

#define TEMPLATE_DEBUG

static char buffer[1024*1024];
static bool fakeLock = false; // NOTE: fakeLock may lock failed

cv::Mat  grayImage, out_Canny;
int min_Thresh = 3;
int max_Thresh = 30;

struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    bool            saveFrame;
    int             saveIdx;
    cv::Mat         depth;
    cv::Mat         leftIR;
    cv::Mat         rightIR;
    cv::Mat         color;
    cv::Mat         point3D;
};

void frameCallback(TY_FRAME_DATA* frame, void* userdata)
{
    CallbackData* pData = (CallbackData*) userdata;
    LOGD("=== Get frame %d", ++pData->index);

    while(fakeLock){
        MSLEEP(10);
    }
    fakeLock = true;

    pData->depth.release();
    pData->leftIR.release();
    pData->rightIR.release();
    pData->color.release();
    pData->point3D.release();

    parseFrame(*frame, &pData->depth, &pData->leftIR, &pData->rightIR
            , &pData->color, &pData->point3D);

    fakeLock = false;

    if(!pData->color.empty()){
        LOGI("Color format is %s", colorFormatName(TYImageInFrame(*frame, TY_COMPONENT_RGB_CAM)->pixelFormat));
    }

    LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

void eventCallback(TY_EVENT_INFO *event_info, void *userdata)
{
    if (event_info->eventId == TY_EVENT_DEVICE_OFFLINE) {
        LOGD("=== Event Calllback: Device Offline!");
    }
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
			dst_data[i] = 1200;
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

void combineDepth(cv::Mat depth1, cv::Mat depth2, cv::Mat* dst_depth)
{
	int i=0;
	uint16_t* src_data1;
	uint16_t* src_data2;
	uint16_t dst_data[480*640];
	
	src_data1 = (uint16_t *)depth1.data;
	src_data2 = (uint16_t *)depth2.data;
	for(i=0;i<(480*640);i++)
	{
		dst_data[i] = src_data1[i] + src_data2[i];
	}
	
	*dst_depth = cv::Mat(480, 640, CV_16U, dst_data);
}

void Find_Draw_COntours(int, void*)
{
	Canny(grayImage, out_Canny, min_Thresh, max_Thresh *2, 3);
	imshow("window2", out_Canny);
}

void show_template(DepthViewer* depth_view,int counter)
{
	uint16_t data[450*565];
	uint16_t count[450*565];
	uint32_t temp[450*565];
	uint16_t avg_data[450*565];
	uint8_t u8_data[450*565];
	char num_arr[10];
	int i=0,j=0;
	FILE *fp;
	FILE *avg_fp;
	uint16_t  min_count = 0;
	char min_count_str[10];
	
	//DepthViewer depth_view;
	cv::Mat image;
	
	if(count <= 0)
	{
		return;
	}
	
	memset(count,0,450*565*2);
	memset(temp,0,450*565*4);
	for(i=0;i<counter;i++)
	{
		sprintf(num_arr, "depth_img/%d.img", i);
		
		fp = fopen(num_arr, "r+");
		fseek(fp, SEEK_SET, 0);
    	fread(data, 2, 450*565, fp);

    	fclose(fp);
    	for(j=0;j<450*565;j++)
    	{
    		if(data[j] > 0)
    		{
    			temp[j] = temp[j] + data[j];
    			count[j] = count[j]+1;
    		}
    	}
	}
	
	for(i=0;i<450;i++)
	{
		uint32_t valid_value = 0;
		uint16_t valid_count_avg = 0;
		uint16_t valid_count = 0;
		for(j=0;j<565;j++)
		{
			if((((i > 430) && (j >= 260) && (j <= 332)) || ((i >= 408) && (j >= 518)) || (i <= 50)) && (temp[i * 565 + j] == 0))
			{
				count[i * 565 + j] = counter;
			}
			
			if((temp[i * 565 + j] > 0) && (count[i * 565 + j] > 0))
			{
				valid_value = valid_value + temp[i * 565 + j];
				valid_count_avg = valid_count_avg + count[i * 565 + j];
				valid_count++;
			}
			
		}
		LOGD("------------------total valid_value and valid_count_avg and valid_count: %d.%d.%d", valid_value, valid_count_avg, valid_count);
		if(valid_count == 0)
		{
			if((i < 5) || (i>440))
			{
				valid_count = counter;
			}
			else
			{
				valid_count = 1;
			}
		}
		valid_value = valid_value/valid_count;
		valid_count_avg = valid_count_avg/valid_count;
		//valid_value = 1000;
		//valid_count_avg = 5;
		for(j=0;j<565;j++)
		{
			if(i <= 100)
			{
				if(temp[i * 565 + j] == 0)
				{
					temp[i * 565 + j] = valid_value;
					count[i * 565 + j] = valid_count_avg;
				}
			}
		}
	}
	
	for(i=0;i<450*565;i++)
    {
    	if(count[i] > 0)
    	{
    		avg_data[i] = (uint16_t)(temp[i]/count[i]);
    	}
    	else
    	{
    		avg_data[i] = 0;
    	}
    }
    
    min_count = *(std::min_element(count,count+450*565));
    sprintf(min_count_str, "%d", min_count);
    
    //for(i=0;i<(480*640);i++)
	//{
	//	u8_data[i] = avg_data[i]/9;
	//}
	//image = cv::Mat(480, 640, CV_8U, u8_data);
	
	//cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display Image", image);
    
    image = cv::Mat(450, 565, CV_16U, avg_data);
    
    avg_fp = fopen("template.img", "wb");
    fwrite(avg_data, 2, 450*565, avg_fp);
    fclose(avg_fp);
    
    cv::putText(image,min_count_str,cv::Point(2,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,0),2);
    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    depth_view->show("Display Image", image);
    //cv::imshow("Display Image", image);
    //cv::waitKey(10);
}

void show_template_origin(DepthViewer* depth_view,int counter)
{
	uint16_t data[480*640];
	uint16_t count[480*640];
	uint32_t temp[480*640];
	uint16_t avg_data[480*640];
	uint8_t u8_data[480*640];
	char num_arr[10];
	int i=0,j=0;
	FILE *fp;
	FILE *avg_fp;
	uint16_t  min_count = 0;
	char min_count_str[10];
	
	//DepthViewer depth_view;
	cv::Mat image;
	
	if(count <= 0)
	{
		return;
	}
	
	memset(count,0,480*640*2);
	memset(temp,0,480*640*4);
	for(i=0;i<counter;i++)
	{
		sprintf(num_arr, "depth_img1/%d.img", i);
		
		fp = fopen(num_arr, "r+");
		fseek(fp, SEEK_SET, 0);
    	fread(data, 2, 480*640, fp);

    	fclose(fp);
    	for(j=0;j<480*640;j++)
    	{
    		if(data[j] > 0)
    		{
    			temp[j] = temp[j] + data[j];
    			count[j] = count[j]+1;
    		}
    	}
	}
	
	for(i=0;i<480;i++)
	{
		uint32_t valid_value = 0;
		uint16_t valid_count_avg = 0;
		uint16_t valid_count = 0;
		for(j=0;j<640;j++)
		{
			if(j>=68)
			{
				if((((i > 440) && (j >= 330) && (j <= 402)) || ((i >= 478) && (j >= 588)) || (i <= 60)) && (temp[i * 640 + j] == 0))
				{
					count[i * 640 + j] = counter;
				}
			
				if((temp[i * 640 + j] > 0) && (count[i * 640 + j] > 0))
				{
					valid_value = valid_value + temp[i * 640 + j];
					valid_count_avg = valid_count_avg + count[i * 640 + j];
					valid_count++;
				}
			}
			
		}
		LOGD("------------------total valid_value and valid_count_avg and valid_count: %d.%d.%d", valid_value, valid_count_avg, valid_count);
		if(valid_count == 0)
		{
			if((i < 15) || (i>450))
			{
				valid_count = counter;
			}
			else
			{
				valid_count = 1;
			}
		}
		valid_value = valid_value/valid_count;
		valid_count_avg = valid_count_avg/valid_count;
		//valid_value = 1000;
		//valid_count_avg = 5;
		for(j=0;j<640;j++)
		{
			if(i <= 100)
			{
				if(j >= 68)
				{
					if(temp[i * 640 + j] == 0)
					{
						temp[i * 640 + j] = valid_value;
						count[i * 640 + j] = valid_count_avg;
					}
				}
				else
				{
					temp[i * 640 + j] = 0;
					count[i * 640 + j] = 1;
				}
			}
		}
	}
	
	for(i=0;i<480*640;i++)
    {
    	if(count[i] > 0)
    	{
    		avg_data[i] = (uint16_t)(temp[i]/count[i]);
    	}
    	else
    	{
    		avg_data[i] = 0;
    	}
    }
    
    min_count = *(std::min_element(count,count+480*640));
    sprintf(min_count_str, "%d", min_count);
    
    //for(i=0;i<(480*640);i++)
	//{
	//	u8_data[i] = avg_data[i]/9;
	//}
	//image = cv::Mat(480, 640, CV_8U, u8_data);
	
	//cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Display Image", image);
    
    image = cv::Mat(480, 640, CV_16U, avg_data);
    
    avg_fp = fopen("template1.img", "wb");
    fwrite(avg_data, 2, 480*640, avg_fp);
    fclose(avg_fp);
    
    cv::putText(image,min_count_str,cv::Point(2,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,0),2);
    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    depth_view->show("Display Image", image);
    //cv::imshow("Display Image", image);
    //cv::waitKey(10);
}

int main(int argc, char* argv[])
{
    const char* IP = NULL;
    const char* ID = NULL;
    TY_DEV_HANDLE hDevice;
    cv::Mat medium_resut1,medium_resut2,newDepth,blackDepth,dst_Depth,single_blackDepth,all_blackDepth;
    cv::Mat resized_color;
    DepthRender _render;
    int key_value = 0;
    FILE *fp;
    uint16_t t_data[480*640];
    bool m_display = false;

    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-id") == 0){
            ID = argv[++i];
        }else if(strcmp(argv[i], "-ip") == 0){
            IP = argv[++i];
        }else if(strcmp(argv[i], "-h") == 0){
            LOGI("Usage: SimpleView_Callback [-h] [-ip <IP>]");
            return 0;
        }
    }
    
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

    if(IP) {
        LOGD("=== Open device %s", IP);
        ASSERT_OK( TYOpenDeviceWithIP(IP, &hDevice) );
    } else {
        if(ID == NULL){
            LOGD("=== Get device info");
            int n;
            ASSERT_OK( TYGetDeviceNumber(&n) );
            LOGD("     - device number %d", n);

            TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
            ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

            if(n == 0){
                LOGD("=== No device got");
                return -1;
            }
            ID = pBaseInfo[0].id;
        }

        LOGD("=== Open device: %s", ID);
        ASSERT_OK( TYOpenDevice(ID, &hDevice) );
    }

#ifdef DEVELOPER_MODE
    LOGD("=== Enter Developer Mode");
    ASSERT_OK(TYEnterDeveloperMode(hDevice));
#endif

    int32_t allComps;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );
    if(allComps & TY_COMPONENT_RGB_CAM){
        LOGD("=== Has RGB camera, open RGB cam");
        ASSERT_OK( TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM) );
    }

    LOGD("=== Configure components, open depth cam");
    int32_t componentIDs = TY_COMPONENT_DEPTH_CAM | TY_COMPONENT_IR_CAM_LEFT | TY_COMPONENT_IR_CAM_RIGHT;
    // int32_t componentIDs = TY_COMPONENT_DEPTH_CAM;
    // int32_t componentIDs = TY_COMPONENT_DEPTH_CAM | TY_COMPONENT_IR_CAM_LEFT;
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );

    int err = TYSetEnum(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
    ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);

    LOGD("=== Prepare image buffer");
    int32_t frameSize;
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    ASSERT( frameSize >= 640*480*2 );

    LOGD("     - Allocate & enqueue buffers");
    char* frameBuffer[2];
    frameBuffer[0] = new char[frameSize];
    frameBuffer[1] = new char[frameSize];
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize) );
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize) );

    LOGD("=== Register frame callback");
    LOGD("Note:  user should call TYEnqueueBuffer to re-enqueue frame buffer.");
    DepthRender render;
    CallbackData cb_data;
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    cb_data.saveFrame = false;
    cb_data.saveIdx = 0;
    ASSERT_OK( TYRegisterCallback(hDevice, frameCallback, &cb_data) );

    LOGD("=== Register event callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    ASSERT_OK(TYRegisterEventCallback(hDevice, eventCallback, NULL));

    LOGD("=== Disable trigger mode");
    ASSERT_OK( TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false) );

    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );

    LOGD("=== Wait for callback");
    bool exit_main = false;
    DepthViewer depthViewer;

#ifndef TEMPLATE_DEBUG
    /* 打开文件用于读写 */
	fp = fopen("template.img", "r+");
	/* 查找文件的开头 */
   	fseek(fp, SEEK_SET, 0);
   	/* 读取并显示数据 */
	fread(t_data, 2, 480*640, fp);
	
	fclose(fp);
#endif
    
    while(!exit_main){
        while(fakeLock){
            MSLEEP(10);
        }
        fakeLock = true;

        if(!cb_data.depth.empty()){
            depthViewer.show("depth", cb_data.depth);
            
            if(m_display)
            {
            	//show_template(&depthViewer,(cb_data.saveIdx-1));
            	show_template_origin(&depthViewer,(cb_data.saveIdx-1));
            	m_display = false;
            }
            //cv::GaussianBlur(cb_data.depth, medium_resut, cv::Size(11, 11), 0);
			//depthViewer.show("depth_Gaussian11", medium_resut);
			//cv::GaussianBlur(cb_data.depth, medium_resut, cv::Size(3, 3), 0);
			//depthViewer.show("depth_Gaussian3", medium_resut);
#ifndef TEMPLATE_DEBUG	
			depthTransfer(cb_data.depth, t_data, &newDepth, &blackDepth);
			depthViewer.show("newDepth", newDepth);
			depthViewer.show("blackDepth", blackDepth);
			
			cv::GaussianBlur(newDepth, medium_resut1, cv::Size(3, 3), 0);
			//depthViewer.show("newDepth_Gaussian3", medium_resut1);
			
			//cv::GaussianBlur(blackDepth, medium_resut2, cv::Size(3, 3), 0);
			cv::medianBlur(blackDepth,medium_resut2,5);
			depthViewer.show("blackDepth_medium5", medium_resut2);
			
			combineDepth(medium_resut1, medium_resut2, &dst_Depth);
			cv::Rect rect(70, 10, 565, 450);
			dst_Depth = dst_Depth(rect);
			depthViewer.show("combineDepth", dst_Depth);
			
			if(!cb_data.color.empty())
			{
				cv::imshow("color", cb_data.color);
				//cv::resize(dst_Depth, dst_Depth, dst_Depth.size(), 0, 0, 0);
				cv::Rect color_rect(165, 0, 1025, 840);
				cv::resize(cb_data.color(color_rect), resized_color, dst_Depth.size());
				cv::imshow("resized_color", resized_color);
				//check_Black_Depth(dst_Depth,resized_color,&single_blackDepth,&all_blackDepth);
				//depthViewer.show("single_blackDepth", single_blackDepth);
				//depthViewer.show("all_blackDepth", all_blackDepth);
				
				cv::cvtColor(resized_color, grayImage, cv::COLOR_BGR2GRAY);
				cv::blur(grayImage, grayImage, cv::Size(7, 7));
				//cv::imshow("grayImage", grayImage);
				//check_Black_Depth(dst_Depth,grayImage,&single_blackDepth,&all_blackDepth);
				//depthViewer.show("single_blackDepth", single_blackDepth);
				//depthViewer.show("all_blackDepth", all_blackDepth);
				
				Canny(grayImage, out_Canny, min_Thresh, max_Thresh *2, 3);
				cv::Mat gray2Color;
				cv::cvtColor(out_Canny, gray2Color, cv::COLOR_GRAY2BGR);
				cv::imshow("gray2Color", gray2Color);
				//Find_Draw_COntours(0, 0);
				//resized_color = resized_color(rect);
				cv::Mat depthColor = _render.Compute(dst_Depth);
				
				depthColor = depthColor / 2 + gray2Color / 2;
				cv::imshow("depth_color", depthColor);
			}
#endif
        }
        if(!cb_data.point3D.empty())
        {
        	//cv::imshow("point3D", cb_data.point3D);
        	//depthViewer.show("point3D", cb_data.point3D);
        }
        if(!cb_data.leftIR.empty()){
            //cv::imshow("LeftIR", cb_data.leftIR);
        }
        if(!cb_data.rightIR.empty()){
            //cv::imshow("RightIR", cb_data.rightIR);
        }
        if(!cb_data.color.empty()){
            cv::imshow("color", cb_data.color);
            /*
            cv::namedWindow("window1",1);
			cv::imshow("window1", cb_data.color);
			
			//cv::Rect color_rect(165, 0, 1025, 840);
			//cv::resize(cb_data.color(color_rect), resized_color, cb_data.depth.size());

			cv::cvtColor(cb_data.color, grayImage, cv::COLOR_BGR2GRAY);
			//cv::imshow("gray_image", grayImage);
			cv::blur(grayImage, grayImage, cv::Size(7, 7));
			
			cv::createTrackbar("CANNY 值：", "window1", &min_Thresh, max_Thresh, Find_Draw_COntours);
			Find_Draw_COntours(0, 0);
			*/
        }

        if(cb_data.saveFrame && !cb_data.depth.empty() && !cb_data.leftIR.empty() && !cb_data.rightIR.empty()){
            LOGI(">>>> save frame %d", cb_data.saveIdx);
            char f[32];
#ifndef TEMPLATE_DEBUG
            sprintf(f, "%c-%d-d.img", key_value,cb_data.saveIdx);
            FILE* fp = fopen(f, "wb");
            fwrite(cb_data.depth.data, 2, cb_data.depth.size().area(), fp);
            
            fclose(fp);
            
            sprintf(f, "%c-%d-c.img", key_value,cb_data.saveIdx);
            FILE* fpc = fopen(f, "wb");
            //fwrite(medium_resut.data, 2, cb_data.depth.size().area(), fp);
            fwrite(cb_data.color.data, 3, cb_data.color.size().area(), fp);

            // fwrite(cb_data.leftIR.data, 1, cb_data.leftIR.size().area(), fp);
            // fwrite(cb_data.rightIR.data, 1, cb_data.rightIR.size().area(), fp);
            fclose(fpc);
#endif
#ifdef TEMPLATE_DEBUG
			sprintf(f, "depth_img1/%d.img", cb_data.saveIdx++);
			FILE* fp = fopen(f, "wb");
			/*
			cv::Rect rect(70, 10, 565, 450);
			cv::Mat temp_Depth;
			//temp_Depth = cb_data.depth(rect);
			cv::resize(cb_data.depth(rect), temp_Depth, cv::Size(565,450));
			cv::medianBlur(temp_Depth,temp_Depth,5);
			fwrite(temp_Depth.data, 2, temp_Depth.size().area(), fp);
			*/
			
			fwrite(cb_data.depth.data, 2, cb_data.depth.size().area(), fp);
            
            fclose(fp);
#endif
            cb_data.saveFrame = false;
        }

        fakeLock = false;

        int key = cv::waitKey(10);
        switch(key & 0xff){
            case 0xff:
                break;
            case 'q':
                exit_main = true;
                break;
            case 's':
                cb_data.saveFrame = true;
                break;
            case 'c':
            	//show_template(cb_data.saveIdx-1);
            	m_display = true;
            	break;
            default:
#ifndef TEMPLATE_DEBUG
            	if(key == key_value)
            	{
            		cb_data.saveIdx++;
            	}
            	else
            	{
            		key_value = key;
            		cb_data.saveIdx = 0;
            	}
#endif
                LOGD("Unmapped key %d", key);
        }

#ifdef DEVELOPER_MODE
        DEVELOPER_MODE_PRINT();
#endif
    }

    ASSERT_OK( TYStopCapture(hDevice) );
    ASSERT_OK( TYCloseDevice(hDevice) );
    ASSERT_OK( TYDeinitLib() );
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");
    return 0;
}

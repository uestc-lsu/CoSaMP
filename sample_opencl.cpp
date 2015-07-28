#include "time.h"
#include <CL/cl.h>
#include <fstream>
#include <string.h>
#include "head.h"
#include <iostream>
using namespace std;
#define TS 32
#define WPTM 2
#define WPTN 2
void ck(cl_int ret,const char *mes)
{
	if(ret != CL_SUCCESS)
	{
		cout<<mes<<" error,code is:"<<ret<<endl;
	}
	return ;
}
bool GetFileData(const char *path,string & str)
{
	FILE *fp = fopen(path,"r");
	if(fp == NULL)
	{
		cout<<"Open file error."<<endl;
		return false;
	}
	while(feof(fp) == false)
	{
		str+=fgetc(fp);
	}
	return true;
}
extern "C" void setMatrix_async(int m,int n,double const* hA_src,int ldha,cl_mem dB_dst,size_t dB_offset,int lddb,cl_command_queue queue,cl_event *event)
{
	 if (m <= 0 || n <= 0)
        return;

    	size_t buffer_origin[3] = { dB_offset*sizeof(double), 0, 0 };
    	size_t host_orig[3]     = { 0, 0, 0 };
    	size_t region[3]        = { n*sizeof(double), m, 1 };
    	cl_int err = clEnqueueWriteBufferRect(
        	queue, dB_dst,  CL_FALSE,  // non-blocking
        	buffer_origin, host_orig, region,
       		lddb*sizeof(double), 0,
        	ldha*sizeof(double), 0,
        	hA_src, 0, NULL, event );
	clFlush(queue);
    	ck(err,"clEnqueueWriteBufferRect");
}
extern "C" void getMatrix_asyc(int m,int n,cl_mem dA_src,size_t dA_offset,int ldda,double *hB_dst,int ldhb,cl_command_queue queue,cl_event *event)
{
	 if (m <= 0 || n <= 0)
        return;

    	size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
    	size_t host_orig[3]     = { 0, 0, 0 };
    	size_t region[3]        = { n*sizeof(double), m, 1 };
    	cl_int err = clEnqueueReadBufferRect(
        	queue, dA_src, CL_FALSE,  // non-blocking
       		buffer_origin, host_orig, region,
        	ldda*sizeof(double), 0,
        	ldhb*sizeof(double), 0,
        	hB_dst, 0, NULL, event);
    	clFlush(queue);
   	ck(err,"clEnqueueReadBufferRect");
}
void MatrixMulGPU(double *A,double *B,double *C,int m,int n,int p)
{

	cl_int ret;
	cl_uint numPlatform;
	cl_platform_id *platform;    
	cl_uint device_num;
	cl_device_id device;                                                                      
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;
	cl_kernel kernel[2];

	cl_event prof_event;
	cl_ulong ev_start_time = (cl_ulong)0; 
	cl_ulong ev_end_time = (cl_ulong)0;
	cl_ulong run_time = (cl_ulong)0;
	cl_event evt;

	ret = clGetPlatformIDs(0, NULL, &numPlatform);
	ck(ret,"platform");
	platform=(cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatform);
	ret =clGetPlatformIDs(numPlatform, platform, NULL);                                             
	ck(ret,"platform");

	ret = clGetDeviceIDs(platform[1],CL_DEVICE_TYPE_GPU,1,&device,&device_num);         
	ck(ret,"device");
	context = clCreateContext(NULL,1,&device,NULL,NULL,&ret);                      
	ck(ret,"context");
	cmd_queue = clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,&ret);
	ck(ret,"cmd_queue");
	
	cl_mem dev_A,dev_B,dev_BRT,dev_C;
	char *code;
	string str;
	if(GetFileData("./simpleMultiply.cl",str) == false)  
	{
		cout<<"CL File Error."<<endl;
	}
	code = new char[str.size()];
	strcpy(code,str.c_str());
	code[str.size()-1] = NULL;
	program = clCreateProgramWithSource(context,1,(const char **)&code,NULL,&ret); 
	ck(ret,"Create program with source");
	ret = clBuildProgram(program,1,&device,NULL,NULL,NULL);                             
	ck(ret,"build");
	kernel[0] = clCreateKernel(program,"transpose",&ret);
	ck(ret,"create kernel");
	kernel[1] = clCreateKernel(program,"MatrixMult",&ret);
	ck(ret,"create kernel");
	int lda=((p+31)/32)*32;
	int ldb=((n+31)/32)*32;
	int ldbt=((p+31)/32)*32;
	int ldc=((n+31)/32)*32;
	int szA = m * lda;  
        int szB = p * ldb;  
	int szBT= n * ldbt;
    	int szC = m * ldc; 
	dev_A= clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(double)*szA,NULL,&ret);	
	ck(ret,"create buffer A");
	dev_B = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(double)*szB,NULL,&ret);
	ck(ret,"create buffer B");
	dev_BRT = clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(double)*szBT,NULL,&ret);
	ck(ret,"create buffer B");
	dev_C= clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(double)*szC,NULL,&ret);
	ck(ret,"create buffer C");
	/*************************         数据写入OpenCL设备            ************************/
	setMatrix_async(m,p,A,p,dev_A,0,lda,cmd_queue,NULL);
	setMatrix_async(p,n,B,n,dev_B,0,ldb,cmd_queue,NULL);
	
	size_t global[2];
	size_t local[2]={16,16};
	global[0]=(size_t) (n+local[0]-1)/local[0];
	global[0]*=local[0];
	global[1]=(size_t) (p+local[1]-1)/local[1];
	global[1]*=local[1];

	ret  = clSetKernelArg(kernel[0], 0, sizeof(int), &p);  
  	ret |= clSetKernelArg(kernel[0], 1, sizeof(int), &n);  
   	ret |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &dev_B);  
	ret |= clSetKernelArg(kernel[0], 3, sizeof(int), &ldb); 
   	ret |= clSetKernelArg(kernel[0], 4, sizeof(cl_mem), &dev_BRT);  
	ret |= clSetKernelArg(kernel[0], 5, sizeof(int), &ldbt); 
	ck(ret,"clSetKernelArg");
	ret = clEnqueueNDRangeKernel(cmd_queue,kernel[0],2,NULL,global,local,0,NULL,NULL);  
	ck(ret,"GPU NDRange");
	clFinish(cmd_queue);

	local[0]=TS/WPTN;
	local[1]=TS/WPTM;
	global[0]=(size_t) (n/WPTN+local[0]-1)/local[0];
	global[0]*=local[0];
	global[1]=(size_t) (m/WPTM+local[1]-1)/local[1];
	global[1]*=local[1];

	ret  = clSetKernelArg(kernel[1], 0, sizeof(int), &m);  
  	ret |= clSetKernelArg(kernel[1], 1, sizeof(int), &n);  
  	ret |= clSetKernelArg(kernel[1], 2, sizeof(int), &p);  
   	ret |= clSetKernelArg(kernel[1], 3, sizeof(cl_mem), &dev_A); 
	ret |= clSetKernelArg(kernel[1], 4, sizeof(int), &lda);  
   	ret |= clSetKernelArg(kernel[1], 5, sizeof(cl_mem), &dev_BRT);  
	ret |= clSetKernelArg(kernel[1], 6, sizeof(int), &ldbt); 
   	ret |= clSetKernelArg(kernel[1], 7, sizeof(cl_mem), &dev_C);  
	ret |= clSetKernelArg(kernel[1], 8, sizeof(int), &ldc);  
	ck(ret,"clSetKernelArg");
	ret = clEnqueueNDRangeKernel(cmd_queue,kernel[1],2,NULL,global,local,0,NULL,NULL);  
	ck(ret,"GPU NDRange");
	clFinish(cmd_queue);

	getMatrix_asyc(m,n,dev_C,0,ldc,C,n,cmd_queue,NULL);

	clReleaseContext(context);
	clReleaseCommandQueue(cmd_queue);
	clReleaseProgram(program);
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseMemObject(dev_A);
	clReleaseMemObject(dev_B);
	clReleaseMemObject(dev_BRT);
	clReleaseMemObject(dev_C);
}

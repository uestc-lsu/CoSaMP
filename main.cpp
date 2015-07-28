#include <stdlib.h>
#include "head.h"
#include <time.h>
#include <math.h>
#include "string.h"
#include <iomanip>
#include<sys/time.h>
#include <iostream>
using namespace std;
typedef unsigned char BYTE;

void Usage()
{
	printf(
			"%s",
              "Usage: CS <-i> inputfilepath <-o> IDWToutputfilenamepath  <-l> Wavelet layer <s> compressionradio <-r> image block size\n"
              "\n"
			  "  -i: the input the GeoTiff image path.\n"
			  "  -o: the ouput the reconsitution image path.\n"
              "  -l: wavelet layer.\n"
              "  -s: sparse image compression radio. \n"
              "  -r: image block size.\n"
		);        
    exit(1);
}

int main(int argc, char* argv[])
{
	int i,j,x,y,h,l,x_block,y_block;
	double duration[5]={0};    
	struct timeval point[10];
	
	char *readPath=NULL; //Read image path
	char *writePath=NULL; //write image path
	unsigned char *hehe=NULL;
	unsigned char **pImageData=&hehe; //read image data
	int width; //image size
	int height;
	int size=-1;

	int nChannels=1; //image channel
	double *adfGeoTransform=NULL; 
	char *pszSrcWKT =NULL; 
	
	double *InputData=NULL; 
	double *OutputData=NULL;
	int nLayer =4; //DWT number
	double *ImageData=NULL; 
	double *SparseMat=NULL; 
	float radio=-1; //image sample radio
	int m; 
	
	double *GaussMat=NULL; 
	double *SampleMat=NULL;
	double *RecMat=NULL; 
	double s,u,k,g;
	double Mse=0;
	double fMse = 0;
    double fPsnr = 0; 
	double error=0; 
	double match=0; 

	for( i = 1; i < argc; i++ )
        {
		if( strcmp(argv[i],"-i") == 0)
		       	readPath = argv[++i];
           	else if( strcmp(argv[i],"-o") == 0)
				writePath= argv[++i];
           	else if(strcmp(argv[i],"-l") == 0)
                	nLayer=atoi(argv[++i]);
            else if(strcmp(argv[i],"-s") == 0)
                	radio=atof(argv[++i]);
           	else  if(strcmp(argv[i],"-r")==0)
                	size=atoi(argv[++i]);
            else
            {
                Usage();
            }
        }

	
	if(readPath==NULL || writePath ==NULL || radio <= 0 || nLayer<=0 || size<=0)
		Usage();

	int bmpWidth=size; //block image size
	int bmpHeight=size;
	adfGeoTransform=new double[6];
	pszSrcWKT= new char[1000];
	//read image information
	gettimeofday(&point[0],NULL);
	readImageGDAL(pImageData,width,height,nChannels,readPath,adfGeoTransform,pszSrcWKT);
	gettimeofday(&point[1],NULL);
	
	InputData= new double[width* height];
	OutputData= new double[width* height];

	x_block=(height+(bmpHeight-1))/bmpHeight;
	y_block=(width+(bmpWidth-1))/bmpWidth;

	for(h=0;h<x_block;h++)
		for(l=0;l<y_block;l++){
		ImageData= new double[bmpWidth* bmpHeight];
		SparseMat= new double[bmpWidth* bmpHeight];
		for( i=h*bmpHeight;i<(h+1)*bmpHeight;i++)
			for(j=l*bmpWidth;j<(l+1)*bmpWidth;j++){
				InputData[i*width+j]=(double)(*pImageData)[i*width+j];
				ImageData[(i-h*bmpHeight)*bmpWidth+(j-l*bmpWidth)]=(double)(*pImageData)[i*width+j]; 
			}

		//Use DWT to sparse image 
		filterset(p_t);
		gettimeofday(&point[2],NULL);
		DwtND(ImageData,bmpHeight,bmpWidth,nLayer);	
		gettimeofday(&point[3],NULL);
		for( y=0; y<bmpHeight; y++)
			for( x=0; x<bmpWidth; x++)
				SparseMat[y*bmpWidth+x]=ImageData[y*bmpWidth+x];

		//image sampling
		m=radio*bmpWidth;
		GaussMat=new double[m* bmpHeight];
		SampleMat=new double[m* bmpWidth];
		gettimeofday(&point[4],NULL);
        CompressSample(SparseMat, m, bmpWidth, bmpHeight,GaussMat,SampleMat);
		gettimeofday(&point[5],NULL);
		cout<<"Compress sample success!"<<endl;

		//image reconstruction
		RecMat=new double[bmpWidth*bmpHeight];
		gettimeofday(&point[6],NULL);
		//use CoSaMP algorithm to reconstruction sampled data
		CosampMat(SampleMat,GaussMat,bmpWidth,bmpHeight,m,RecMat);
		gettimeofday(&point[7],NULL);
		cout<<"CoSaMP seccess!"<<endl;

		//IDWT and restore image
		gettimeofday(&point[8],NULL);
		IDwtND(RecMat, bmpHeight,bmpWidth, nLayer);
		gettimeofday(&point[9],NULL);
		for( y=h*bmpHeight;y<(h+1)*bmpHeight;y++)
			for(x=l*bmpWidth;x<(l+1)*bmpWidth;x++){ 
				(*pImageData)[y*width+x]=(unsigned char)RecMat[(y-h*bmpHeight)*bmpWidth+(x-l*bmpWidth)];
				OutputData[y*width+x]=RecMat[(y-h*bmpHeight)*bmpWidth+(x-l*bmpWidth)];
          	}
	
		duration[0] = (point[1].tv_sec - point[0].tv_sec) + 1.0e-6*(point[1].tv_usec - point[0].tv_usec);
		duration[1] = (point[3].tv_sec - point[2].tv_sec) + 1.0e-6*(point[3].tv_usec - point[2].tv_usec);
		duration[2] = (point[5].tv_sec - point[4].tv_sec) + 1.0e-6*(point[5].tv_usec - point[4].tv_usec);
		duration[3] = (point[7].tv_sec - point[6].tv_sec) + 1.0e-6*(point[7].tv_usec - point[6].tv_usec);
		duration[4] = (point[9].tv_sec - point[8].tv_sec) + 1.0e-6*(point[9].tv_usec - point[8].tv_usec);
		
	}
	//save the restored image
	WriteImageGDAL(writePath ,*pImageData,width,height,nChannels,adfGeoTransform,pszSrcWKT);
	//output the time of each part of CS
	cout<<"Image read = "<<duration[0]<<"s"<<endl;
    cout<<"DWT time = "<<duration[1]<<"s"<<endl;
    cout<<"Sample time = "<<duration[2]<<"s"<<endl;
    cout<<"OMP time = "<<duration[3]<<"s"<<endl;
    cout<<"IDWT time = "<<duration[4]<<"s"<<endl;
	
   	//evaluation index,MSE,PSNR,ERROR,MATCH
    for(y=0;y<height; y++){
       	for(x=0; x<width; x++){
            Mse += (OutputData[y*width+x] - InputData[y*width+x])*(OutputData[y*width+x] - InputData[y*width+x]);
       	}
    }
    fMse =Mse /(height*width);
    fPsnr = 10*log10(255*255/fMse);
	
	u=0;s=0;
	for(y=0;y<height; y++){		
		for(x=0; x<width; x++)
			s+=OutputData[y*width+x]*OutputData[y*width+x];
	}
	s=sqrt(s);
	u=sqrt(Mse);
	error=u/s;

	for(y=0;y<height; y++){
        for(x=0; x<width; x++){		
			k+=(abs(OutputData[y*width+x])-abs(InputData[y*width+x]))*(abs(OutputData[y*width+x])-abs(InputData[y*width+x]));
			g+=(abs(OutputData[y*width+x])+abs(InputData[y*width+x]))*(abs(OutputData[y*width+x])+abs(InputData[y*width+x]));
		}
	}
	k=sqrt(abs(k));
	g=sqrt(abs(g));
	cout<<"k="<<k<<endl;
	cout<<"l="<<g<<endl;
	match=1-(k/g);

	cout<<"MSE="<<fMse<<endl;
	cout<<"psnr="<<fPsnr<<endl;
	cout<<"error="<<error<<endl;
	cout<<"match="<<match<<endl;
	
	//free memory
	delete []adfGeoTransform;
	delete []pszSrcWKT;
	delete []ImageData;
	delete []SparseMat;
	delete []InputData;
	delete []OutputData;
	delete []GaussMat;
	delete []SampleMat;
	delete []RecMat;
	return 0;
}

